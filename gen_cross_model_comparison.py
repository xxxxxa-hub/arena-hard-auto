import argparse
import json
import os
import re
import time
import concurrent.futures
import itertools
import numpy as np
import random
import threading

import tiktoken
import shortuuid
import tqdm

from utils.add_markdown_info import count_markdown_elements, remove_pattern
from utils.completion import (
    load_questions,
    load_model_answers,
    make_config,
    get_endpoint,
    registered_api_completion,
    registered_engine_completion,
    reorg_answer_file,
    API_ERROR_OUTPUT,
)


# Global lock for thread-safe file writing
file_write_lock = threading.Lock()

def get_single_response(question: dict, model_name: str, response_idx: int, settings: dict, answer_file: str):
    """Generate a single response for a question - used in parallel processing"""
    # build messages
    messages = []
    if "sys_prompt" in settings:
        messages.append({"role": "system", "content": settings["sys_prompt"]})
        
    messages.append({"role": "user", "content": question["prompt"]})
    
    # retrieve the api completion function from register
    api_completion_func = registered_api_completion[settings["api_type"]]
    
    # build arguments for api completions
    kwargs = settings | {
        "api_dict": get_endpoint(settings["endpoints"]),
        "messages": messages,
    }
    
    output = api_completion_func(**kwargs)
    
    if output is API_ERROR_OUTPUT:
        print(f"API error for {model_name} response {response_idx+1} question {question['uid']}, skipping...")
        return
    
    # Create answer entry
    ans = {
        "uid": question["uid"],
        "ans_id": shortuuid.uuid(),
        "response_idx": response_idx,
        "model": model_name,
        "messages": messages + [{"role": "assistant", "content": output}],
        "tstamp": time.time(),
    }
    
    # Add metadata
    encoding = tiktoken.encoding_for_model("gpt-4o")
    metadata = {
        "token_len": len(encoding.encode(output['answer'], disallowed_special=()))
    }
    ans["metadata"] = metadata | count_markdown_elements(
        remove_pattern(
            output['answer'], 
            re.compile("```([^`]*)```")
        ),
        suffix="",
    )
    
    # Thread-safe file writing
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with file_write_lock:
        with open(answer_file, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(ans, ensure_ascii=False) + "\n")
    
    return ans


def get_multiple_answers_for_model(
    questions: list, model_name: str, settings: dict, num_responses: int = 8, output_dir: str = None
):
    """Generate multiple responses for all questions from a specific model using parallel processing"""
    
    answer_file = os.path.join(output_dir, f"{model_name}_multi_responses.jsonl")
    print(f"Generating {num_responses} responses per question for {model_name}")
    print(f"Output to {answer_file}")
    
    # Clear the output file at the beginning
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "w", encoding="utf-8") as f:
        pass  # Just clear the file
    
    # Get parallel workers setting
    parallel = settings.get("parallel", 1)
    print(f"Using {parallel} parallel workers")
    
    # Create list of all (question, response_idx) pairs for parallel processing
    tasks = []
    for question in questions:
        for response_idx in range(num_responses):
            tasks.append((question, response_idx))
    
    total_tasks = len(tasks)
    print(f"Total API calls to make: {total_tasks}")
    
    all_responses = []
    
    # Process in parallel using ThreadPoolExecutor (same as gen_answer.py)
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = []
        
        for question, response_idx in tasks:
            future = executor.submit(
                get_single_response,
                question,
                model_name,
                response_idx,
                settings,
                answer_file
            )
            futures.append(future)
        
        # Process completed futures with progress bar
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), 
            total=len(futures),
            desc=f"Generating {model_name} responses"
        ):
            result = future.result()
            if result:
                all_responses.append(result)
    
    print(f"Generated {len(all_responses)} responses for {model_name}")
    return all_responses


def cross_model_minimax_selection(
    your_model_responses: list, 
    baseline_responses: list, 
    reward_model_func,
    question: dict
):
    """
    Select the best response from your model using minimax against baseline responses
    
    Args:
        your_model_responses: List of responses from your model
        baseline_responses: List of responses from baseline model
        reward_model_func: Function that takes two responses and returns P(response1 > response2)
        question: Question dictionary
    
    Returns:
        Best response from your model according to minimax against baseline
    """
    n_your = len(your_model_responses)
    n_baseline = len(baseline_responses)
    
    if n_your == 0:
        return None, None, []
    
    if n_baseline == 0:
        # If no baseline responses, just return the first your_model response
        return your_model_responses[0], np.array([]), [1.0] * n_your
    
    print(f"Computing cross-model comparisons: {n_your} vs {n_baseline} responses")
    
    # Create comparison matrix: your_model_responses vs baseline_responses
    # comparison_matrix[i][j] = P(your_model_response_i > baseline_response_j)
    comparison_matrix = np.zeros((n_your, n_baseline))
    
    for i in range(n_your):
        for j in range(n_baseline):
            # Get probability that your model response i is better than baseline response j
            prob_your_better = reward_model_func(
                your_model_responses[i], 
                baseline_responses[j], 
                question
            )
            comparison_matrix[i][j] = prob_your_better
    
    # Minimax: for each of your responses, find its worst-case performance against baseline
    worst_case_scores = []
    
    for i in range(n_your):
        # For your response i, find minimum probability of winning against any baseline response
        worst_case = min(comparison_matrix[i][j] for j in range(n_baseline))
        worst_case_scores.append(worst_case)
    
    # Select your response with highest worst-case score
    best_idx = np.argmax(worst_case_scores)
    
    print(f"Cross-model minimax selection: Your response {best_idx} with worst-case score {worst_case_scores[best_idx]:.3f}")
    print(f"All worst-case scores: {[f'{score:.3f}' for score in worst_case_scores]}")
    
    return your_model_responses[best_idx], comparison_matrix, worst_case_scores


def dummy_reward_model(response1, response2, question):
    """
    Dummy reward model for testing - replace with your actual reward model
    """
    import random
    # Simple heuristic: longer responses are slightly preferred, but with randomness
    len1 = len(response1["messages"][-1]["content"]["answer"])
    len2 = len(response2["messages"][-1]["content"]["answer"])
    
    base_prob = 0.5 + 0.1 * (len1 - len2) / max(len1, len2, 1)
    noise = random.uniform(-0.2, 0.2)
    return max(0.1, min(0.9, base_prob + noise))


def load_reward_model_func(reward_model_config):
    """
    Load your actual reward model function here
    This should return a function that takes two responses and returns P(response1 > response2)
    """
    # TODO: Implement loading of your actual reward model
    print("Warning: Using dummy reward model. Replace with your actual reward model.")
    return dummy_reward_model


def filter_and_sample_questions(questions, subcategory=None, sample_size=None, random_seed=42):
    """
    Filter questions by subcategory and sample a subset
    
    Args:
        questions: List of question dictionaries
        subcategory: Subcategory to filter by (None for all subcategories)
        sample_size: Number of questions to sample (None for all questions)
        random_seed: Random seed for reproducible sampling
        
    Returns:
        List of filtered and sampled questions
    """
    # Set random seed for reproducible results
    random.seed(random_seed)
    
    # Filter by subcategory if specified
    if subcategory is not None:
        filtered_questions = [q for q in questions if q.get("subcategory") == subcategory]
        print(f"Filtered to {len(filtered_questions)} questions in subcategory '{subcategory}'")
    else:
        filtered_questions = questions
        print(f"Using all {len(filtered_questions)} questions from all subcategories")
    
    # Sample if size specified
    if sample_size is not None and sample_size < len(filtered_questions):
        sampled_questions = random.sample(filtered_questions, sample_size)
        print(f"Randomly sampled {len(sampled_questions)} questions")
    else:
        sampled_questions = filtered_questions
        if sample_size is not None:
            print(f"Sample size ({sample_size}) >= available questions ({len(filtered_questions)}), using all")
    
    # Show subcategory distribution
    subcategory_counts = {}
    for q in sampled_questions:
        subcat = q.get("subcategory", "unknown")
        subcategory_counts[subcat] = subcategory_counts.get(subcat, 0) + 1
    
    print(f"Final question distribution:")
    for subcat, count in sorted(subcategory_counts.items()):
        print(f"  {subcat}: {count} questions")
    
    return sampled_questions


def group_responses_by_question(responses):
    """Group responses by question UID"""
    grouped = {}
    for response in responses:
        uid = response["uid"]
        if uid not in grouped:
            grouped[uid] = []
        grouped[uid].append(response)
    return grouped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multiple responses for cross-model comparison")
    parser.add_argument(
        "--config-file", type=str, default="config/gen_answer_config.yaml",
        help="Config file for answer generation"
    )
    parser.add_argument(
        "--endpoint-file", type=str, default="config/api_config.yaml",
        help="API endpoint configuration file"
    )
    parser.add_argument(
        "--your-model", type=str, required=True,
        help="Your model name (must be in endpoint config)"
    )
    parser.add_argument(
        "--baseline-model", type=str, required=True,
        help="Baseline model name (must be in endpoint config)"
    )
    parser.add_argument(
        "--num-responses", type=int, default=8,
        help="Number of responses to generate per question per model"
    )
    parser.add_argument(
        "--reward-model-config", type=str, default=None,
        help="Path to reward model configuration"
    )
    parser.add_argument(
        "--subcategory", type=str, default=None,
        help="Question subcategory to filter by (e.g., 'coding', 'creative_writing', 'math'). Default: use all subcategories"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Number of questions to randomly sample from the subcategory. Default: use all questions"
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)"
    )
    args = parser.parse_args()

    config = make_config(args.config_file)
    endpoints = make_config(args.endpoint_file)

    # Verify models exist in endpoints
    assert args.your_model in endpoints, f"Your model '{args.your_model}' not found in endpoints"
    assert args.baseline_model in endpoints, f"Baseline model '{args.baseline_model}' not found in endpoints"

    # Load and filter questions
    question_file = os.path.join("data", config["bench_name"], "question.jsonl")
    all_questions = load_questions(question_file)
    
    # Filter and sample questions
    questions = filter_and_sample_questions(
        all_questions, 
        subcategory=args.subcategory, 
        sample_size=args.sample_size,
        random_seed=args.random_seed
    )

    # Create output directory
    output_dir = os.path.join("data", config["bench_name"], "cross_model_comparison")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Cross-model comparison setup:")
    print(f"  Your model: {args.your_model}")
    print(f"  Baseline model: {args.baseline_model}")
    print(f"  Questions: {len(questions)}")
    print(f"  Responses per model per question: {args.num_responses}")
    print(f"  Total comparisons per question: {args.num_responses} x {args.num_responses} = {args.num_responses**2}")

    # Generate responses for your model
    print(f"\n{'='*60}")
    print(f"STEP 1: Generating responses for YOUR MODEL ({args.your_model})")
    print(f"{'='*60}")
    
    your_model_settings = endpoints[args.your_model]
    your_responses = get_multiple_answers_for_model(
        questions, args.your_model, your_model_settings, args.num_responses, output_dir
    )

    # Generate responses for baseline model
    print(f"\n{'='*60}")
    print(f"STEP 2: Generating responses for BASELINE MODEL ({args.baseline_model})")
    print(f"{'='*60}")
    
    baseline_model_settings = endpoints[args.baseline_model]
    baseline_responses = get_multiple_answers_for_model(
        questions, args.baseline_model, baseline_model_settings, args.num_responses, output_dir
    )

    # Group responses by question
    your_responses_by_question = group_responses_by_question(your_responses)
    baseline_responses_by_question = group_responses_by_question(baseline_responses)

    # Load reward model
    reward_model_func = load_reward_model_func(args.reward_model_config)

    # Run cross-model minimax selection
    print(f"\n{'='*60}")
    print(f"STEP 3: Running Cross-Model Minimax Selection")
    print(f"{'='*60}")

    best_responses_file = os.path.join(output_dir, f"{args.your_model}_vs_{args.baseline_model}_best.jsonl")
    comparison_info_file = os.path.join(output_dir, f"{args.your_model}_vs_{args.baseline_model}_comparison_info.jsonl")

    for question in tqdm.tqdm(questions, desc="Cross-model comparison"):
        uid = question["uid"]
        
        your_question_responses = your_responses_by_question.get(uid, [])
        baseline_question_responses = baseline_responses_by_question.get(uid, [])
        
        if len(your_question_responses) == 0:
            print(f"Warning: No responses from your model for question {uid}")
            continue
            
        if len(baseline_question_responses) == 0:
            print(f"Warning: No baseline responses for question {uid}")
            continue

        print(f"\nProcessing question {uid}")
        print(f"  Your model responses: {len(your_question_responses)}")
        print(f"  Baseline responses: {len(baseline_question_responses)}")

        # Run cross-model minimax
        best_response, comparison_matrix, worst_case_scores = cross_model_minimax_selection(
            your_question_responses, baseline_question_responses, reward_model_func, question
        )

        if best_response is None:
            continue

        # Save best response
        with open(best_responses_file, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(best_response, ensure_ascii=False) + "\n")

        # Save comparison information
        comparison_info = {
            "uid": uid,
            "your_model": args.your_model,
            "baseline_model": args.baseline_model,
            "num_your_responses": len(your_question_responses),
            "num_baseline_responses": len(baseline_question_responses),
            "selected_response_idx": best_response["response_idx"],
            "worst_case_scores": worst_case_scores,
            "comparison_matrix": comparison_matrix.tolist(),
            "tstamp": time.time()
        }

        with open(comparison_info_file, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(comparison_info, ensure_ascii=False) + "\n")

    # Reorganize the best responses file
    reorg_answer_file(best_responses_file)

    print(f"\n{'='*60}")
    print(f"CROSS-MODEL COMPARISON COMPLETED")
    print(f"{'='*60}")
    print(f"Best responses from {args.your_model}: {best_responses_file}")
    print(f"Comparison info: {comparison_info_file}")
    print(f"All responses saved in: {output_dir}")
