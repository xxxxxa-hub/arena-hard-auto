#!/usr/bin/env python3
"""
Generate judgment scores for minimax best responses
Converts minimax best responses to judgment format and evaluates them
"""

import json
import yaml
import argparse
import os
import concurrent.futures
from typing import Dict, List

from tqdm import tqdm

from utils.completion import (
    load_questions,
    registered_api_completion,
    load_model_answers,
    get_endpoint,
    make_config,
)

# Import filtering function from cross_model_reward_evaluation
from cross_model_reward_evaluation import filter_and_sample_questions

from utils.judge_utils import JUDGE_SETTINGS


def get_score(judgment, patterns):
    import re
    for pattern in patterns:
        pattern = re.compile(pattern)
        
        matches = pattern.findall(judgment.upper())
        matches = [m for m in matches if m != ""]
        
        if len(set(matches)) > 0:
            return matches[-1].strip("\n")
    return None


def pairwise_judgment(question, baseline, answer, reference, configs, settings):
    prompt_args = {
        "QUESTION": question['prompt'],
        "ANSWER_A": baseline["messages"][-1]["content"]['answer'],
        "ANSWER_B": answer["messages"][-1]["content"]['answer'],
    }
    
    if reference:
        prompt_args[f"REFERENCE"] = reference["messages"][-1]["content"]['answer']
        
    user_prompt = configs["prompt_template"].format(**prompt_args)
    messages = [
        {
            "role": "system", 
            "content": JUDGE_SETTINGS[question["category"]]["system_prompt"],
        },
        {
            "role": "user", 
            "content": user_prompt,
        }
    ]

    # build arguments for api completions
    kwargs = settings | {
        "api_dict": get_endpoint(settings["endpoints"]),
        "messages": messages,
    }
    kwargs['temperature'] = configs['temperature']
    kwargs['max_tokens'] = configs['max_tokens']
    
    api_completion_func = registered_api_completion[settings["api_type"]]
    output = api_completion_func(**kwargs)
    
    if output is None:
        return None

    score = get_score(output['answer'], configs["regex_patterns"])

    result = {
        "score": score,
        "judgment": output,
        "prompt": messages,
    }
    return result


def judgment(args):
    answer = args['answer']
    baseline = args['baseline']
    
    output = {
        "uid": args['question']["uid"],
        "category": args['question']["category"],
        "judge": args['configs']['judge_model'],
        "model": answer["model"],
        "baseline": baseline["model"],
        "games": []
    }

    # round 1
    result = pairwise_judgment(
        question=args['question'],
        baseline=baseline,
        answer=answer,
        reference=args['reference'],
        configs=args['configs'],
        settings=args['settings'],
    )
    output["games"].append(result)
        
    # round 2
    result = pairwise_judgment(
        question=args['question'],
        baseline=answer,
        answer=baseline,
        reference=args['reference'],
        configs=args['configs'],
        settings=args['settings'],
    )
    output["games"].append(result)

    with open(args['output_file'], "a", encoding="utf-8") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")


def load_minimax_best_responses(filepath: str) -> Dict[str, Dict]:
    """Load minimax best responses from .jsonl file"""
    responses = {}
    
    if not os.path.exists(filepath):
        print(f"Warning: Minimax best responses file not found: {filepath}")
        return responses
        
    print(f"Loading minimax best responses from: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            response = json.loads(line.strip())
            uid = response["uid"]
            
            # Reformat to match expected structure for gen_judgment.py
            formatted_response = {
                "uid": uid,
                "model": response["model"],  # Should be the your_model name
                "messages": response["messages"],
                "tstamp": response.get("tstamp", 0),
                "metadata": response.get("metadata", {})
            }
            
            responses[uid] = formatted_response
            
    print(f"Loaded {len(responses)} minimax best responses")
    return responses


def create_model_answers_structure(minimax_responses: Dict[str, Dict], model_name: str) -> Dict[str, Dict[str, Dict]]:
    """Convert minimax responses to model_answers structure expected by gen_judgment.py"""
    model_answers = {model_name: minimax_responses}
    return model_answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate judgment scores for minimax best responses")
    parser.add_argument("--setting-file", type=str, default="config/arena-hard-v2.0.yaml",
                       help="Path to judgment settings file")
    parser.add_argument("--endpoint-file", type=str, default="config/api_config.yaml",
                       help="Path to API endpoint configuration")
    parser.add_argument("--minimax-file", type=str, required=True,
                       help="Path to minimax best responses .jsonl file")
    parser.add_argument("--model-name", type=str, 
                       help="Model name for the minimax responses (auto-detected if not provided)")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for judgments (auto-generated if not provided)")
    
    # Filtering options (same as in other scripts)
    parser.add_argument("--subcategory", type=str,
                       help="Filter questions by subcategory")
    parser.add_argument("--sample-size", type=int,
                       help="Number of questions to sample")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for sampling")
    
    args = parser.parse_args()
    print(f"Arguments: {args}")

    # Load configurations
    configs = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)

    print(f'Judge model: {configs["judge_model"]}, reference: {configs["reference"]}, '
          f'temperature: {configs["temperature"]}, max tokens: {configs["max_tokens"]}')

    # Load questions
    question_file = os.path.join("data", configs["bench_name"], "question.jsonl")
    all_questions = load_questions(question_file)
    
    # Apply filtering (same logic as other scripts)
    if args.subcategory or args.sample_size:
        filtered_questions = filter_and_sample_questions(
            all_questions, 
            subcategory=args.subcategory,
            sample_size=args.sample_size,
            random_seed=args.random_seed
        )
        questions = filtered_questions
        print(f"Filtered questions: {len(questions)} out of {len(all_questions)}")
    else:
        questions = all_questions
        print(f"Using all {len(questions)} questions")
    
    # Load minimax best responses
    minimax_responses = load_minimax_best_responses(args.minimax_file)
    
    if not minimax_responses:
        print("No minimax responses found. Exiting.")
        exit(1)
    
    # Auto-detect model name if not provided
    if not args.model_name:
        # Get model name from first response
        first_response = next(iter(minimax_responses.values()))
        model_name = first_response["model"]
        print(f"Auto-detected model name: {model_name}")
    else:
        model_name = args.model_name
    
    # Create model_answers structure
    model_answers = create_model_answers_structure(minimax_responses, model_name)
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Auto-generate output directory based on minimax file path
        minimax_dir = os.path.dirname(args.minimax_file)
        output_dir = os.path.join(minimax_dir, "minimax_judgment", configs['judge_model'])
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{model_name}_minimax.jsonl")
    print(f"Output file: {output_file}")
    
    # Load existing judgments to avoid duplicates
    existing_judgments = {}
    if os.path.exists(output_file):
        print(f"Loading existing judgments from: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                judgment = json.loads(line.strip())
                existing_judgments[judgment["uid"]] = judgment
        print(f"Found {len(existing_judgments)} existing judgments")

    # Load baseline model answers
    answer_dir = os.path.join("data", configs["bench_name"], "model_answer")
    baseline_answers = load_model_answers(answer_dir)
    
    # Load reference answers if specified
    if configs["reference"]:
        ref_answers = [baseline_answers[model] for model in configs["reference"]]
    else:
        ref_answers = None

    # Get endpoint settings for the judge model
    endpoint_settings = endpoint_list[configs["judge_model"]]

    # Process judgments
    with concurrent.futures.ThreadPoolExecutor(max_workers=endpoint_settings["parallel"]) as executor:
        futures = []
        skipped_count = 0
        
        for question in questions:
            uid = question["uid"]
            
            # Skip if we already have judgment for this question
            if uid in existing_judgments:
                skipped_count += 1
                continue
            
            # Skip if we don't have a minimax response for this question
            if uid not in minimax_responses:
                print(f"Warning: No minimax response found for question {uid}")
                continue
            
            # Get baseline answer
            baseline_model = JUDGE_SETTINGS[question["category"]]["baseline"]
            if baseline_model not in baseline_answers or uid not in baseline_answers[baseline_model]:
                print(f"Warning: No baseline answer found for question {uid} (baseline: {baseline_model})")
                continue
            
            # Prepare judgment arguments
            kwargs = {
                "question": question,
                "answer": minimax_responses[uid],
                "baseline": baseline_answers[baseline_model][uid],
                "reference": [ref_answer[uid] for ref_answer in ref_answers] if ref_answers else None,
                "configs": configs,
                "settings": endpoint_settings,
                "output_file": output_file
            }
            
            future = executor.submit(judgment, kwargs)
            futures.append(future)

        if skipped_count > 0:
            print(f"Skipped {skipped_count} existing judgments")

        print(f"Processing {len(futures)} judgment tasks...")
        
        # Wait for all judgments to complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating judgments"):
            try:
                future.result()
            except Exception as e:
                print(f"Error in judgment: {e}")

    print(f"\nJudgment generation completed!")
    print(f"Results saved to: {output_file}")
    
    # Print summary
    final_judgments = {}
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                judgment = json.loads(line.strip())
                final_judgments[judgment["uid"]] = judgment
        
        print(f"Total judgments generated: {len(final_judgments)}")
        print(f"Minimax responses evaluated: {len([uid for uid in minimax_responses if uid in final_judgments])}")
