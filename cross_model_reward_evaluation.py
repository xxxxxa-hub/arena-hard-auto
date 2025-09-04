import argparse
import json
import os
import time
import random
import threading
import numpy as np
from typing import Dict, List, Tuple, Any
import concurrent.futures
from tqdm import tqdm

from utils.completion import (
    load_questions,
    make_config,
)


class CrossModelRewardEvaluator:
    """
    Base class for cross-model reward evaluation
    Compares responses from your model against baseline model responses
    """
    
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path) if config_path else {}
    
    def load_config(self, config_path: str) -> Dict:
        """Load reward model configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                if config_path.endswith(('.yaml', '.yml')):
                    try:
                        import yaml
                        return yaml.safe_load(f)
                    except ImportError:
                        raise ImportError("Please install PyYAML: pip install PyYAML")
                else:
                    return json.load(f)
        return {}
    
    def evaluate_pair(self, your_response: Dict, baseline_response: Dict, question: Dict) -> float:
        """
        Evaluate which response is better: your model vs baseline
        
        Args:
            your_response: Response from your model
            baseline_response: Response from baseline model
            question: Question dictionary
            
        Returns:
            float: Probability that your_response is better than baseline_response (0.0 to 1.0)
        """
        raise NotImplementedError("Implement this method in your reward model class")


class SimpleCrossModelReward(CrossModelRewardEvaluator):
    """Simple cross-model reward model for demonstration"""
    
    def evaluate_pair(self, your_response: Dict, baseline_response: Dict, question: Dict) -> float:
        """Simple heuristic-based cross-model evaluation"""
        your_content = your_response["messages"][-1]["content"]["answer"]
        baseline_content = baseline_response["messages"][-1]["content"]["answer"]
        
        # Simple length-based scoring
        your_score = len(your_content) / 1000.0
        baseline_score = len(baseline_content) / 1000.0
        
        # Convert to probability using sigmoid
        diff = your_score - baseline_score
        prob = 1.0 / (1.0 + np.exp(-diff))
        
        # Add some randomness
        noise = np.random.normal(0, 0.05)
        prob = max(0.05, min(0.95, prob + noise))
        
        return prob


class LLMCrossModelReward(CrossModelRewardEvaluator):
    """
    LLM-based cross-model reward model
    Uses another LLM to judge which response is better
    """
    
    def __init__(self, config_path: str = None):
        super().__init__(config_path)
        self.judge_model = self.config.get("judge_model", "gpt-4.1-mini")
        self.judge_prompt_template = self.config.get(
            "judge_prompt_template", 
            self._default_judge_prompt()
        )
        
        # Import completion functions
        from utils.completion import registered_api_completion, get_endpoint
        api_type = self.config.get("api_type", "openai")
        self.api_completion_func = registered_api_completion.get(api_type)
        self.api_dict = self.config.get("api_dict", None)
    
    def _default_judge_prompt(self) -> str:
        return """Compare the following two responses to the given question and determine which one is better.

Question: {question}

Response A: {response_a}

Response B: {response_b}

Evaluate based on accuracy, helpfulness, clarity, and relevance.
Output your choice in the format "A" or "B". Do not output any other text.

Choice: """
    
    def _call_openai_with_logprobs(self, messages: list, model: str) -> tuple:
        """Call OpenAI API with assistant message injection for precise logprob extraction"""
        try:
            import openai
            
            if self.api_dict:
                client = openai.OpenAI(
                    base_url=self.api_dict.get("api_base"),
                    api_key=self.api_dict.get("api_key"),
                )
            else:
                client = openai.OpenAI()
            
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.0,
                max_tokens=3,
                logprobs=True,
                top_logprobs=20
            )
            
            if completion.choices and len(completion.choices) > 0:
                choice = completion.choices[0]
                response_text = choice.message.content
                
                if choice.logprobs and choice.logprobs.content:
                    first_token = choice.logprobs.content[0]
                    if first_token.top_logprobs:
                        logprobs_dict = {}
                        for token_logprob in first_token.top_logprobs:
                            logprobs_dict[token_logprob.token] = token_logprob.logprob
                        return response_text, logprobs_dict
                
                return response_text, {}
            
        except Exception as e:
            print(f"Error calling OpenAI with logprobs: {e}")
            return None, {}
        
        return None, {}
    
    def evaluate_pair(self, your_response: Dict, baseline_response: Dict, question: Dict) -> float:
        """Use LLM with logprobs to evaluate which response is better"""
        
        your_content = your_response["messages"][-1]["content"]["answer"]
        baseline_content = baseline_response["messages"][-1]["content"]["answer"]
        
        # Prepare the judgment prompt
        prompt = self.judge_prompt_template.format(
            question=question["prompt"],
            response_a=your_content,
            response_b=baseline_content
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Call OpenAI with logprobs
            response_text, logprobs_dict = self._call_openai_with_logprobs(messages, self.judge_model)
            
            if logprobs_dict:
                # Extract probabilities for A and B
                prob_a = logprobs_dict.get('A', float('-inf'))
                prob_b = logprobs_dict.get('B', float('-inf'))
                
                if prob_a != float('-inf') and prob_b != float('-inf'):
                    # Convert logprobs to probabilities and normalize
                    import math
                    p_a = math.exp(prob_a)
                    p_b = math.exp(prob_b)
                    total = p_a + p_b
                    if total > 0:
                        probability = p_a / total
                        print(f"Logprob evaluation: P(A)={p_a:.4f}, P(B)={p_b:.4f}, P(A>B)={probability:.3f}")
                        return probability
            
        except Exception as e:
            raise ValueError(f"Error in LLM evaluation: {e}")
        
        return 0.5


class LlamaCrossModelReward(CrossModelRewardEvaluator):
    """
    Llama-based reward model using Hugging Face InferenceClient
    """
    
    def __init__(self, config_path: str = None):
        super().__init__(config_path)
        self.judge_model = self.config.get("judge_model", "meta-llama/Llama-3.1-70B-Instruct")
        self.hf_token = self.config.get("hf_token", None)  # Hugging Face token
        
        self._default_judge_prompt = self.config.get(
            "judge_prompt_template", 
            """Compare the following two responses to the given question and determine which one is better.

Question: {question}

Response A: {response_a}

Response B: {response_b}

Evaluate based on accuracy, helpfulness, clarity, and relevance.
Output your choice as either "A" or "B" only."""
        )
    
    def _call_huggingface(self, messages: list) -> tuple:
        """Call Hugging Face model via InferenceClient and return (response_text, probability)"""
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError("Please install huggingface_hub: pip install huggingface_hub")
        
        # Initialize client
        if self.hf_token:
            client = InferenceClient(provider="nebius", token=self.hf_token)
        else:
            client = InferenceClient(provider="nebius")
        
        # Use chat completion with logprobs to get token probabilities
        try:
            response = client.chat_completion(
                messages=messages,
                model=self.judge_model,
                max_tokens=1,  # Only generate 1 token (A or B)
                temperature=0.0,
                logprobs=True,
                top_logprobs=20  # Get top 20 candidate tokens
            )
            
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                response_text = choice.message.content.strip()
                
                # Extract probability from logprobs
                probability = None
                if choice.logprobs and choice.logprobs.content and len(choice.logprobs.content) > 0:
                    # Get the first token's logprobs
                    first_token_logprobs = choice.logprobs.content[0]
                    
                    # Look for 'A' and 'B' tokens in top_logprobs
                    if hasattr(first_token_logprobs, 'top_logprobs') and first_token_logprobs.top_logprobs:
                        import math
                        prob_a = None
                        prob_b = None
                        
                        for logprob_item in first_token_logprobs.top_logprobs:
                            token_text = logprob_item.token.strip().upper()
                            token_prob = math.exp(logprob_item.logprob)
                            
                            if token_text == 'A':
                                prob_a = token_prob
                            elif token_text == 'B':
                                prob_b = token_prob
                        
                        # Calculate P(A>B) based on the token probabilities
                        if prob_a is not None and prob_b is not None:
                            total_prob = prob_a + prob_b
                            if total_prob > 0:
                                probability = prob_a / total_prob
                                print(f"🔍 Debug: P(A)={prob_a:.4f}, P(B)={prob_b:.4f}, P(A>B)={probability:.4f}")
                        elif prob_a is not None:
                            probability = prob_a
                            print(f"🔍 Debug: P(A)={prob_a:.4f}, P(B)=0, P(A>B)={probability:.4f}")
                        elif prob_b is not None:
                            probability = 1.0 - prob_b
                            print(f"🔍 Debug: P(A)=0, P(B)={prob_b:.4f}, P(A>B)={probability:.4f}")
                
                # Format response as [[A]] or [[B]] for consistency
                if response_text.upper() in ['A', 'B']:
                    formatted_response = f"[[{response_text.upper()}]]"
                else:
                    formatted_response = response_text
                
                return formatted_response, probability
                
        except Exception as e:
            print(f"Chat completion with logprobs failed: {e}")
        
        # Fallback to simple chat completion without logprobs
        try:
            response = client.chat_completion(
                messages=messages,
                model=self.judge_model,
                max_tokens=10,
                temperature=0.0
            )
            
            if response.choices and len(response.choices) > 0:
                response_text = response.choices[0].message.content.strip()
                return response_text, None
                
        except Exception as e:
            print(f"Simple chat completion failed: {e}")
        
        return "", None
    
    def evaluate_pair(self, your_response: Dict, baseline_response: Dict, question: Dict) -> float:
        """Use Llama model to evaluate which response is better"""
        
        your_content = your_response["messages"][-1]["content"]["answer"]
        baseline_content = baseline_response["messages"][-1]["content"]["answer"]
        
        prompt = self._default_judge_prompt.format(
            question=question["prompt"],
            response_a=your_content,
            response_b=baseline_content
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response_text, raw_probability = self._call_huggingface(messages)
            
            if response_text:
                # Parse the response text
                response_text = response_text.strip().upper()
                
                if "[[A]]" in response_text or response_text.startswith("A"):
                    # If we have probability from logprobs, use it; otherwise use default
                    if raw_probability is not None:
                        probability = raw_probability
                        print(f"Llama evaluation: '{response_text}' -> P(A>B)={probability:.3f} (from logprobs)")
                    else:
                        probability = 0.8  # Default high confidence for A
                        print(f"Llama evaluation: '{response_text}' -> P(A>B)={probability} (default)")
                    return probability
                    
                elif "[[B]]" in response_text or response_text.startswith("B"):
                    # FIXED: For B choices, probability should be 1 - raw_probability
                    if raw_probability is not None:
                        probability = 1.0 - raw_probability  # Convert to P(A>B)
                        print(f"Llama evaluation: '{response_text}' -> P(A>B)={probability:.3f} (from logprobs)")
                    else:
                        probability = 0.2  # Default low confidence for A when B is chosen
                        print(f"Llama evaluation: '{response_text}' -> P(A>B)={probability} (default)")
                    return probability
                    
                else:
                    print(f"Llama evaluation: '{response_text}' -> P(A>B)=0.5 (unclear)")
                    return 0.5  # Neutral if unclear
            
        except Exception as e:
            print(f"Error in Llama evaluation: {e}")
            return 0.5
        
        return 0.5


def load_cross_model_responses(comparison_dir: str, your_model: str, baseline_model: str):
    """Load responses from both models"""
    
    your_responses_file = os.path.join(comparison_dir, f"{your_model}_multi_responses.jsonl")
    baseline_responses_file = os.path.join(comparison_dir, f"{baseline_model}_multi_responses.jsonl")
    
    def load_responses(filepath):
        responses_by_uid = {}
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                for line in f:
                    response = json.loads(line)
                    uid = response["uid"]
                    if uid not in responses_by_uid:
                        responses_by_uid[uid] = []
                    responses_by_uid[uid].append(response)
        
        # Sort responses by response_idx to ensure correct ordering
        for uid in responses_by_uid:
            responses_by_uid[uid].sort(key=lambda x: x["response_idx"])
        
        return responses_by_uid
    
    your_responses = load_responses(your_responses_file)
    baseline_responses = load_responses(baseline_responses_file)
    
    return your_responses, baseline_responses


def filter_and_sample_questions(questions, subcategory=None, sample_size=None, random_seed=42):
    """Filter questions by subcategory and sample a subset"""
    random.seed(random_seed)
    
    if subcategory is not None:
        filtered_questions = [q for q in questions if q.get("subcategory") == subcategory]
        print(f"Filtered to {len(filtered_questions)} questions in subcategory '{subcategory}'")
    else:
        filtered_questions = questions
        print(f"Using all {len(filtered_questions)} questions from all subcategories")
    
    if sample_size is not None and sample_size < len(filtered_questions):
        sampled_questions = random.sample(filtered_questions, sample_size)
        print(f"Randomly sampled {len(sampled_questions)} questions")
    else:
        sampled_questions = filtered_questions
        if sample_size is not None:
            print(f"Sample size ({sample_size}) >= available questions ({len(filtered_questions)}), using all")
    
    return sampled_questions


def process_single_question(args_tuple):
    """Process a single question for parallel execution"""
    uid, question, your_responses, baseline_responses, reward_model, max_workers = args_tuple
    
    try:
        # Run cross-model minimax
        best_response, comparison_matrix, worst_case_scores = cross_model_minimax(
            your_responses, baseline_responses, reward_model, question, max_workers
        )
        
        if best_response is None:
            return None
        
        # Prepare results
        result = {
            'uid': uid,
            'question': question,
            'best_response': best_response,
            'comparison_matrix': comparison_matrix.tolist(),
            'worst_case_scores': worst_case_scores,
            'n_your': len(your_responses),
            'n_baseline': len(baseline_responses),
            'best_idx': worst_case_scores.index(max(worst_case_scores)) if worst_case_scores else 0
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing question {uid}: {e}")
        return None


def cross_model_minimax(
    your_responses: List[Dict], 
    baseline_responses: List[Dict], 
    reward_model: CrossModelRewardEvaluator, 
    question: Dict,
    max_workers: int = 8
) -> Tuple[Dict, np.ndarray, List[float]]:
    """Run cross-model minimax to select best response from your model"""
    n_your = len(your_responses)
    n_baseline = len(baseline_responses)
    
    if n_your == 0:
        return None, np.array([]), []
    
    if n_baseline == 0:
        return your_responses[0], np.array([]), [1.0] * n_your
    
    # Create cross-model comparison matrix with parallel evaluation
    comparison_matrix = np.zeros((n_your, n_baseline))
    
    def evaluate_comparison(args):
        """Helper function for parallel evaluation"""
        i, j, your_resp, baseline_resp, question = args
        prob = reward_model.evaluate_pair(your_resp, baseline_resp, question)
        return i, j, prob
    
    # Create all comparison tasks
    comparison_tasks = []
    for i in range(n_your):
        for j in range(n_baseline):
            comparison_tasks.append((i, j, your_responses[i], baseline_responses[j], question))
    
    # Execute comparisons in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(comparison_tasks), desc="Cross-model evaluation") as pbar:
            future_to_task = {executor.submit(evaluate_comparison, task): task for task in comparison_tasks}
            
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    i, j, prob = future.result()
                    comparison_matrix[i][j] = prob
                    pbar.update(1)
                except Exception as e:
                    print(f"Error in comparison: {e}")
                    # Set default probability on error
                    task = future_to_task[future]
                    i, j = task[0], task[1]
                    comparison_matrix[i][j] = 0.5
                    pbar.update(1)
    
    # Minimax: find worst-case performance for each of your responses
    worst_case_scores = []
    for i in range(n_your):
        worst_case = min(comparison_matrix[i]) if n_baseline > 0 else 0.5
        worst_case_scores.append(worst_case)
    
    # Select your response with best worst-case performance
    best_idx = np.argmax(worst_case_scores)
    
    return your_responses[best_idx], comparison_matrix, worst_case_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-model reward evaluation and minimax selection")
    parser.add_argument("--bench-name", type=str, default="arena-hard-v2.0")
    parser.add_argument("--your-model", type=str, required=True)
    parser.add_argument("--baseline-model", type=str, required=True)
    parser.add_argument("--reward-model-type", type=str, default="simple", choices=["simple", "llm", "llama"])
    parser.add_argument("--reward-model-config", type=str, default=None)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum number of parallel workers for reward model evaluation")
    parser.add_argument("--question-parallel", type=int, default=2, help="Number of questions to process in parallel")
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

    # Load and filter questions
    question_file = os.path.join("data", args.bench_name, "question.jsonl")
    all_questions = load_questions(question_file)
    
    # Filter and sample questions
    filtered_questions = filter_and_sample_questions(
        all_questions, 
        subcategory=args.subcategory, 
        sample_size=args.sample_size,
        random_seed=args.random_seed
    )
    
    # Convert to dict for lookup
    questions = {q["uid"]: q for q in filtered_questions}

    # Load cross-model responses
    comparison_dir = os.path.join("data", args.bench_name, "cross_model_comparison")
    your_responses_by_uid, baseline_responses_by_uid = load_cross_model_responses(
        comparison_dir, args.your_model, args.baseline_model
    )

    if not your_responses_by_uid:
        print(f"No responses found for your model {args.your_model}")
        print(f"Run gen_cross_model_comparison.py first")
        exit(1)

    if not baseline_responses_by_uid:
        print(f"No responses found for baseline model {args.baseline_model}")
        print(f"Run gen_cross_model_comparison.py first")
        exit(1)

    # Initialize reward model
    if args.reward_model_type == "simple":
        reward_model = SimpleCrossModelReward(args.reward_model_config)
    elif args.reward_model_type == "llm":
        reward_model = LLMCrossModelReward(args.reward_model_config)
    elif args.reward_model_type == "llama":
        reward_model = LlamaCrossModelReward(args.reward_model_config)
    else:
        raise ValueError(f"Unknown reward model type: {args.reward_model_type}")

    # Output files
    best_responses_file = os.path.join(comparison_dir, f"{args.your_model}_vs_{args.baseline_model}_minimax_best.jsonl")
    evaluation_info_file = os.path.join(comparison_dir, f"{args.your_model}_vs_{args.baseline_model}_evaluation_info.jsonl")

    print(f"Cross-model evaluation:")
    print(f"  Your model: {args.your_model} ({len(your_responses_by_uid)} questions)")
    print(f"  Baseline model: {args.baseline_model} ({len(baseline_responses_by_uid)} questions)")
    print(f"  Reward model: {args.reward_model_type}")

    # Process each question (only those in filtered set)
    available_uids = set(your_responses_by_uid.keys()) & set(baseline_responses_by_uid.keys())
    filtered_uids = set(questions.keys())
    common_uids = available_uids & filtered_uids
    
    print(f"  Available questions with responses: {len(available_uids)}")
    print(f"  Filtered questions: {len(filtered_uids)}")
    print(f"  Common questions to process: {len(common_uids)}")

    # Clear output files
    with open(best_responses_file, "w", encoding="utf-8") as f:
        pass
    with open(evaluation_info_file, "w", encoding="utf-8") as f:
        pass
    
    # Create question processing tasks
    question_tasks = []
    for uid in common_uids:
        if uid not in questions:
            continue
        
        question = questions[uid]
        your_question_responses = your_responses_by_uid[uid]
        baseline_question_responses = baseline_responses_by_uid[uid]
        
        if len(your_question_responses) == 0 or len(baseline_question_responses) == 0:
            continue
        
        task = (uid, question, your_question_responses, baseline_question_responses, reward_model, args.max_workers)
        question_tasks.append(task)
    
    print(f"  Processing {len(question_tasks)} questions with {args.question_parallel} parallel workers")
    
    # Thread-safe file writing
    write_lock = threading.Lock()
    
    def save_result(result):
        """Thread-safe result saving"""
        if result is None:
            return
        
        with write_lock:
            # Save best response
            with open(best_responses_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result['best_response'], ensure_ascii=False) + "\n")
            
            # Save evaluation info
            eval_info = {
                "uid": result['uid'],
                "your_model": args.your_model,
                "baseline_model": args.baseline_model,
                "num_your_responses": result['n_your'],
                "num_baseline_responses": result['n_baseline'],
                "selected_response_idx": result['best_response']["response_idx"],
                "worst_case_scores": result['worst_case_scores'],
                "comparison_matrix": result['comparison_matrix'],
                "avg_performance_vs_baseline": float(np.mean(result['comparison_matrix'])),
                "best_worst_case_score": float(max(result['worst_case_scores'])) if result['worst_case_scores'] else 0.0,
                "reward_model_type": args.reward_model_type,
                "tstamp": time.time()
            }
            
            with open(evaluation_info_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(eval_info, ensure_ascii=False) + "\n")
    
    # Process questions in parallel
    if args.question_parallel == 1:
        # Sequential processing
        for task in tqdm(question_tasks, desc="Cross-model evaluation"):
            result = process_single_question(task)
            save_result(result)
    else:
        # Parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.question_parallel) as executor:
            with tqdm(total=len(question_tasks), desc="Cross-model evaluation") as pbar:
                future_to_task = {executor.submit(process_single_question, task): task for task in question_tasks}
                
                for future in concurrent.futures.as_completed(future_to_task):
                    try:
                        result = future.result()
                        save_result(result)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing question: {e}")
                        pbar.update(1)

    print(f"\nCross-model evaluation completed!")
    print(f"Best responses: {best_responses_file}")
    print(f"Evaluation info: {evaluation_info_file}")
