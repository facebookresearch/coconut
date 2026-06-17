"""
Quick test script for Coconut branching with majority voting.

This is a simplified version for quick testing and debugging of the branching
and majority voting functionality. Use this to verify the system works before
running larger experiments.

Usage:
    python quick_branch_test.py                    # Use default config.yaml
    python quick_branch_test.py --config my.yaml   # Use custom config file
    python quick_branch_test.py experiment.num_questions=50  # Override via CLI
"""

import torch
import json
import sys
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from coconut import Coconut
from datasets import load_dataset
from typing import List, Dict, Any
import re
from collections import Counter
import random
import os
import signal

from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm


def load_config(config_path: str = "config.yaml") -> DictConfig:
    """Load configuration from YAML file and merge with CLI overrides."""
    # Load base config
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Merge with CLI arguments (allows overrides like: experiment.num_questions=50)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    
    return cfg


def get_results_dir(cfg: DictConfig) -> Path:
    """Get the results directory, creating it if needed."""
    results_dir = Path(cfg.output_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_checkpoint_file(cfg: DictConfig) -> Path:
    """Generate checkpoint file path based on config."""
    results_dir = get_results_dir(cfg)
    model_name_safe = cfg.model.name.replace('/', '_')
    return results_dir / f"checkpoint_{cfg.benchmark}_{model_name_safe}.json"


def load_checkpoint(checkpoint_file: Path) -> Dict[str, Any]:
    """Load checkpoint if it exists."""
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        print(f"✓ Loaded checkpoint: {len(checkpoint['results'])} questions completed")
        return checkpoint
    return None


def save_checkpoint(checkpoint_file: Path, results: List[Dict], completed_ids: set, config: Dict):
    """Save checkpoint to disk."""
    checkpoint = {
        "config": config,
        "completed_question_ids": list(completed_ids),
        "results": results,
        "last_saved": datetime.now().isoformat()
    }
    temp_file = checkpoint_file.with_suffix('.json.tmp')
    with open(temp_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    os.replace(temp_file, checkpoint_file)
    print(f"✓ Checkpoint saved: {len(results)} questions completed")


def setup_model(model_name: str):
    """Initialize model and add special tokens."""
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )

    # Qwen3 is released in bfloat16; loading it in float16 overflows and yields
    # inf/nan logits that crash multinomial sampling, so it needs bfloat16 too.
    if "gpt-oss-20b" in model_name or "Qwen3" in model_name:
        dtype = torch.bfloat16
        print("Using bfloat16")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        dtype = torch.float16
        print("Using float16")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )

    special_tokens = {
        'additional_special_tokens': ['<|latent|>', '<|start-latent|>', '<|end-latent|>']
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens")

    base_model.resize_token_embeddings(len(tokenizer))

    latent_token_id = tokenizer.convert_tokens_to_ids('<|latent|>')
    start_latent_id = tokenizer.convert_tokens_to_ids('<|start-latent|>')
    end_latent_id = tokenizer.convert_tokens_to_ids('<|end-latent|>')
    eos_token_id = tokenizer.eos_token_id

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Model loaded!")
    print(f"  <|latent|>: {latent_token_id}")
    print(f"  <|start-latent|>: {start_latent_id}")
    print(f"  <|end-latent|>: {end_latent_id}")

    return tokenizer, base_model, latent_token_id, start_latent_id, end_latent_id, eos_token_id


def load_benchmark_dataset(benchmark: str = "gsm8k", num_questions: int = None, random_seed: int = None) -> List[Dict[str, str]]:
    """
    Load benchmark dataset (GSM8K, GSM-Symbolic, MMLU, MATH, or GPQA).

    Args:
        benchmark: One of "gsm8k", "gsm-symbolic", "mmlu", "math", "gpqa",
                   "gpqa-diamond", or "gpqa-extended"
        num_questions: Number of questions to load (None = all)
        random_seed: Seed for random sampling (None = no randomization, uses first N)

    Returns:
        List of dictionaries with "question", "answer", "question_id" fields
    """
    print(f"\nLoading {benchmark.upper()} test set...")

    if benchmark == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="test")
    elif benchmark == "gsm-symbolic":
        dataset = load_dataset("apple/gsm-symbolic", split="test", trust_remote_code=True)
    elif benchmark == "mmlu":
        dataset = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    elif benchmark == "math":
        # MATH uses the 500-question Hendrycks test split only. The downstream
        # cap (n = min(num_questions, total)) limits this to at most 500, so
        # requesting more simply uses the full test split without switching
        # sources. We report this split for MATH in the paper.
        dataset = load_dataset("nlile/hendrycks-MATH-benchmark", split="test")
    elif benchmark in ("gpqa", "gpqa-diamond", "gpqa-extended"):
        # GPQA is gated on the Hub, so we prefer the local normalized copy and
        # only fetch from the Hub (which needs a token) when it is absent.
        # data.download_benchmarks owns the fetch and schema normalization, and
        # is a no-op when the file already exists, so this stays modular and
        # works for a fresh user without a local cache.
        import data
        data_dir = Path(__file__).resolve().parent / "data"
        local_path = data_dir / benchmark / "test.json"
        if not local_path.exists():
            data.download_benchmarks(data_dir=str(data_dir), benchmarks=[benchmark])
        with open(local_path) as f:
            rows = json.load(f)
        # Remap the normalized schema to the raw column names the parser expects.
        dataset = [
            {
                "Question": r["question"],
                "Correct Answer": r["correct_answer"],
                "Incorrect Answer 1": r["incorrect_answer_1"],
                "Incorrect Answer 2": r["incorrect_answer_2"],
                "Incorrect Answer 3": r["incorrect_answer_3"],
            }
            for r in rows
        ]
    else:
        raise ValueError(
            f"Unknown benchmark: {benchmark}. "
            "Must be one of: gsm8k, gsm-symbolic, mmlu, math, gpqa, gpqa-diamond, gpqa-extended"
        )

    # Determine which indices to use
    total = len(dataset)
    n = min(num_questions or total, total)

    if random_seed is not None:
        random.seed(random_seed)
        indices = random.sample(range(total), n)
        print(f"Randomly sampling {n} questions (seed={random_seed})")
    else:
        indices = list(range(n))

    # Build questions list
    questions = []
    answer_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    for i in tqdm(indices, desc="Loading questions", leave=False):
        item = dataset[i]

        if benchmark in ("gsm8k", "gsm-symbolic"):
            questions.append({
                "question": item["question"],
                "answer": item["answer"],
                "question_id": i,
                "benchmark": benchmark
            })
        elif benchmark == "mmlu":
            choices_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(item["choices"])])
            formatted_question = f"{item['question']}\n\n{choices_text}"
            correct_answer = answer_mapping[item["answer"]]
            questions.append({
                "question": formatted_question,
                "answer": f"#### {correct_answer}",
                "question_id": i,
                "benchmark": "mmlu",
                "subject": item.get("subject", "unknown"),
                "choices": item["choices"],
                "correct_choice_idx": item["answer"]
            })
        elif benchmark == "math":
            if "answer" in item:
                # nlile format: answer is pre-extracted
                answer = item["answer"]
                level = item.get("level", "")
                subject = item.get("subject", "")
            else:
                # qwedsacf format: extract answer from \boxed{} in solution
                solution = item["solution"]
                boxed_match = re.search(r'\\boxed\{', solution)
                if boxed_match:
                    start = boxed_match.end()
                    depth = 1
                    pos = start
                    while pos < len(solution) and depth > 0:
                        if solution[pos] == '{':
                            depth += 1
                        elif solution[pos] == '}':
                            depth -= 1
                        pos += 1
                    answer = normalize_math_answer(solution[start:pos - 1]) if depth == 0 else ""
                else:
                    answer = ""
                level = item.get("level", "").replace("Level ", "").strip()
                subject = item.get("type", "")
            questions.append({
                "question": item["problem"],
                "answer": f"#### {answer}",
                "question_id": i,
                "benchmark": "math",
                "level": level,
                "subject": subject
            })
        else:  # gpqa variants
            # Shuffle the four options with a per-question seed for reproducibility
            options = [
                item["Correct Answer"],
                item["Incorrect Answer 1"],
                item["Incorrect Answer 2"],
                item["Incorrect Answer 3"],
            ]
            rng = random.Random((random_seed or 0) + i)
            rng.shuffle(options)
            correct_idx = options.index(item["Correct Answer"])
            choices_text = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(options)])
            questions.append({
                "question": f"{item['Question']}\n\n{choices_text}",
                "answer": f"#### {chr(65 + correct_idx)}",
                "question_id": i,
                "benchmark": benchmark,
                "choices": options,
                "correct_choice_idx": correct_idx
            })

    print(f"Loaded {len(questions)} questions from {benchmark.upper()}")
    return questions


def normalize_math_answer(answer: str) -> str:
    """Normalize a LaTeX math answer string for comparison."""
    answer = answer.strip()
    # Strip markdown emphasis and LaTeX display/inline delimiters the model
    # often wraps the answer in (e.g. "**$42$**" or "\[ \frac{3}{4} \]").
    answer = answer.replace('**', '')
    answer = re.sub(r'\\[\[\]()]', '', answer)      # \[ \] \( \)
    answer = re.sub(r'\\boxed\s*\{(.*)\}\s*$', r'\1', answer)
    answer = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\left', '', answer)
    answer = re.sub(r'\\right', '', answer)
    answer = answer.strip().strip('$').strip()
    answer = re.sub(r'\s+', ' ', answer).strip()
    answer = answer.rstrip('.').strip()
    return answer


def answers_match(predicted: str, expected: str, benchmark: str) -> bool:
    """Check if predicted and expected answers match, with benchmark-specific logic."""
    if predicted == expected:
        return True
    if benchmark == "math":
        # Try numeric comparison to catch equivalent float/fraction representations
        # (e.g. "0.75" == "\frac{3}{4}" after the model outputs a decimal)
        try:
            pred_val = float(predicted.replace(',', ''))
            exp_val = float(expected.replace(',', ''))
            return abs(pred_val - exp_val) < 1e-6
        except ValueError:
            pass
    return False


def extract_answer(text: str, benchmark: str = "gsm8k") -> str:
    if benchmark in ("mmlu", "gpqa", "gpqa-diamond", "gpqa-extended"):
        # The model may wrap the answer letter in markdown bold, parentheses,
        # dollar signs, or LaTeX (\boxed{}, \text{}, \[ \]). Allow any run of
        # such wrappers between an answer cue and the A-D letter. The trailing
        # \b stops us from matching a letter inside a word (e.g. "Avogadro").
        OPEN = r'(?:\*{1,2}|\$|\(|\\boxed\s*\{|\\text\s*\{|\\mathbf\s*\{|\\\[|\\\(|\{|:|\s)*'
        cue = (r'(?:####|'
               r'(?i:(?:the\s+)?'
               r'(?:final\s+answer|correct\s+(?:answer|option|choice)|answer|option|choice)'
               r'\s*(?:is)?'
               r'))')
        combined = cue + r'[\s:]*' + OPEN + r'([A-D])\b'
        # Take the LAST match: the model's concluding statement, not an earlier
        # tentative mention or the restated prompt.
        matches = list(re.finditer(combined, text))
        if matches:
            return matches[-1].group(1).upper()

        # Fallbacks on the tail: a wrapped or trailing standalone letter.
        tail = text[-400:]
        for pattern in (r'\\boxed\s*\{\s*(?:\\text\s*\{\s*)?([A-D])',
                        r'\*\*\(?([A-D])\)?\*\*',
                        r'\(([A-D])\)',
                        r'\b([A-D])\s*\.?\s*$'):
            ms = list(re.finditer(pattern, tail))
            if ms:
                return ms[-1].group(1).upper()

        return "NO_ANSWER_FOUND"

    elif benchmark == "math":
        # Prefer the LAST \boxed{...} (the final answer), matching nested braces.
        boxed_starts = [m.end() for m in re.finditer(r'\\boxed\s*\{', text)]
        if boxed_starts:
            start = boxed_starts[-1]
            depth = 1
            pos = start
            while pos < len(text) and depth > 0:
                if text[pos] == '{':
                    depth += 1
                elif text[pos] == '}':
                    depth -= 1
                pos += 1
            if depth == 0:
                return normalize_math_answer(text[start:pos - 1])

        # Fallback cue phrases, tolerating bold/LaTeX wrappers (handled by the
        # normalizer). Take the LAST match.
        for pattern in (r'####\s*(.+?)(?:\n|$)',
                        r'(?i:(?:the\s+)?final\s+answer\s+is)[:\s]*(.+?)(?:\n|$)',
                        r'(?i:(?:the\s+)?answer\s+is)[:\s]*(.+?)(?:\n|$)',
                        r'(?i:final\s+answer)[:\s]*(.+?)(?:\n|$)'):
            ms = list(re.finditer(pattern, text))
            if ms:
                cand = normalize_math_answer(ms[-1].group(1).strip())
                if cand:
                    return cand

        return "NO_ANSWER_FOUND"

    else:
        def clean_number(num_str: str) -> str:
            return num_str.replace(',', '').replace('$', '').strip()

        # TODO: Qwen3-specific answer extraction
        # Qwen3 models wrap answers in bold markdown (**$X**) and emit <think>...</think>
        # blocks with intermediate calculations that interfere with the last-number fallback.
        # To enable, add `model_name: str = ""` to the function signature, pass it through
        # from the call sites, and uncomment the block below:
        #
        # search_text = text
        # if model_name and "qwen3" in model_name.lower():
        #     search_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        #     if not search_text:
        #         search_text = text  # nothing outside think block, fall back to full text
        #
        # Then replace `text` with `search_text` in the patterns loop and fallback below,
        # and add these two patterns to the list (bold-markdown variants):
        #     r'[Tt]he final answer is:?\s*\*{0,2}\$?\*{0,2}\s*([\d,]+)',
        #     r'\*\*\$?([\d,]+)\*\*\s*[.,]?\s*$',

        patterns = [
            r'\\boxed\{([\d,]+)\}',
            r'####\s*([\d,]+)',
            r'[Tt]he final answer is:?\s*\$?\s*([\d,]+)',
            r'[Ff]inal answer:?\s*\$?\s*([\d,]+)',
            r'[Aa]nswer:?\s*\$?\s*([\d,]+)',
            r'is:?\s*\$?\s*([\d,]+)\s*\.?\s*$',
        ]

        for pattern in patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                return clean_number(matches[-1].group(1))

        numbers = re.findall(r'[\d,]+', text)
        if numbers:
            cleaned_numbers = [clean_number(n) for n in numbers if clean_number(n)]
            if cleaned_numbers:
                return cleaned_numbers[-1]

        return "NO_ANSWER_FOUND"


def create_coconut_input(tokenizer, question: str, start_latent_id: int, end_latent_id: int, model_name: str = "", benchmark: str = "gsm8k"):
    """Create input for Coconut model with latent reasoning markers."""
    
    # Weaker instruction-followers (Llama, Mixtral) tend to reason at length and
    # trail off on hard questions without ever committing to an answer, which the
    # extractor scores as an abstention and collapses coverage accuracy. They need
    # the same forceful, unambiguous closing directive.
    needs_strict_format = any(m in model_name.lower() for m in ("llama", "mixtral"))

    if benchmark in ("mmlu", "gpqa", "gpqa-diamond", "gpqa-extended"):
        if needs_strict_format:
            instruction = (
                f"{question}\n\nSolve this step by step, keeping your reasoning concise. "
                f"You MUST finish with a final line in exactly this format: 'The answer is X', "
                f"where X is exactly one of the letters A, B, C, or D. "
                f"Do not stop before writing that final line."
            )
        else:
            instruction = f"{question}\n\nPlease solve this step by step. At the end, clearly state your final answer as 'The answer is X' where X is A, B, C, or D."
    elif benchmark == "math":
        if needs_strict_format:
            instruction = (
                f"{question}\n\nSolve this step by step, keeping your reasoning concise. "
                f"You MUST finish with your final answer written as \\boxed{{ANSWER}}. "
                f"Do not stop before writing the \\boxed{{}} answer."
            )
        else:
            instruction = f"{question}\n\nPlease solve this step by step. At the end, clearly state your final answer using \\boxed{{}} notation."
    else:
        instruction = f"{question}\n\nPlease solve this step by step."
    
    messages = [
        {"role": "user", "content": instruction}
    ]

    question_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors='pt',
        add_special_tokens=True
    )
    if hasattr(question_ids, 'input_ids'):
        question_ids = question_ids.input_ids

    if "gpt-oss" in model_name:
        # Model expects: <|start|>assistant<|channel|>...<|message|>
        # Insert latent markers after adding the channel structure
        channel_token = tokenizer.convert_tokens_to_ids('<|channel|>')
        message_token = tokenizer.convert_tokens_to_ids('<|message|>')
        
        # Encode "analysis" as that's the first channel it expects
        analysis_ids = tokenizer.encode('analysis', add_special_tokens=False)
        
        # Build: question_ids + <|channel|> + analysis + <|message|> + <|start-latent|> + <|end-latent|>
        channel_prefix = torch.tensor([[channel_token] + analysis_ids + [message_token]])
        start_token = torch.tensor([[start_latent_id]])
        end_token = torch.tensor([[end_latent_id]])
        
        input_ids = torch.cat([question_ids, channel_prefix, start_token, end_token], dim=1)
    else:
        # Standard format for other models
        start_token = torch.tensor([[start_latent_id]])
        end_token = torch.tensor([[end_latent_id]])
        input_ids = torch.cat([question_ids, start_token, end_token], dim=1)

    return input_ids


def extract_generated_only(tokenizer, full_ids: torch.Tensor, original_input_ids: torch.Tensor, end_latent_id: int) -> str:
    """Extract only the newly generated tokens after <end-latent>."""
    end_latent_positions = (original_input_ids[0] == end_latent_id).nonzero(as_tuple=True)[0]

    if len(end_latent_positions) > 0:
        end_latent_pos_expanded = end_latent_positions[0].item() + 8

        if full_ids.shape[1] > end_latent_pos_expanded + 1:
            generated_ids = full_ids[0, end_latent_pos_expanded + 1:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            return generated_text.strip()

    original_len = original_input_ids.shape[1]
    if full_ids.shape[1] > original_len:
        generated_ids = full_ids[0, original_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text.strip()

    return ""


def majority_vote(answers: List[str]) -> str:
    """Perform majority voting on a list of answers."""
    if not answers:
        return "NO_ANSWER_FOUND"

    vote_counts = Counter(answers)
    most_common = vote_counts.most_common(1)[0][0]
    return most_common


def test_question_with_branching(
    coconut_model,
    tokenizer,
    question: str,
    noise_scale: float,
    start_latent_id: int,
    end_latent_id: int,
    num_branches: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    device: str,
    noise_type: str,
    noise_direction: str,
    noise_schedule: str = "none",
    lambda_decay: float = 0.5,
    noise_at_step: int = 1,
    benchmark: str = "gsm8k",
    model_name: str = ""
) -> Dict[str, Any]:
    """Test a single question using branching generation with majority voting."""
    coconut_model.eval()

    input_ids = create_coconut_input(tokenizer, question, start_latent_id, end_latent_id, 
                                     model_name=model_name, benchmark=benchmark)
    original_input_ids = input_ids.clone()
    input_ids = input_ids.to(device)

    with torch.no_grad():
        try:
            generation_output = coconut_model.generate_with_branching(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                num_branches=num_branches,
                temperature=temperature,
                top_p=top_p,
                noise_scale=noise_scale,
                noise_type=noise_type,
                noise_direction=noise_direction,
                noise_schedule=noise_schedule,
                lambda_decay=lambda_decay,
                noise_at_step=noise_at_step,
                model_name=model_name,
                return_diversity_metrics=True
            )

            all_branches, diversity_metrics = generation_output
            d_k_mean = diversity_metrics["D_K"]["d_k_mean"]
            d_k_late = diversity_metrics["D_K"]["d_k_late"]
            d_k_normalized = diversity_metrics["D_K"]["d_k_normalized"]
            d_k_pairwise = diversity_metrics["D_K"]["d_k_pairwise"]
            d_k_pair_labels = diversity_metrics["D_K"]["pair_labels"]

            branch_texts = []
            branch_answers = []

            for branch_ids in all_branches:
                branch_text = extract_generated_only(
                    tokenizer,
                    branch_ids,
                    original_input_ids,
                    end_latent_id
                )
                branch_answer = extract_answer(branch_text, benchmark)

                branch_texts.append(branch_text)
                branch_answers.append(branch_answer)

            majority_answer = majority_vote(branch_answers)
            vote_distribution = dict(Counter(branch_answers))

            return {
                "success": True,
                "branch_texts": branch_texts,
                "branch_answers": branch_answers,
                "majority_answer": majority_answer,
                "vote_distribution": vote_distribution,
                "num_branches": num_branches,
                "d_k": d_k_mean,
                "d_k_late": d_k_late,
                "d_k_normalized": d_k_normalized,
                "d_k_pairwise": d_k_pairwise,
                "d_k_pair_labels": d_k_pair_labels,
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "branch_texts": [],
                "branch_answers": [],
                "majority_answer": "NO_ANSWER_FOUND",
                "vote_distribution": {},
                "num_branches": num_branches,
                "error": str(e)
            }


def run_quick_test(cfg: DictConfig):
    """Run a quick test of branching and majority voting."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Convert config to dict for serialization
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    print("\n" + "="*70)
    print("QUICK BRANCHING + MAJORITY VOTING TEST")
    print("="*70)
    print(f"Model: {cfg.model.name}")
    print(f"Benchmark: {cfg.benchmark}")
    print(f"Questions: {cfg.experiment.num_questions}")
    print(f"Branches: {cfg.experiment.num_branches}")
    print(f"Noise scales: {cfg.noise.scales}")
    print(f"Noise type: {cfg.noise.type}")
    print(f"Temperature: {cfg.sampling.temperature}, Top-p: {cfg.sampling.top_p}")
    print(f"Output directory: {get_results_dir(cfg)}")
    print(f"Device: {device}")
    print("="*70)

    # Check for existing checkpoint
    checkpoint_file = get_checkpoint_file(cfg)
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint:
        results = checkpoint["results"]
        completed_ids = set(checkpoint["completed_question_ids"])
        print(f"Resuming from checkpoint: {len(completed_ids)} questions already done")
    else:
        results = []
        completed_ids = set()

    # Setup
    tokenizer, base_model, latent_token_id, start_latent_id, end_latent_id, eos_token_id = setup_model(cfg.model.name)

    if "gpt-oss" in cfg.model.name:
        hidden_layer_idx = 1
    else:
        hidden_layer_idx = -1

    coconut_model = Coconut(
        base_causallm=base_model,
        latent_token_id=latent_token_id,
        start_latent_id=start_latent_id,
        end_latent_id=end_latent_id,
        eos_token_id=eos_token_id,
        hidden_layer_idx=hidden_layer_idx
    )

    questions = load_benchmark_dataset(
        cfg.benchmark, 
        cfg.experiment.num_questions, 
        random_seed=cfg.experiment.random_seed
    )

    questions_to_process = [q for q in questions if q["question_id"] not in completed_ids]
    print(f"Questions remaining: {len(questions_to_process)}/{len(questions)}")

    def sigterm_handler(signum, frame):
        print("\n⚠ Received SIGTERM, saving checkpoint...")
        save_checkpoint(checkpoint_file, results, completed_ids, config_dict)
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    # Main progress bar
    pbar = tqdm(
        questions_to_process, 
        desc="Processing questions",
        initial=len(completed_ids),
        total=len(questions)
    )

    try:
        for q_data in pbar:
            pbar.set_postfix({
                "id": q_data["question_id"],
                "completed": len(completed_ids)
            })

            question_results = {
                "question_id": q_data["question_id"],
                "question": q_data["question"],
                "expected_answer": q_data["answer"],
                "noise_tests": []
            }

            expected_answer = extract_answer(q_data["answer"], cfg.benchmark)

            # Noise scale progress bar
            noise_pbar = tqdm(
                cfg.noise.scales, 
                desc=f"  Noise scales", 
                leave=False
            )

            for noise_scale in noise_pbar:
                noise_pbar.set_postfix({"noise": noise_scale})

                result = test_question_with_branching(
                    coconut_model=coconut_model,
                    tokenizer=tokenizer,
                    question=q_data["question"],
                    noise_scale=noise_scale,
                    start_latent_id=start_latent_id,
                    end_latent_id=end_latent_id,
                    num_branches=cfg.experiment.num_branches,
                    temperature=cfg.sampling.temperature,
                    top_p=cfg.sampling.top_p,
                    max_new_tokens=cfg.model.max_new_tokens,
                    device=device,
                    noise_type=cfg.noise.type,
                    noise_direction=cfg.noise.direction,
                    noise_schedule=cfg.noise.get("schedule", "none"),
                    lambda_decay=cfg.noise.get("lambda_decay", 0.5),
                    noise_at_step=cfg.noise.get("injection_step", 1),
                    benchmark=cfg.benchmark,
                    model_name=cfg.model.name
                )

                if result["success"]:
                    is_correct = answers_match(result["majority_answer"], expected_answer, cfg.benchmark)

                    question_results["noise_tests"].append({
                        "noise_scale": noise_scale,
                        "branch_texts": result["branch_texts"],
                        "branch_answers": result["branch_answers"],
                        "majority_answer": result["majority_answer"],
                        "vote_distribution": result["vote_distribution"],
                        "expected_answer": expected_answer,
                        "is_correct": is_correct,
                        "num_branches": result["num_branches"],
                        "d_k": result["d_k"],
                        "d_k_late": result["d_k_late"],
                        "d_k_normalized": result["d_k_normalized"],
                        "d_k_pairwise": result["d_k_pairwise"],
                        "d_k_pair_labels": result["d_k_pair_labels"]
                    })
                else:
                    question_results["noise_tests"].append({
                        "noise_scale": noise_scale,
                        "error": result["error"]
                    })

            results.append(question_results)
            completed_ids.add(q_data["question_id"])

            # Checkpoint every N questions
            if len(completed_ids) % cfg.checkpoint.interval == 0:
                save_checkpoint(checkpoint_file, results, completed_ids, config_dict)

    except KeyboardInterrupt:
        print("\n⚠ Interrupted, saving checkpoint...")
        save_checkpoint(checkpoint_file, results, completed_ids, config_dict)
        sys.exit(0)

    # Final checkpoint
    save_checkpoint(checkpoint_file, results, completed_ids, config_dict)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    total_correct = {scale: 0 for scale in cfg.noise.scales}
    total_count = {scale: 0 for scale in cfg.noise.scales}
    total_d_k = {scale: 0.0 for scale in cfg.noise.scales}

    for q_result in results:
        for test in q_result["noise_tests"]:
            noise = test.get("noise_scale")
            if noise is not None and noise in total_count and "is_correct" in test:
                total_count[noise] += 1
                if test["is_correct"]:
                    total_correct[noise] += 1
                if "d_k" in test:
                    total_d_k[noise] += test["d_k"]

    print("\nAccuracy and path diversity by noise scale:")
    for noise_scale in cfg.noise.scales:
        if total_count[noise_scale] > 0:
            acc = total_correct[noise_scale] / total_count[noise_scale] * 100
            mean_dk = total_d_k[noise_scale] / total_count[noise_scale]
            print(f"  Noise={noise_scale}: {total_correct[noise_scale]}/{total_count[noise_scale]} ({acc:.2f}%), mean D_K={mean_dk:.4f}")

    # Save final results
    results_dir = get_results_dir(cfg)
    model_name_safe = cfg.model.name.replace('/', '_')
    output_file = results_dir / f"noisy_coconut_{cfg.benchmark}_{model_name_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "config": config_dict,
            "results": results,
            "summary": {
                "total_questions": len(results),
                "accuracy_by_noise": {
                    str(scale): {
                        "correct": total_correct[scale],
                        "total": total_count[scale],
                        "accuracy": total_correct[scale] / total_count[scale] if total_count[scale] > 0 else 0,
                        "mean_d_k": total_d_k[scale] / total_count[scale] if total_count[scale] > 0 else 0
                    }
                    for scale in cfg.noise.scales
                }
            }
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    # Parse --config argument if provided, otherwise use default
    config_path = "config.yaml"
    
    # Check for --config argument
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--config" and i + 2 < len(sys.argv):
            config_path = sys.argv[i + 2]
            # Remove --config and its value from sys.argv so OmegaConf doesn't see them
            sys.argv.pop(i + 1)
            sys.argv.pop(i + 1)
            break
        elif arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            sys.argv.pop(i + 1)
            break
    
    cfg = load_config(config_path)
    run_quick_test(cfg)