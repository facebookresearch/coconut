import torch
import torch.distributed as dist
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset as HFDataset
import gc
import os
import re

from utils import Config


def compute_final_answer_reward(predicted_answer: str, ground_truth_answer: str) -> float:
    """
    Compute reward based on whether the final answer matches the ground truth.
    Returns 1.0 for correct answer, 0.0 for incorrect.
    
    Args:
        predicted_answer: The answer predicted by the model
        ground_truth_answer: The correct answer
        
    Returns:
        float: 1.0 if answers match, 0.0 otherwise
    """
    # Clean and normalize answers
    pred_clean = predicted_answer.replace(",", "").strip().lower()
    gt_clean = ground_truth_answer.replace(",", "").strip().lower()
    
    # Exact match
    if pred_clean == gt_clean:
        return 1.0
    
    # Try numeric comparison for numerical answers
    try:
        pred_num = float(pred_clean)
        gt_num = float(gt_clean)
        # Allow small floating point tolerance
        if abs(pred_num - gt_num) < 1e-5:
            return 1.0
    except (ValueError, TypeError):
        pass
    
    return 0.0


def compute_format_reward(generated_text: str) -> float:
    """
    Compute reward based on whether the reasoning steps follow the <<computation>> format.
    Returns a score between 0.0 and 1.0 based on format adherence.
    
    Args:
        generated_text: The full generated text including reasoning and answer
        
    Returns:
        float: Score between 0.0 and 1.0 for format quality
    """
    # Extract the reasoning part (before the final answer with ###)
    if "###" in generated_text:
        reasoning_part = generated_text.split("###")[0]
    else:
        reasoning_part = generated_text
    
    # Count valid computation steps in <<...>> format
    # Valid format: <<expression=result>>
    valid_steps = re.findall(r'<<[^<>]+>>', reasoning_part)
    
    # If no steps found, give minimal reward
    if len(valid_steps) == 0:
        return 0.1
    
    # Check if steps contain '=' (proper format)
    steps_with_equals = sum(1 for step in valid_steps if '=' in step)
    
    # Score based on proportion of properly formatted steps
    format_score = steps_with_equals / len(valid_steps) if len(valid_steps) > 0 else 0.0
    
    # Bonus for having multiple reasoning steps (incentivize showing work)
    step_count_bonus = min(len(valid_steps) / 5.0, 0.2)  # Up to 0.2 bonus for 5+ steps
    
    return min(format_score + step_count_bonus, 1.0)


def compute_grpo_reward(
    generated_text: str,
    ground_truth_answer: str,
    reward_weight_final_answer: float = 1.0,
    reward_weight_format: float = 0.1,
) -> float:
    """
    Compute the total GRPO reward combining final answer and format rewards.
    
    Args:
        generated_text: The full generated text including reasoning and answer
        ground_truth_answer: The correct answer
        reward_weight_final_answer: Weight for the final answer reward component
        reward_weight_format: Weight for the format reward component
        
    Returns:
        float: Combined weighted reward score
    """
    # Extract predicted answer (after ###)
    if "###" in generated_text:
        predicted_answer = generated_text.split("###")[-1].strip()
    else:
        # If no ### marker, try to get the last line or token
        predicted_answer = generated_text.strip().split("\n")[-1].strip()
    
    # Compute individual rewards
    answer_reward = compute_final_answer_reward(predicted_answer, ground_truth_answer)
    format_reward = compute_format_reward(generated_text)
    
    # Combine rewards with weights
    total_reward = (
        reward_weight_final_answer * answer_reward +
        reward_weight_format * format_reward
    )
    
    return total_reward

# Define reward function for GRPO
# GRPO calls reward function with keyword arguments:
# reward_func(prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs)
def reward_function(prompts, completions, *, answer, completion_ids=None, **kwargs):
        """
        Compute rewards for generated completions.
        
        Args:
            prompts: list of input prompt strings
            completions: list of generated completion strings
            answer: list of ground truth answer strings
            completion_ids: list of token IDs for completions (optional)
            **kwargs: additional arguments (e.g., trainer_state)
            
        Returns:
            list[float]: Reward scores for each completion
        """
        rewards = []
        answer_rewards = []
        format_rewards = []
        
        # Match each completion with its ground truth
        for idx, (prompt, completion, gt_answer) in enumerate(zip(prompts, completions, answer)):
            # Extract predicted answer (after ###)
            if "###" in completion:
                predicted_answer = completion.split("###")[-1].strip()
            else:
                # If no ### marker, try to get the last line
                predicted_answer = completion.strip().split("\n")[-1].strip()
            
            # Compute individual reward components
            answer_reward = compute_final_answer_reward(predicted_answer, gt_answer)
            format_reward = compute_format_reward(completion)
            
            # Compute total reward
            reward = 1.0 * answer_reward + 0.1 * format_reward
            
            rewards.append(reward)
            answer_rewards.append(answer_reward)
            format_rewards.append(format_reward)
            
            # Print first few examples for debugging
            if idx < 3:
                print(f"\n=== Reward Debug (sample {idx}) ===")
                print(f"Prompt: {prompt[:100]}...")
                print(f"Completion: {completion[:200]}...")
                print(f"Ground truth: {gt_answer}")
                print(f"Predicted: {predicted_answer}")
                print(f"Answer reward: {answer_reward}, Format reward: {format_reward}, Total: {reward}")
        
        # Print aggregate statistics
        if len(rewards) > 0:
            import numpy as np
            print(f"\n=== Batch Reward Statistics ===")
            print(f"Mean answer reward: {np.mean(answer_rewards):.4f}")
            print(f"Mean format reward: {np.mean(format_rewards):.4f}")
            print(f"Mean total reward: {np.mean(rewards):.4f}")
            print(f"Answer correct rate: {sum(r > 0.5 for r in answer_rewards) / len(answer_rewards):.2%}")
        
        return rewards

def train_grpo_style(
    configs: Config,
    epoch: int,
    local_rank: int,
    model,
    pbar,
    rank: int,
    save_dir: str,
    tokenizer,
    total_train_steps: int,
    base_dataset_train,
    wandb_run,
    world_size: int,
    max_new_tokens: int,
):
    """
    GRPO training style using TRL's GRPOTrainer.
    Generates multiple completions per prompt and optimizes based on rewards.
    
    Args:
        configs: Configuration object
        epoch: Current training epoch
        local_rank: Local GPU rank
        model : original wrapped model
        pbar: Progress bar
        rank: Global rank
        save_dir: Directory to save checkpoints
        tokenizer: Model tokenizer
        total_train_steps: Total training steps so far
        base_dataset_train: Base training dataset
        wandb_run: Wandb run object for logging
        world_size: Number of GPUs
        max_new_tokens: Maximum tokens for generation
    """
    
    # Prepare dataset for GRPO training
    # GRPO needs prompts (questions) and reference answers for reward computation
    train_prompts = []
    train_answers = []

    grpo_dataset = []
    
    for idx in range(len(base_dataset_train)):
        sample = base_dataset_train[idx]
        # Construct the prompt (question only)
        question_tokens = sample["question_tokenized"]
        prompt_text = tokenizer.decode(question_tokens, skip_special_tokens=True)
        train_prompts.append(prompt_text)
        
        # Get the ground truth answer
        answer_tokens = sample["answer_tokenized"]
        answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        # Extract just the answer (remove ### prefix if present)
        if "###" in answer_text:
            answer_text = answer_text.split("###")[-1].strip()
        train_answers.append(answer_text)
        grpo_dataset.append({
            "prompt": prompt_text,
            "answer": answer_text,
        })
    
    # Create GRPO config
    grpo_config = GRPOConfig(
        output_dir=save_dir,
        learning_rate=configs.lr,
        per_device_train_batch_size=configs.batch_size_training,
        gradient_accumulation_steps=configs.gradient_accumulation_steps,
        max_steps=getattr(configs, 'max_steps', 500),
        max_grad_norm=getattr(configs, 'max_grad_norm', 0.1),
        warmup_steps=0,
        logging_steps=getattr(configs, 'logging_steps', 10),
        save_steps=500,
        bf16=configs.bf16,
        remove_unused_columns=False,
        # Disable GRPOTrainer's built-in logging - we'll manually log to our existing wandb_run
        report_to="none",
        # GRPO specific parameters
        max_prompt_length=getattr(configs, 'max_prompt_length', 1024),
        max_completion_length=getattr(configs, 'max_completion_length', 512),
        num_generations=getattr(configs, 'num_generations', 8),
        temperature=getattr(configs, 'temperature', 1.0),
        top_p=getattr(configs, 'top_p', 1.0),
        top_k=getattr(configs, 'top_k', None),
        beta=getattr(configs, 'beta', 0.0),
        num_iterations=getattr(configs, 'num_iterations', 1),
        epsilon=getattr(configs, 'epsilon', 0.2),
        scale_rewards=getattr(configs, 'scale_rewards', 'group'),
        # loss_type=getattr(configs, 'loss_type', 'dapo'),
        mask_truncated_completions=getattr(configs, 'mask_truncated_completions', False),
        shuffle_dataset=getattr(configs, 'shuffle_dataset', True),
    )
    

    
    # Prepare dataset in TRL format
    # GRPO trainer expects a dataset with 'prompt' field (string prompts, not tokenized)
    grpo_dataset = HFDataset.from_list(grpo_dataset)
    
    # Get the unwrapped model for GRPO trainer
    # GRPO trainer expects the base model without FSDP/DDP wrapper
    unwrapped_model = model
    
    # Initialize GRPO trainer
    try:
        if rank == 0:
            print(f"Initializing GRPO trainer for epoch {epoch + 1}")
            print(f"Training on {len(train_prompts)} prompts with {grpo_config.num_generations} generations each")
        
        grpo_trainer = GRPOTrainer(
            model=unwrapped_model,
            reward_funcs=reward_function,
            args=grpo_config,
            train_dataset=grpo_dataset,
            processing_class=tokenizer,
        )
        
        # Train
        if rank == 0:
            print(f"Starting GRPO training for epoch {epoch + 1}")
        
        grpo_trainer.train()
        
        # Update the parallel_model with trained weights
        # This ensures the FSDP wrapped model gets the updates
        # if hasattr(parallel_model, 'module'):
        model.load_state_dict(unwrapped_model.state_dict())
        
        # Log metrics to the same wandb run as the rest of the code
        if wandb_run and rank == 0:
            # Extract metrics from the trainer's log history
            if grpo_trainer.state.log_history:
                metrics = grpo_trainer.state.log_history[-1]
                # Log GRPO-specific metrics with grpo/ prefix to keep them organized
                grpo_metrics = {
                    "grpo/epoch": epoch + 1,
                }
                # Add all numeric metrics from trainer
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and k not in ['epoch', 'step']:
                        # Use grpo/ prefix for clarity
                        grpo_metrics[f"grpo/{k}"] = v
                
                wandb_run.log(grpo_metrics)
                
                if rank == 0:
                    print(f"Logged GRPO metrics to wandb: {list(grpo_metrics.keys())}")
        
    except Exception as e:
        if rank == 0:
            print(f"Error during GRPO training: {e}")
            import traceback
            traceback.print_exc()
            print("Raising exception - GRPO training failed")
        # Raise to let user know there's an issue
        raise
    
    pbar.close()
    if torch.cuda.is_available():
        dist.barrier()
    
    # Save checkpoint
    if not configs.save_only_improve and not configs.debug and not configs.only_eval:

        states = unwrapped_model.state_dict()
            
        if rank == 0:
            torch.save(states, os.path.join(save_dir, f"checkpoint_{epoch + 1}"))
            print("Saving GRPO model checkpoint.")

        if torch.cuda.is_available():
            dist.barrier()
            torch.cuda.empty_cache()

        del states
        gc.collect()

