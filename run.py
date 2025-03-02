

# Standard PyTorch imports for deep learning
import torch
import torch.distributed  # For distributed training
import torch.optim as optim  # For optimization algorithms
from transformers import AutoModelForCausalLM, AutoTokenizer  # For loading pretrained LLMs

# Weights & Biases for experiment tracking
import wandb

# Distributed training utilities
from torch.nn.parallel import DistributedDataParallel as DDP  # For parallel training across GPUs
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # For memory-efficient model sharding
import torch.distributed as dist  # For distributed communication
from torch.utils.data.distributed import DistributedSampler  # For distributing data across GPUs
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy  # For FSDP wrapping policy
# Model-specific imports for FSDP wrapping
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

# Project-specific imports
from coconut import Coconut  # The Coconut model wrapper
from dataset import (
    get_dataset,  # Loads dataset
    get_question_latent_dataset,  # Prepares dataset for question generation
    get_cot_latent_dataset,  # Prepares dataset for chain-of-thought training
    MyCollator,  # Handles batching and padding
)

# Utility imports
from tqdm import tqdm  # For progress bars
from copy import copy  # For deep copying objects
import itertools  # For iterating over multiple iterables
import os, sys  # For file system operations
import yaml  # For parsing YAML configuration files
import json  # For parsing JSON files
import gc  # For garbage collection
import argparse  # For command-line arguments
import functools  # For higher-order functions
from utils import Config, set_seed  # Project utilities


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")  # Path to the configuration file
    args = parser.parse_args()

    # Initialize distributed training environment
    dist.init_process_group("nccl")  # NCCL backend is optimized for GPU communication
    local_rank = int(os.environ["LOCAL_RANK"])  # GPU ID for the current process
    rank = int(os.environ["RANK"])  # Global rank of the current process
    world_size = int(os.environ["WORLD_SIZE"])  # Total number of processes
    torch.cuda.set_device(local_rank)  # Set the current GPU device

    # Load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    # Print config only on the main process (rank 0)
    if rank == 0:
        print("Config:", config_dict)

    # Create config object and set random seed for reproducibility
    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)  # Directory to save model checkpoints

    # Create save directory if it doesn't exist (only on rank 0)
    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    # Wait for all processes to reach this point
    torch.distributed.barrier()
    cur_ckpts = os.listdir(save_dir)  # List existing checkpoints

    # Handle checkpoint resumption logic
    # Check if the job was preempted and needs to be resumed
    if len(cur_ckpts) > 0 and not configs.only_eval:
        # If there are previous checkpoints and we're not just evaluating,
        # we need to resume from the latest checkpoint
        if rank == 0:
            print(
                f"Warning: found previous run and gonna resume from that. the inputted `resume` argument is ignored!"
            )

        # Get list of checkpoint files and sort by epoch number
        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))

        # Get the latest checkpoint
        latest_checkpoint = checkpoints[-1] if checkpoints else None
        configs.resume = int(latest_checkpoint.split("_")[1])  # Extract epoch number
        load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)

        configs.load_model_path = load_dir
        print(f"Loading from previous run epoch_{configs.resume}!")

    elif configs.resume != 0:
        # If manually specifying resume epoch
        if configs.load_model_path == "None":
            print(
                f"Warning: you want to skip the first {configs.resume} but you are not loading any existing checkpoint!"
            )
            # This is not an intended use case
        print(
            f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs"
        )

    # Load pretrained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS token
    
    # Add special tokens for latent space modeling
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    
    # Get token IDs for the special tokens
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    loaded = False  # Track if model weights have been loaded

    # Load pretrained weights if specified
    if configs.load_model_path != "None":
        saved_weights = torch.load(
            configs.load_model_path, map_location=torch.device(rank)
        )

        # Handle different loading scenarios
        if configs.coconut and not any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # Loading a base model into coconut model
            # e.g., for GSM8k, using a SFTed model to skip stage 0
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

        elif not configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # Cannot load coconut model weights into a standard causal LM
            raise ValueError("Cannot load coconut model weights into a causallm model")

        elif configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # Loading from a preempted run - will handle later
            pass

        else:
            # Resume or evaluate SFT model
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    # Initialize new token embeddings if needed
    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        # If using new tokens, initialize their embeddings and LM heads
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        
        # Initialize new token embeddings with existing token embedding
        # This helps stabilize the training
        for token_id in [latent_id, start_id, end_id]:
            target_embedding = embeddings.weight.data[token_id]
            embeddings.weight.data[token_id] = target_embedding
            # The input embeddings and lm heads are tied in GPT2, but we set them explicitly
            lm_head = model.lm_head
            lm_head.weight.data[token_id] = lm_head.weight.data[target_id]

    # Configure model type based on training mode
    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    # Wrap model in Coconut wrapper if using Coconut
    if configs.coconut:
        model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    # Load model weights if not loaded yet
    if configs.load_model_path != "None" and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

    # Log FSDP initialization
    print(f"Running FSDP on rank = {rank}, world size = {world_size}")
    model = model.to(rank)  # Move model to GPU

    # Define policy for wrapping transformer layers with FSDP
    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            # GPT2Block,       # for GPT2, we don't need to shard layers (it becomes DDP)
            LlamaDecoderLayer  # only shard llama's layers for memory efficiency
        },
    )

    # Convert model to bfloat16 precision if specified
    if configs.bf16:
        model.to(torch.bfloat16)

    # Choose parallelism strategy based on mode
    if configs.only_eval:
        # Use DDP for evaluation mode (to avoid bugs in FSDP)
        parallel_model = DDP(model, device_ids=[rank])
    else:
        # Use FSDP for training mode (for memory efficiency)
        parallel_model = FSDP(
            model, auto_wrap_policy=llama_auto_wrap_policy, device_id=rank
        )

    # Free up memory
    del model

    # Print model architecture (only on rank 0)
    if rank == 0:
        print(parallel_model)

    # Prepare validation data
    # Load ground truth answers and chain-of-thought explanations for evaluation
    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [
        d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))
    ]
    cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]

    # Load validation dataset
    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=32 if configs.debug else 100000000
    )

    # Load training dataset if not in evaluation-only mode
    if not configs.only_eval:
        base_dataset_train = get_dataset(
            configs.train_path, tokenizer, max_size=5000 if configs.debug else 100000000
        )

    # Set maximum token generation length based on dataset
    if "gsm" in configs.val_path:
        max_new_tokens = 64  # GSM8K problems need shorter responses
    else:
        max_new_tokens = 128  # Other datasets may need longer responses

    total_train_steps = 0  # Counter for training steps

    # Initialize Weights & Biases for experiment tracking (only if not debugging)
    if not configs.debug and not configs.only_eval and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])  # Table for logging text examples
    else:
        wandb_run = None

    # Initialize optimizer
    if configs.reset_optimizer:
        optimizer = None
    else:
        optimizer = optim.AdamW(
            parallel_model.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )

    best_acc = 0  # Track best accuracy for model saving

    # Initialize data collator for batching
    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    # Main training loop
    for epoch in range(configs.resume, configs.num_epochs):
        # Determine current training stage based on epoch
        scheduled_stage = (
            0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage
        )
        
        # Prepare validation dataset for generation
        dataset_gen_val = get_question_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            configs,
            start_id,
            latent_id,
            end_id,
            no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
        )

        # Create validation dataloader for generation
        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            batch_size=1,  # Generate one sample at a time
            collate_fn=collator,
            sampler=DistributedSampler(dataset_gen_val, shuffle=False),
        )

        # Training steps (skip if in evaluation-only mode)
        if not configs.only_eval:
            # Prepare training dataset
            dataset_train = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_train,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
                shuffle=True,  # Shuffle training data
            )

            # Create training dataloader
            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=False,  # DistributedSampler handles shuffling
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_train, shuffle=True),
            )

            # Note: The sampler is deterministic even with shuffle=True
            # so we shuffle the dataset when constructing it (at every epoch)

            # Prepare validation dataset for loss computation
            dataset_loss_val = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
            )

            # Create validation dataloader for loss
            valid_loss_dataloader = torch.utils.data.DataLoader(
                dataset_loss_val,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_loss_val, shuffle=False),
            )

            # Reset optimizer if specified
            if configs.reset_optimizer:
                del optimizer
                optimizer = optim.AdamW(
                    parallel_model.parameters(),
                    lr=configs.lr,
                    weight_decay=configs.weight_decay,
                )

            # Set model to training mode
            parallel_model.module.train()

            # Calculate total training steps for progress bar
            total_length = len(train_dataloader) // configs.gradient_accumulation_steps
            
            # Initialize progress bar
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )

            # Training loop
            for step, batch in enumerate(train_dataloader):
                # Log training data examples on first step
                if step == 0 and wandb_run and rank == 0:
                    print("logging training data")
                    cur_bs = len(batch["input_ids"])
                    text_str = ""
                    for data_idx in range(cur_bs):
                        for token_idx in range(len(batch["input_ids"][data_idx])):
                            text_str += (
                                str(batch["input_ids"][data_idx][token_idx].item())
                                + " "
                                + str(batch["labels"][data_idx][token_idx].item())
                                + " "
                                + tokenizer.decode(
                                    batch["input_ids"][data_idx][token_idx]
                                )
                                + "\n"
                            )
                        text_str += "====" * 10 + "\n"
                    text_table.add_data(total_train_steps, text_str)
                    # Copy the table due to a bug in wandb
                    # https://github.com/wandb/wandb/issues/2981
                    wandb_run.log({"data_table": copy(text_table)})

                # Increment step counter
                total_train_steps += 1
                
                # Move batch to GPU
                batch = {
                    key: batch[key].to(rank) for key in batch.keys() if key != "idx"
                }

                # Forward pass
                outputs = parallel_model(**batch)

                # Compute loss and scale by gradient accumulation steps
                loss = outputs.loss / configs.gradient_accumulation_steps
                # Backward pass
                loss.backward()

                # Update parameters after accumulating gradients
                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                # Log training metrics to W&B
                if wandb_run and rank == 0:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float()
                        * configs.gradient_accumulation_steps,
                    }
                    wandb_run.log(log_dict)

                # Update progress bar description
                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}"
                )
            
            # Close progress bar
            pbar.close()
            
            # Synchronize all processes
            dist.barrier()

            # Save model checkpoint if configured
            if (
                not configs.save_only_improve
                and not configs.debug
                and not configs.only_eval
            ):
                # Get model state dictionary
                states = parallel_model.state_dict()
                
                # Save model (only on rank 0)
                if rank == 0:
                    torch.save(
                        states, os.path.join(save_dir, f"checkpoint_{epoch + 1}")
                    )
                    print("saving model.")

                # Wait for save to complete
                dist.barrier()
                
                # Clean up memory
                del states
                gc.collect()
                torch.cuda.empty_cache()

            # Compute validation loss
            total_loss = 0

            with torch.no_grad():
                # Set model to evaluation mode
                parallel_model.module.eval()
                
                # Validation loss computation loop
                for step, batch in enumerate(valid_loss_dataloader):
                    # Move batch to GPU
                    batch = {
                        key: batch[key].to(rank) for key in batch.keys() if key != "idx"
                    }

                    # Forward pass
                    outputs = parallel_model(**batch)
                    loss = outputs.loss
                    
                    # Sum loss across all processes
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    total_loss += loss.item() / world_size

                # Log validation loss to W&B
                if wandb_run and rank == 0:
                    log_dict = {
                        "eval/loss": total_loss / len(valid_loss_dataloader),
                    }
                    wandb_run.log(log_dict)
                    print("eval loss", total_loss / len(valid_loss_dataloader))

        # Validation generation accuracy evaluation
        total_length = len(valid_gen_dataloader)

        # Initialize progress bar for evaluation
        pbar = tqdm(
            colour="blue", desc=f"Test Accuracy", total=total_length, dynamic_ncols=True
        )
        
        # Initialize counters for correct answers, correct CoT, and total examples
        cor, cor_cot, total = (
            torch.tensor(0, device=rank),
            torch.tensor(0, device=rank),
            torch.tensor(0, device=rank),
        )

        with torch.no_grad():
            # Set model to evaluation mode
            parallel_model.module.eval()
            
            # Evaluation loop
            for idx, batch in enumerate(valid_gen_dataloader):
                test_idx = batch["idx"][0]  # Get example index

                # Move batch to GPU, filtering out unnecessary keys
                batch = {
                    k: v.to(rank)
                    for k, v in batch.items()
                    if v != None and k not in ["idx", "position_ids"]
                }
                # Workaround for https://github.com/huggingface/transformers/issues/32492

                # Verify batch size is 1 (generating one example at a time)
                assert len(batch["input_ids"]) == 1
                
                # Get ground truth answers and questions
                answer = answers_val[test_idx.cpu().item()]
                answer_cot = cot_val[test_idx.cpu().item()]
                question = question_val[test_idx.cpu().item()]

                # Increment total counter
                total += 1

                # Generate model output
                # synced_gpus=True in FSDP mode ensures all GPUs generate the same number of examples
                outputs = parallel_model.module.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    synced_gpus=not configs.only_eval,
                )

                # Decode the generated text
                text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract answer and CoT from the output
                answer_output = text_output.split("#")[-1].replace(",", "").strip()
                cot_output = (
                    ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()
                )

                # Print some examples for inspection
                if idx < 5 and rank == 0:
                    print(
                        f"Question {test_idx}: Answer = '{answer}' CoT = '{answer_cot}'"
                    )
                    print(f"Full output: '{tokenizer.decode(outputs[0])}'")
                    print(f"Extracted Output: '{answer_output}'")

                # Update correctness counters
                cor += answer_output == answer  # Check if answer matches
                cor_cot += cot_output == answer_cot  # Check if CoT matches

                # Update progress bar
                pbar.update(1)
                pbar.set_description(
                    f"Test accuracy: {round(float(cor.detach().float() / total.detach().float()), 2)}"
                )

            # Close progress bar
            pbar.close()
            print(f"Device {rank}: Cor={cor}, CoT={cor_cot}, Total={total}")

        # Sum metrics across all processes
        dist.all_reduce(cor_cot, op=dist.ReduceOp.SUM)  # Sum correct CoT counts
        dist.all_reduce(cor, op=dist.ReduceOp.SUM)      # Sum answer correct counts
        dist.all_reduce(total, op=dist.ReduceOp.SUM)    # Sum total examples

        # Convert tensor values to Python scalars
        cor_cot = cor_cot.item()
        cor = cor.item()
        total = total.item()
        
        # Print final results (only on rank 0)
        if rank == 0:
            print(f"Accuracy on validation set: {cor} / {total} = {cor/total}")
            print(f"CoT match on validation set: {cor_cot} / {total} = {cor_cot/total}")
        
        # Flush output to ensure it's displayed
        sys.stdout.flush()

        # Log final metrics to W&B
        if wandb_run:
            wandb_run.log({"eval/acc": cor / total, "eval/cot_em": cor_cot / total})

        # Break loop if only evaluating
        if configs.only_eval:
            break

        # Synchronize processes before next epoch
        dist.barrier()
        
        # Save model if accuracy improved and configured to only save on improvement
        if (
            cor / total > best_acc
            and configs.save_only_improve
            and not configs.debug
            and not configs.only_eval
        ):
            # Get model state
            states = parallel_model.state_dict()

            # Save model (only on rank 0)
            if rank == 0:
                torch.save(states, os.path.join(save_dir, f"checkpoint_{epoch + 1}"))
                print("saving model.")

            # Update best accuracy
            best_acc = cor / total

            # Synchronize processes
            dist.barrier()
            
            # Clean up memory
            del states
            gc.collect()
            torch.cuda.empty_cache()


# Entry point for script execution
if __name__ == "__main__":
    main()