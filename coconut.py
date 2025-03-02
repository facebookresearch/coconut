import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

# Define a named tuple to organize and return multiple values from forward method
Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8  # Maximum number of latent tokens to process


class Coconut(nn.Module):
    """
    Coconut (Continuous Chains of Thought) model that enhances a base language model
    by allowing for intermediate latent tokens that act as continuous thought vectors.
    """
    def __init__(
        self,
        base_causallm,  # Base language model (e.g., GPT2, Llama3)
        latent_token_id,  # Token ID that represents a latent token placeholder
        start_latent_id,  # Token ID marking the start of latent tokens
        end_latent_id,  # Token ID marking the end of latent tokens
        eos_token_id,  # End of sequence token ID
    ):
        # Initialize the parent class (nn.Module)
        super(Coconut, self).__init__()
        
        # Counter for forward passes during generation (used for synced_gpus)
        self.gen_forward_cnt = 0
        
        # Store model parameters
        self.base_causallm = base_causallm  # The underlying language model
        self.latent_token_id = latent_token_id  # ID for latent token placeholders
        self.eos_token_id = eos_token_id  # End of sequence token ID
        self.start_latent_id = start_latent_id  # ID marking start of latent tokens
        self.end_latent_id = end_latent_id  # ID marking end of latent tokens

        # Get the embedding layer from the base model
        # Different model architectures store embeddings in different places
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        """
        Forward pass of the Coconut model.
        
        This method processes input with latent tokens by:
        1. Finding all latent token positions
        2. Iteratively replacing latent tokens with computed hidden states
        3. Calculating loss based on outputs
        """
        # Initialize list to store logits from different forward passes
        logits = []

        # Find all positions where latent tokens appear in the input
        # Returns a tensor with shape (num_latent_tokens, 2) containing [batch_idx, position]
        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        # Group latent token positions by batch item
        # Creates a list of lists where each sublist contains positions of latent tokens for one batch item
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        # Find the maximum number of latent tokens in any batch item
        # This determines how many iterations we'll need
        max_n_latents = max([len(l) for l in latent_lists])

        # Initialize the computation range to process the entire input sequence
        next_compute_range = (0, input_ids.shape[1])
        
        # Convert input token IDs to embeddings
        inputs_embeds = self.embedding(input_ids)

        # If we have latent tokens, adjust the initial computation range to stop at the first latent token
        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # Process only up to the earliest latent token position

        # Initialize KV cache to None (will store key-value pairs for attention)
        kv_cache = None

        # Process each latent token in sequence
        for pass_idx in range(max_n_latents):
            # On first pass or when no latent tokens, kv_cache is None
            if kv_cache == None:
                # First forward pass through the base model
                # Only process tokens up to the next latent token
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,  # Request hidden states for later use
                )
                hidden_states_offset = 0  # No offset for first pass

            else:
                # On subsequent passes, reuse the KV cache for tokens already processed
                # This avoids recomputing attention for tokens we've seen before
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],  # Key cache for past tokens
                        v[:, :, : next_compute_range[0], :],  # Value cache for past tokens
                    )
                    for k, v in kv_cache
                ]

                # Forward pass with KV cache
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],  # Full attention mask up to current position
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,  # Use cached key-values
                    output_hidden_states=True,
                )

                # When using KV cache, hidden states indexing needs adjustment
                hidden_states_offset = next_compute_range[0]
                # This offset accounts for tokens processed in previous passes 
                # that aren't included in the current hidden states

            # Store logits from this pass
            logits.append(outputs.logits)

            # Update the computation range for the next pass
            # If this is the last latent token, process the rest of the sequence
            # Otherwise, move forward one token to process the next latent token
            next_compute_range = (
                next_compute_range[1],  # Start from where we left off
                (
                    input_ids.shape[1]  # If this is the last latent, go to the end
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1  # Otherwise, just process one more token
                ),
            )

            # Get the hidden states from the last layer
            hidden_states = outputs.hidden_states[-1]  # Get the last layer hidden states
            kv_cache = outputs.past_key_values  # Update KV cache for next pass

            # Now we replace latent tokens with continuous thought vectors
            # These are derived from hidden states of preceding tokens

            # Determine which latent tokens to replace in this pass
            # For each batch item, get the latent token at position pass_idx
            filling_indices = [
                (instance_idx, mask_list[pass_idx])  # (batch_idx, token_position)
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx  # Only if this batch item has enough latent tokens
            ]

            # To avoid in-place operations which can cause issues with autograd,
            # we break down the embeddings tensor into individual tensors
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]  # Get embedding for each position
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # Replace latent token embeddings with continuous thoughts
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                # Replace latent token with the hidden state of the previous token
                # This implements the "continuous chain of thought" mechanism
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            # Reassemble the modified embeddings into a single tensor
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # Final forward pass to process remaining tokens after all latent tokens
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=(
                # Use KV cache if available
                [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )

        # Add final logits to our collection
        logits.append(outputs.logits)

        # Increment the forward pass counter (used for synced_gpus during generation)
        self.gen_forward_cnt += max_n_latents + 1

        # Concatenate all logits from different passes
        logits = torch.cat(logits, dim=-2)
        
        # Shift logits and labels for language modeling loss calculation
        # This aligns predictions with targets (predict next token from current token)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Calculate cross-entropy loss
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),  # Flatten predictions
            shift_labels.view(-1)  # Flatten targets
        )

        # Return loss, embeddings, and logits in a named tuple
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        """Set the model to training mode"""
        self.base_causallm.train()

    def eval(self):
        """Set the model to evaluation mode"""
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used in generation
        max_new_tokens=16,  # Maximum number of tokens to generate
        output_embedding=False,  # Whether to return embeddings (for analysis)
        synced_gpus=False,  # Whether to sync with other GPUs (for FSDP)
        **kwargs
    ):
        """
        Generate text autoregressively from the model.
        
        This method:
        1. Processes input through Coconut's forward pass
        2. Generates tokens one by one using greedy decoding
        3. Supports returning embeddings for analysis
        4. Handles synchronization in distributed training
        """
        # Reset the forward pass counter
        self.gen_forward_cnt = 0

        # Currently only supports batch size of 1
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        # Initialize the output tokens list with input tokens
        tokens = input_ids[0].detach().tolist()

        # Create a placeholder for labels (not used for generation, but needed for forward pass)
        labels = input_ids.clone()  # placeholder. not used.
        
        # Initial forward pass to process the input sequence
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),  # All tokens attended to
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),  # Position IDs
        )
        
        # Get the embeddings after processing latent tokens
        inputs_embeds = outputs.inputs_embeds

        # Generate the first token using the logits from the initial forward pass
        next_token = torch.argmax(outputs.logits[0, -1]).item()  # Greedy decoding
        tokens.append(next_token)  # Add to output tokens
        
        # Convert the new token to embedding and add to our input embeddings
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # Generate remaining tokens one by one
        for _ in range(max_new_tokens - 1):
            # Forward pass through the base model (no need for Coconut's forward method now)
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1  # Increment forward pass counter
            
            # Get the next token (greedy decoding)
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            
            # Stop if we hit the end of sequence token
            if next_token == self.eos_token_id:
                break
                
            # Add token to our output
            tokens.append(next_token)
            
            # Convert to embedding and add to input
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        # For distributed training with FSDP, ensure all devices do the same number of forward passes
        # This prevents hanging during synchronization
        if synced_gpus:
            # Do dummy forward passes until we reach the expected count
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        # Return embeddings if requested (for analysis)
        if output_embedding:
            # Return both tokens and embeddings
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds
        else:
            # Return only tokens
            return torch.tensor(tokens).view(1, -1)