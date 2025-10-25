"""
Flax evaluation pipeline for NSA transformer model.
Implements inference and test-time adaptation using JAX/Flax.
"""

import argparse
import json
import os
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path
from shutil import rmtree

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from tqdm import tqdm

from auxilaries.generate_transformation import generate_samples
from plots import return_task_grid
from small_transformer_based.flax_model import FlaxCustomTransformer, count_parameters
from small_transformer_based.flax_train import (
    CustomDataset,
    CustomTokenizer,
    TrainState,
    collate_fn_jax,
    cross_entropy_loss,
    extract_input_output_pairs,
)
from utils import PathConfig
from task import Task


def format_grid_for_tokenizer(grid_data):
    """
    Format grid data directly for tokenizer without LLM prompt generation.
    Takes train data (list of input-output pairs) and formats as text.
    """
    examples = []
    for example in grid_data:
        input_grid = example["input"]
        output_grid = example["output"]
        # Convert grids to pipe-separated format
        input_grid_str = "\n".join(["|".join(map(str, row)) for row in input_grid])
        output_grid_str = "\n".join(["|".join(map(str, row)) for row in output_grid])
        examples.append(f"Input:\n{input_grid_str}\nOutput:\n{output_grid_str}")
    return "\n".join(examples)


def create_predict_fn(model):
    """
    Create a JIT-compiled prediction function for a given model.

    Args:
        model: Flax model

    Returns:
        JIT-compiled prediction function
    """

    @jax.jit
    def predict_batch(params, input_ids, attention_mask):
        """
        Batch inference (JIT-compiled for optimal performance on Apple Silicon).

        Args:
            params: Model parameters
            input_ids: Input token IDs of shape [batch, seq_len]
            attention_mask: Attention mask of shape [batch, seq_len]

        Returns:
            Logits of shape [batch, num_cls_tokens, vocab_size]
        """
        return model.apply(params, input_ids, attention_mask=attention_mask, deterministic=True)

    return predict_batch


def extract_first_from_logits(tokenizer, logits, index, number, already_added=None):
    """
    Extract top-k predictions from logits for a specific classification token.

    Args:
        tokenizer: Tokenizer instance
        logits: Model logits of shape [batch, num_cls_tokens, vocab_size]
        index: Which classification token to extract from (0, 1, or 2)
        number: Number of top predictions to extract
        already_added: List of predictions to filter out

    Returns:
        List of top-k transformation names
    """
    if already_added is None:
        already_added = []

    # Get top 30 predictions for this classification token
    # logits[0][index] has shape [vocab_size]
    top_k_indices = jnp.argsort(logits[0][index])[-30:][::-1]  # Top 30 in descending order

    # Decode token IDs to transformation names
    corresponding_tokens = [tokenizer.decode([int(token_id)]).strip() for token_id in top_k_indices]

    # Filter out already added predictions and "no_trans"
    top_predictions = [x for x in corresponding_tokens if x not in already_added + ["no_trans"]]

    # Return top 'number' predictions
    return top_predictions[:number]


def batch_predict_transformations(state, model, tokenizer, task_data_list):
    """
    Predict transformations for multiple tasks in parallel using batch inference.

    Args:
        state: Train state with model parameters
        model: Flax model
        tokenizer: Tokenizer instance
        task_data_list: List of (task_id, task_file, data_path) tuples

    Returns:
        Dictionary mapping task_id to list of predicted transformations
    """
    # Create JIT-compiled prediction function
    predict_batch = create_predict_fn(model)

    all_predictions = {}

    # Prepare all inputs
    all_input_ids = []
    valid_tasks = []

    for task_id, task_file, data_path in task_data_list:
        try:
            grid = return_task_grid(task_file)["train"]
            prompt = format_grid_for_tokenizer(grid)
            prompt = extract_input_output_pairs(prompt)
            input_ids = tokenizer.encode(prompt)
            all_input_ids.append(np.array(input_ids, dtype=np.int32))
            valid_tasks.append((task_id, task_file, data_path))
        except Exception as e:
            print(f"Error preparing task {task_id}: {e}")
            all_predictions[task_id] = []

    if not valid_tasks:
        return all_predictions

    # Pad sequences to uniform length
    max_len = max(len(ids) for ids in all_input_ids)
    padded_inputs = []
    for input_ids in all_input_ids:
        padded = np.pad(input_ids, (0, max_len - len(input_ids)), constant_values=0)
        padded_inputs.append(padded)

    # Convert to JAX arrays
    padded_input_ids = jnp.array(padded_inputs)
    attention_mask = (padded_input_ids != tokenizer.vocab.get("<PAD>", 0)).astype(jnp.float32)

    # Batch inference
    batch_logits = predict_batch(state.params, padded_input_ids, attention_mask)

    # Extract predictions for each task
    for idx, (task_id, task_file, data_path) in enumerate(valid_tasks):
        logits = batch_logits[idx : idx + 1]  # Keep batch dim

        # Check if second and third tokens are needed
        second_token = jnp.argmax(logits[0][2]).item()
        include_second = tokenizer.decode([second_token]) != "no_trans"
        include_third = False  # Only 3 cls tokens (0,1,2)

        top3_predictions = []
        if not include_second and not include_third:
            to_consider = 5
            top3_predictions = extract_first_from_logits(
                tokenizer=tokenizer,
                logits=logits,
                index=0,
                number=to_consider,
                already_added=[],
            )
        if include_second:
            to_consider = 4
            preds_first = extract_first_from_logits(
                tokenizer=tokenizer, logits=logits, index=0, number=to_consider
            )
            preds_second = extract_first_from_logits(
                tokenizer=tokenizer,
                logits=logits,
                index=1,
                number=to_consider,
                already_added=preds_first,
            )
            top3_predictions += preds_first + preds_second
        if include_third:
            to_consider = 3
            preds_third = extract_first_from_logits(
                tokenizer=tokenizer,
                logits=logits,
                index=2,
                number=to_consider,
                already_added=top3_predictions,
            )
            top3_predictions += preds_third

        # Filter and deduplicate
        top3_predictions = [pred for pred in top3_predictions if pred != "no_trans"]
        top3_predictions = list(dict.fromkeys(top3_predictions))
        all_predictions[task_id] = top3_predictions

    return all_predictions


def evaluate_with_tta(
    state, model, tokenizer, task_id, task_file, data_path, tta_epochs=15, rng=None
):
    """
    Perform test-time adaptation for a single task.

    Args:
        state: Train state with model parameters
        model: Flax model
        tokenizer: Tokenizer instance
        task_id: Task identifier (without .json)
        task_file: Task filename (with .json)
        data_path: Path to dataset directory
        tta_epochs: Number of epochs for fine-tuning
        rng: JAX random key

    Returns:
        List of predicted transformations
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # Create JIT-compiled prediction function
    predict_batch = create_predict_fn(model)

    # Clone state for task-specific fine-tuning
    tta_state = TrainState.create(
        apply_fn=state.apply_fn,
        params=deepcopy(state.params),
        tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(5e-5)),
        dropout_rng=rng,
    )

    # Generate or load task-specific synthetic data
    output_folder = PathConfig.get_tta_dir(task_id)
    all_transformations_path = PathConfig.get_tta_task_file(task_id)

    if os.path.exists(all_transformations_path):
        try:
            with open(all_transformations_path) as f:
                tta_data = json.load(f)
        except Exception as e:
            print(f"Error loading TTA data for {task_id}: {e}. Regenerating data.")
            os.remove(all_transformations_path)
            if os.path.exists(output_folder):
                rmtree(output_folder)
            tta_data = []
    else:
        tta_data = []

    # Check consistency
    if os.path.exists(output_folder) and len(os.listdir(output_folder)) != len(tta_data):
        if abs(len(os.listdir(output_folder)) - len(tta_data)) > 100 and len(tta_data) < 2000:
            print(f"Inconsistent TTA data for {task_id}. Regenerating.")
            if os.path.exists(all_transformations_path):
                os.remove(all_transformations_path)
            rmtree(output_folder)
            tta_data = []

    # Generate synthetic data if needed
    if not tta_data:
        Path(output_folder).mkdir(exist_ok=True, parents=True)
        print(f"--- Data Generation for {task_id} ---")
        generate_samples(
            number_of_samples=2500,
            output_folder=output_folder,
            all_transformations_path=all_transformations_path,
            no_of_trans=3,
            transformation_ops=None,
            chosen_task=task_id,
        )
        with open(all_transformations_path) as f:
            tta_data = json.load(f)

    # Filter data
    tta_data = [x for x in tta_data if len(x["input"]) < 12000]
    tta_dataset = CustomDataset(tta_data, tokenizer, max_length=2048)

    print(f"--- Model Training for {tta_epochs} Epochs ---")
    padding_idx = tokenizer.vocab["<PAD>"]

    # Fine-tune for specified epochs
    for epoch in range(tta_epochs):
        # Create batches manually
        batch_size = 16
        num_batches = len(tta_dataset) // batch_size

        # Shuffle indices
        rng, shuffle_rng = jax.random.split(rng)
        indices = jax.random.permutation(shuffle_rng, len(tta_dataset))

        progress_bar = tqdm(range(num_batches), desc=f"TTA Epoch {epoch + 1}")

        for batch_idx in progress_bar:
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]

            batch_data = [tta_dataset[int(i)] for i in batch_indices]
            batch = collate_fn_jax(batch_data, padding_value=padding_idx)

            # Training step
            input_ids, output_ids = batch
            dropout_rng, new_dropout_rng = jax.random.split(tta_state.dropout_rng)

            def loss_fn(params):
                attention_mask = (input_ids != padding_idx).astype(jnp.float32)
                logits = tta_state.apply_fn(
                    params,
                    input_ids,
                    attention_mask=attention_mask,
                    deterministic=False,
                    rngs={"dropout": dropout_rng},
                )
                return cross_entropy_loss(logits, output_ids, padding_idx)

            loss, grads = jax.value_and_grad(loss_fn)(tta_state.params)
            tta_state = tta_state.apply_gradients(grads=grads)
            tta_state = tta_state.replace(dropout_rng=new_dropout_rng)

            progress_bar.set_postfix(loss=f"{float(loss):.4f}")

    # Predict using fine-tuned model
    try:
        grid = return_task_grid(task_file)["train"]
    except Exception as e:
        print(f"Error retrieving grid for task {task_file}: {e}")
        return []

    prompt = format_grid_for_tokenizer(grid)
    prompt = extract_input_output_pairs(prompt)
    input_ids = tokenizer.encode(prompt)
    input_ids = jnp.array([input_ids], dtype=jnp.int32)
    attention_mask = (input_ids != tokenizer.vocab.get("<PAD>", 0)).astype(jnp.float32)

    # Inference with fine-tuned model
    logits = predict_batch(tta_state.params, input_ids, attention_mask)

    # Extract predictions
    second_token = jnp.argmax(logits[0][2]).item()
    include_second = tokenizer.decode([second_token]) != "no_trans"
    include_third = False

    top3_predictions = []
    if not include_second and not include_third:
        to_consider = 5
        top3_predictions = extract_first_from_logits(
            tokenizer=tokenizer,
            logits=logits,
            index=0,
            number=to_consider,
            already_added=[],
        )
    if include_second:
        to_consider = 4
        preds_first = extract_first_from_logits(
            tokenizer=tokenizer, logits=logits, index=0, number=to_consider
        )
        preds_second = extract_first_from_logits(
            tokenizer=tokenizer,
            logits=logits,
            index=1,
            number=to_consider,
            already_added=preds_first,
        )
        top3_predictions += preds_first + preds_second
    if include_third:
        to_consider = 3
        preds_third = extract_first_from_logits(
            tokenizer=tokenizer,
            logits=logits,
            index=2,
            number=to_consider,
            already_added=top3_predictions,
        )
        top3_predictions += preds_third

    top3_predictions = [pred for pred in top3_predictions if pred != "no_trans"]
    top3_predictions = list(dict.fromkeys(top3_predictions))

    return top3_predictions


def _solve_single_task(task_data):
    """Worker function for parallel task solving."""
    task_id, task_file, data_path, predictions = task_data
    task_key = task_file.replace(".json", "")

    try:
        task = Task(
            os.path.join(data_path, task_file),
            proposed_transformations=predictions,
        )
        solved = task.solve()
        if solved:
            print(f"Task {task_id} solved successfully with predicted transformations.")
        else:
            print(f"Task {task_id} could not be solved with predicted transformations.")
    except Exception as e:
        print(f"Error solving task {task_id} with predicted transformations: {e}")
        solved = False

    return task_key, solved, predictions


def evaluate_true(state, model, tokenizer, tta=True, tta_epochs=15, num_workers=3, max_tasks=None):
    """
    Main evaluation function maintaining existing interface.

    Args:
        state: Train state with model parameters
        model: Flax model
        tokenizer: Tokenizer instance
        tta: Whether to use test-time adaptation
        tta_epochs: Number of epochs for TTA fine-tuning
        num_workers: Number of parallel workers for task solving
        max_tasks: Maximum number of tasks to evaluate per split

    Returns:
        Dictionary with evaluation results
    """
    dataset_splits = {"train": "dataset/training", "val": "dataset/validation"}
    results = {"train": {}, "val": {}}
    proposed_transformations_dict = {"train": {}, "val": {}}

    for split, directory in dataset_splits.items():
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist. Skipping {split} split.")
            continue

        task_files = [f for f in os.listdir(directory) if f.endswith(".json")]
        task_ids = [f for f in task_files]

        # Determine results file name
        if split == "val" and tta:
            results_file = PathConfig.get_data_path("arga_evaluation_tta.json")
        elif split == "train" and tta:
            results_file = PathConfig.get_data_path(f"arga_training_tta_epoch{tta_epochs}.json")
        elif split == "train" and not tta:
            results_file = PathConfig.get_data_path("arga_training_no_tta.json")
        elif split == "val" and not tta:
            results_file = PathConfig.get_data_path("arga_evaluation_no_tta.json")
        else:
            raise ValueError("Invalid dataset split encountered.")

        # Load existing results if available
        if os.path.exists(results_file):
            print(f"File {results_file} found and will be updated!")
            with open(results_file) as f:
                loaded_results = json.load(f)
                if split in loaded_results:
                    results[split].update(loaded_results[split])
        else:
            print(f"No existing {results_file} found. Creating a new one.")

        correct_solved = sum(
            1 for task_id in results[split] if results[split][task_id].get("solved")
        )

        # Apply max_tasks limit if specified
        if max_tasks is not None:
            task_ids = task_ids[:max_tasks]

        total_tasks = len(task_ids)
        print(
            f"Processing {total_tasks} tasks in the '{split}' split. "
            f"{correct_solved} already solved."
        )
        print(f"Using {num_workers} parallel workers for task solving (when not using TTA).")

        # Filter out tasks already solved
        tasks_to_process = []
        for task_id_json in task_ids:
            task_id = task_id_json
            task_key = task_id_json.replace(".json", "")
            if task_key in results[split]:
                if results[split][task_key].get("solved"):
                    continue
            tasks_to_process.append((task_key, task_id, directory))

        if not tasks_to_process:
            print(f"All tasks in '{split}' split already solved.")
            continue

        print(f"Tasks to process: {len(tasks_to_process)}")

        # TTA mode: sequential processing
        if tta:
            print("TTA mode: using sequential evaluation (not parallelized)")
            rng = jax.random.PRNGKey(0)

            for task_key, task_id, data_path in tqdm(
                tasks_to_process, desc=f"Evaluating {split.capitalize()} Tasks (TTA)"
            ):
                # Perform TTA and get predictions
                top3_predictions = evaluate_with_tta(
                    state, model, tokenizer, task_key, task_id, data_path, tta_epochs, rng
                )

                proposed_transformations_dict[split][task_id] = top3_predictions

                # Solve task with predictions
                try:
                    task = Task(
                        os.path.join(data_path, task_id),
                        proposed_transformations=top3_predictions,
                    )
                    solved = task.solve()
                    if solved:
                        print(f"Task {task_id} solved successfully with predicted transformations.")
                        correct_solved += 1
                    else:
                        print(f"Task {task_id} could not be solved with predicted transformations.")
                except Exception as e:
                    print(f"Error solving task {task_id} with predicted transformations: {e}")
                    solved = False

                results[split][task_key] = {
                    "solved": solved,
                    "predictions": top3_predictions,
                }

                # Save results incrementally
                with open(results_file, "w") as f:
                    json.dump({split: results[split]}, f, indent=4)

        else:
            # Non-TTA: Use hybrid parallel approach
            print("\n=== Phase 1: Batch Model Inference ===")
            task_data_list = [(key, tid, data_path) for key, tid, data_path in tasks_to_process]
            all_predictions = batch_predict_transformations(state, model, tokenizer, task_data_list)
            print(f"Predicted transformations for {len(all_predictions)} tasks")

            print("\n=== Phase 2: Parallel Task Solving ===")
            # Prepare task solving jobs
            solve_jobs = [
                (task_key, task_id, data_path, all_predictions.get(task_key, []))
                for task_key, task_id, data_path in tasks_to_process
            ]

            # Parallel task solving with progress bar
            if num_workers > 1:
                with Pool(processes=num_workers) as pool:
                    solve_results = list(
                        tqdm(
                            pool.imap(_solve_single_task, solve_jobs),
                            total=len(solve_jobs),
                            desc=f"Solving {split.capitalize()} Tasks",
                        )
                    )
            else:
                # Sequential fallback
                solve_results = [
                    _solve_single_task(job)
                    for job in tqdm(solve_jobs, desc=f"Solving {split.capitalize()} Tasks")
                ]

            # Collect results
            for task_key, solved, predictions in solve_results:
                if solved:
                    correct_solved += 1
                results[split][task_key] = {
                    "solved": solved,
                    "predictions": predictions,
                }
                proposed_transformations_dict[split][task_key + ".json"] = predictions

            # Save results
            with open(results_file, "w") as f:
                json.dump({split: results[split]}, f, indent=4)
            print(f"Results saved to {results_file}")

        print(f"Number of '{split}' tasks correctly solved: {correct_solved} out of {total_tasks}")

    # Save proposed transformations
    proposed_transformations_file = PathConfig.get_data_path("proposed_transformations.txt")
    with open(proposed_transformations_file, "w") as f:
        for split in proposed_transformations_dict:
            for task_id, transformations in proposed_transformations_dict[split].items():
                f.write(f"{split}/{task_id}: {transformations}\n")
    print(f"Proposed transformations saved to {proposed_transformations_file}")

    return results


def main():
    """Main entry point for evaluation script."""
    # CLI argument parser
    parser = argparse.ArgumentParser(description="Evaluate the Flax transformer model on ARC tasks")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=3,
        help="Number of parallel workers for task solving (default: 3)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to evaluate per split (default: None = all tasks)",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help="Enable test-time adaptation (default: False)",
    )
    parser.add_argument(
        "--tta-epochs",
        type=int,
        default=15,
        help="Number of epochs for TTA fine-tuning (default: 15)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: auto-detect latest in cache/checkpoints/)",
    )
    args = parser.parse_args()

    # Configure JAX
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    # Load tokenizer
    tokenizer = CustomTokenizer()
    tokenizer.load_vocab(PathConfig.get_data_path("vocab.json"))
    print(f"Vocabulary size: {len(tokenizer.vocab)}")

    # Initialize model
    model = FlaxCustomTransformer(vocab_size=len(tokenizer.vocab))

    # Create dummy state for loading checkpoint
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    params = model.init(rng, dummy_input, deterministic=True)

    # Create train state
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(5e-5))
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx, dropout_rng=rng)

    # Determine checkpoint path
    if args.checkpoint is None:
        # Auto-detect latest checkpoint
        checkpoint_base = PathConfig.get_checkpoint_dir("25.3M")
        if os.path.exists(checkpoint_base):
            checkpoints = [
                f for f in os.listdir(checkpoint_base) if f.endswith(".msgpack") and "final" in f
            ]
            if checkpoints:
                # Sort by epoch number and take the latest
                checkpoints.sort()
                args.checkpoint = os.path.join(checkpoint_base, checkpoints[-1])
                print(f"Auto-detected checkpoint: {args.checkpoint}")
            else:
                print(f"ERROR: No checkpoints found in {checkpoint_base}")
                print("Please train a model first or specify a valid checkpoint path.")
                return
        else:
            print(f"ERROR: Checkpoint directory not found at {checkpoint_base}")
            print("Please train a model first or specify a valid checkpoint path.")
            return

    # Load checkpoint
    print(f"##### LOADING THE MODEL FROM {args.checkpoint}... #####")
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        print("Please train a model first or specify a valid checkpoint path.")
        return

    with open(args.checkpoint, "rb") as f:
        bytes_input = f.read()

    # Load checkpoint dictionary
    checkpoint_dict = serialization.msgpack_restore(bytes_input)

    # Update state with loaded parameters
    state = state.replace(params=checkpoint_dict["model"])
    print("#####... MODEL LOADED #####")

    # Count parameters
    total_params = count_parameters(state.params["params"])
    print(f"Total parameters: {total_params / 1_000_000:.1f}M")

    # Run evaluation
    print(
        f"Running evaluation with {args.num_workers} workers, TTA={args.tta}, "
        f"max_tasks={args.max_tasks}"
    )
    evaluate_true(
        state,
        model,
        tokenizer,
        tta=args.tta,
        tta_epochs=args.tta_epochs,
        num_workers=args.num_workers,
        max_tasks=args.max_tasks,
    )


if __name__ == "__main__":
    main()
