import argparse
import os
import shutil
import json
from multiprocessing import Pool, cpu_count
from auxilaries.grid_transformation import (
    sample_and_apply,
    prepare_and_save_transformed_data,
    append_transformation_to_file,
)
from tqdm import tqdm


def clear_and_create_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)


def initialize_json_file(json_path):
    with open(json_path, "w") as f:
        json.dump([], f)


def update_progress_bar(folder, pbar, target_count):
    current_count = len(os.listdir(folder))
    pbar.n = current_count
    pbar.last_print_n = current_count
    pbar.refresh()


def _generate_single_sample(args):
    """Worker function for parallel sample generation."""
    no_of_trans, transformation_ops, chosen_task, timeout, sample_idx = args
    try:
        # Call sample_and_apply directly (timeout handled by pool)
        result = sample_and_apply(
            no_of_trans=no_of_trans,
            transformation_ops=transformation_ops,
            samples="task_based",
            chosen_task=chosen_task,
        )
        if result is None:
            return None
        return result
    except Exception as e:
        if str(e) != "No change":
            pass  # Silently skip errors for cleaner output
        return None


def generate_samples(
    number_of_samples,
    output_folder,
    all_transformations_path,
    no_of_trans,
    transformation_ops=None,
    chosen_task=None,
    timeout: float = 2.0,
    num_workers=None,
):
    """
    Generate samples in parallel using multiple CPU cores.

    Args:
        num_workers: Number of parallel workers (default: CPU count - 1)
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    elif num_workers <= 0:
        num_workers = 1

    print(f"Using {num_workers} parallel workers for data generation")

    pbar = tqdm(total=number_of_samples, desc=f"Generating {no_of_trans} transformation samples")

    # Prepare arguments for each sample
    sample_args = [
        (no_of_trans, transformation_ops, chosen_task, timeout, i) for i in range(number_of_samples)
    ]

    successful_samples = 0

    # Use process pool for parallel generation
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            # Use imap_unordered with chunksize for better performance
            results_iter = pool.imap_unordered(_generate_single_sample, sample_args, chunksize=1)
            for result in results_iter:
                if result is not None:
                    try:
                        original_grids, transformed_grids, transformation_details = result
                        prepare_and_save_transformed_data(
                            original_grids,
                            transformed_grids,
                            transformation_details,
                            output_folder=output_folder,
                        )
                        append_transformation_to_file(
                            all_transformations_path,
                            original_grids,
                            transformed_grids,
                            transformation_details,
                        )
                        successful_samples += 1
                    except Exception:
                        pass
                pbar.update(1)
                if successful_samples >= number_of_samples:
                    break
    else:
        # Sequential fallback for single worker
        for args in sample_args:
            result = _generate_single_sample(args)
            if result is not None:
                try:
                    original_grids, transformed_grids, transformation_details = result
                    prepare_and_save_transformed_data(
                        original_grids,
                        transformed_grids,
                        transformation_details,
                        output_folder=output_folder,
                    )
                    append_transformation_to_file(
                        all_transformations_path,
                        original_grids,
                        transformed_grids,
                        transformation_details,
                    )
                    successful_samples += 1
                except Exception:
                    pass
            pbar.update(1)
            if successful_samples >= number_of_samples:
                break

    pbar.close()
    print(f"Successfully generated {successful_samples} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Generate transformation samples and save them to specified directories."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500000,
        help="Number of samples to generate for each transformation type.",
    )
    parser.add_argument(
        "--one_trans_folder",
        type=str,
        default="final_data8",
        help="Output folder for one transformation samples.",
    )
    parser.add_argument(
        "--two_trans_folder",
        type=str,
        default="generated_llm_data_two_trans1",
        help="Output folder for two transformation samples.",
    )
    parser.add_argument(
        "--all_transformations_path",
        type=str,
        default="full_trans.json",
        help="Path to the JSON file storing all transformations.",
    )
    parser.add_argument(
        "--transformations",
        type=str,
        choices=["one", "two", "both"],
        default="one",
        help='Specify which transformations to perform: "one", "two", or "both".',
    )
    parser.add_argument(
        "--transformation_op",
        type=str,
        nargs="*",
        help="Specify the transformation operators to be used in sample_and_apply.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Per-sample generation timeout in seconds (increase if you see many timeouts).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1, use 1 for sequential)",
    )

    args = parser.parse_args()

    # Clear existing directories and create new ones based on the selected transformation
    if args.transformations in ["one", "both"]:
        print(f"DELETING: {args.one_trans_folder}")
        clear_and_create_folder(args.one_trans_folder)
    if args.transformations in ["two", "both"]:
        print(f"DELETING: {args.two_trans_folder}")
        clear_and_create_folder(args.two_trans_folder)

    # Initialize or clear the all_transformations.json file
    initialize_json_file(args.all_transformations_path)

    # Parse the transformation operators
    transformation_ops = args.transformation_op if args.transformation_op else None

    # Run generation according to selected mode
    if args.transformations in ["one", "both"]:
        # One-step transformation chains
        generate_samples(
            args.samples,
            args.one_trans_folder,
            args.all_transformations_path,
            no_of_trans=1,
            transformation_ops=transformation_ops,
            timeout=args.timeout,
            num_workers=args.workers,
        )

    if args.transformations in ["two", "both"]:
        # Two-step transformation chains
        generate_samples(
            args.samples,
            args.two_trans_folder,
            args.all_transformations_path,
            no_of_trans=2,
            transformation_ops=transformation_ops,
            timeout=args.timeout,
            num_workers=args.workers,
        )


if __name__ == "__main__":
    main()
