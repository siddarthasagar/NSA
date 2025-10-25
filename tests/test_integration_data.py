"""Integration tests for data generation workflow.

This module tests the end-to-end data generation pipeline, verifying that
synthetic training data is correctly generated and saved to the cache directory.
"""

import json
import os
from pathlib import Path


from auxilaries.generate_transformation import (
    generate_samples,
    initialize_json_file,
)
from utils import PathConfig


class TestDataGenerationMinimal:
    """Test minimal data generation workflow."""

    def test_data_generation_minimal(self, test_cache_dir: Path, tmp_path: Path) -> None:
        """Test minimal data generation workflow with 10 samples.

        This test verifies that:
        1. Data generation completes without errors
        2. Output files are created in correct cache location
        3. JSON structure contains required fields
        4. Generated data is valid

        Args:
            test_cache_dir: Temporary cache directory fixture
            tmp_path: Temporary path for output folder
        """
        # Setup
        num_samples = 10
        output_folder = str(tmp_path / "test_data_output")
        all_transformations_path = PathConfig.get_data_path("test_full_trans.json")

        # Ensure cache directories exist
        PathConfig.ensure_cache_dirs()

        # Initialize the transformations JSON file
        initialize_json_file(all_transformations_path)

        # Generate samples with minimal configuration
        # Use num_workers=1 for deterministic testing
        generate_samples(
            number_of_samples=num_samples,
            output_folder=output_folder,
            all_transformations_path=all_transformations_path,
            no_of_trans=1,  # Single transformation for speed
            transformation_ops=None,  # Random transformations
            chosen_task=None,  # Random task selection
            timeout=2.0,
            num_workers=1,  # Sequential for test stability
        )

        # Verify output folder was created
        assert os.path.exists(output_folder), "Output folder was not created"

        # Verify individual sample folders were created
        sample_folders = [
            f for f in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, f))
        ]
        assert len(sample_folders) > 0, "No sample folders were created"
        assert len(sample_folders) <= num_samples, (
            f"Too many samples created: {len(sample_folders)}"
        )

        # Verify each sample folder contains required files
        for folder_name in sample_folders:
            folder_path = os.path.join(output_folder, folder_name)

            # Check for transformed_data.json
            json_path = os.path.join(folder_path, "transformed_data.json")
            assert os.path.exists(json_path), f"Missing transformed_data.json in {folder_name}"

            # Verify JSON structure
            with open(json_path) as f:
                data = json.load(f)

            # Verify required fields
            assert "instruction" in data, f"Missing 'instruction' field in {folder_name}"
            assert "input" in data, f"Missing 'input' field in {folder_name}"
            assert "output" in data, f"Missing 'output' field in {folder_name}"

            # Verify instruction is not empty
            assert isinstance(data["instruction"], str), "Instruction must be a string"
            assert len(data["instruction"]) > 0, "Instruction cannot be empty"

            # Verify input format (should be formatted grid text)
            assert isinstance(data["input"], str), "Input must be a string"
            assert len(data["input"]) > 0, "Input cannot be empty"
            assert "Input:" in data["input"], "Input should contain 'Input:' marker"
            assert "Output:" in data["input"], "Input should contain 'Output:' marker"

            # Verify output is a transformation name or list
            assert isinstance(data["output"], (str, list)), "Output must be string or list"
            if isinstance(data["output"], str):
                assert len(data["output"]) > 0, "Output cannot be empty"

            # Check for transformation info file
            info_path = os.path.join(folder_path, "transformation_info.txt")
            assert os.path.exists(info_path), f"Missing transformation_info.txt in {folder_name}"

            # Check for image files (at least original and transformed)
            image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
            assert len(image_files) >= 2, f"Missing image files in {folder_name}"

        # Verify all_transformations.json was created and populated
        assert os.path.exists(all_transformations_path), "all_transformations.json was not created"

        with open(all_transformations_path) as f:
            all_transformations = json.load(f)

        # Verify it's a list
        assert isinstance(all_transformations, list), "all_transformations must be a list"

        # Verify it contains entries
        assert len(all_transformations) > 0, "all_transformations is empty"
        assert len(all_transformations) <= num_samples, (
            f"Too many transformations: {len(all_transformations)}"
        )

        # Verify each transformation entry has required fields
        for idx, transformation in enumerate(all_transformations):
            assert "instruction" in transformation, f"Missing 'instruction' in entry {idx}"
            assert "input" in transformation, f"Missing 'input' in entry {idx}"
            assert "output" in transformation, f"Missing 'output' in entry {idx}"

            # Verify output is a valid transformation name
            output = transformation["output"]
            assert isinstance(output, str), f"Output must be string in entry {idx}"
            assert len(output) > 0, f"Output cannot be empty in entry {idx}"

    def test_data_generation_with_specific_transformation(
        self, test_cache_dir: Path, tmp_path: Path
    ) -> None:
        """Test data generation with a specific transformation operation.

        This test verifies that data generation works when specifying
        a particular transformation operation.

        Args:
            test_cache_dir: Temporary cache directory fixture
            tmp_path: Temporary path for output folder
        """
        # Setup
        num_samples = 5
        output_folder = str(tmp_path / "test_specific_trans")
        all_transformations_path = PathConfig.get_data_path("test_specific_trans.json")

        # Ensure cache directories exist
        PathConfig.ensure_cache_dirs()

        # Initialize the transformations JSON file
        initialize_json_file(all_transformations_path)

        # Generate samples with specific transformation
        generate_samples(
            number_of_samples=num_samples,
            output_folder=output_folder,
            all_transformations_path=all_transformations_path,
            no_of_trans=1,
            transformation_ops=["rotate_grid"],  # Specific transformation
            chosen_task=None,
            timeout=2.0,
            num_workers=1,
        )

        # Verify output was created
        assert os.path.exists(output_folder), "Output folder was not created"

        # Verify samples were generated
        sample_folders = [
            f for f in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, f))
        ]
        assert len(sample_folders) > 0, "No samples were generated"

        # Verify transformation file exists
        assert os.path.exists(all_transformations_path), "Transformations file not created"

    def test_data_generation_cache_location(self, test_cache_dir: Path, tmp_path: Path) -> None:
        """Test that data generation uses correct cache directory paths.

        This test verifies that PathConfig correctly directs output
        to the cache directory structure.

        Args:
            test_cache_dir: Temporary cache directory fixture
            tmp_path: Temporary path for output folder
        """
        # Verify cache directories exist
        PathConfig.ensure_cache_dirs()

        # Check that cache subdirectories were created
        assert os.path.exists(PathConfig.DATA_DIR), "cache/data directory not created"
        assert os.path.exists(PathConfig.CHECKPOINTS_DIR), "cache/checkpoints directory not created"
        assert os.path.exists(PathConfig.LOGS_DIR), "cache/logs directory not created"
        assert os.path.exists(PathConfig.TTA_DIR), "cache/tta directory not created"

        # Test PathConfig.get_data_path
        test_filename = "test_file.json"
        data_path = PathConfig.get_data_path(test_filename)

        # Verify path is in cache/data
        assert PathConfig.DATA_DIR in data_path, "Data path not in cache/data directory"
        assert test_filename in data_path, "Filename not in data path"

        # Verify path is absolute or relative to project root
        assert not data_path.startswith("/tmp"), "Should not use system temp directory"

    def test_data_generation_json_validity(self, test_cache_dir: Path, tmp_path: Path) -> None:
        """Test that generated JSON files are valid and well-formed.

        This test performs deeper validation of the JSON structure
        and content to ensure data quality.

        Args:
            test_cache_dir: Temporary cache directory fixture
            tmp_path: Temporary path for output folder
        """
        # Setup
        num_samples = 3
        output_folder = str(tmp_path / "test_json_validity")
        all_transformations_path = PathConfig.get_data_path("test_json_validity.json")

        # Ensure cache directories exist
        PathConfig.ensure_cache_dirs()

        # Initialize the transformations JSON file
        initialize_json_file(all_transformations_path)

        # Generate samples
        generate_samples(
            number_of_samples=num_samples,
            output_folder=output_folder,
            all_transformations_path=all_transformations_path,
            no_of_trans=1,
            transformation_ops=None,
            chosen_task=None,
            timeout=2.0,
            num_workers=1,
        )

        # Get sample folders
        sample_folders = [
            f for f in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, f))
        ]

        # Verify at least one sample was generated
        assert len(sample_folders) > 0, "No samples generated"

        # Check first sample in detail
        first_sample = sample_folders[0]
        json_path = os.path.join(output_folder, first_sample, "transformed_data.json")

        with open(json_path) as f:
            data = json.load(f)

        # Verify input contains grid data
        input_text = data["input"]
        assert "|" in input_text, "Input should contain pipe-separated grid values"

        # Verify input has multiple lines (grid rows)
        input_lines = input_text.strip().split("\n")
        assert len(input_lines) > 1, "Input should have multiple lines"

        # Verify output is a known transformation
        output = data["output"]
        if isinstance(output, str):
            # Should be a transformation name or space-separated list
            transformations = output.split()
            assert len(transformations) > 0, "Output should contain at least one transformation"

        # Verify instruction is the expected system prompt
        assert "ARC" in data["instruction"], "Instruction should mention ARC tasks"
