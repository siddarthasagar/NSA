# Requirements Document

## Introduction

This specification addresses code cleanup and organization for the NSA (Neuro-Symbolic ARC) project after completing the PyTorch to JAX/Flax migration. The project needs to consolidate output files into a centralized cache directory and identify dead code through pytest coverage analysis.

## Glossary

- **NSA System**: The Neuro-Symbolic ARC solver that combines transformer models with symbolic search
- **Cache Directory**: A centralized folder (`cache/`) at the project root for all generated artifacts
- **Output Artifacts**: Generated files including checkpoints, JSON data files, logs, and temporary files
- **Dead Code**: Unused or residual code from the PyTorch implementation that is no longer needed
- **Coverage Report**: pytest-cov analysis showing which code paths are executed during tests
- **Flax Pipeline**: The JAX/Flax-based training and evaluation system that replaced PyTorch
- **Quick Tasks**: Minimal versions of data generation, training, and evaluation for testing
- **Integration Tests**: Tests that exercise complete workflows end-to-end to verify system functionality
- **Tests Directory**: A `tests/` folder at the project root containing all pytest test files

## Requirements

### Requirement 1: Centralized Cache Directory Structure

**User Story:** As a developer, I want all generated output files organized in a single cache directory, so that I can easily review outputs and perform cleanup operations.

#### Acceptance Criteria

1. WHEN THE NSA System initializes, THE NSA System SHALL create a `cache/` directory at the project root IF the directory does not exist
2. THE NSA System SHALL organize output artifacts within `cache/` using subdirectories that maintain the current hierarchy
3. THE NSA System SHALL redirect checkpoint files to `cache/checkpoints/` preserving the existing folder structure
4. THE NSA System SHALL redirect JSON output files to `cache/data/` preserving filenames
5. THE NSA System SHALL redirect log files to `cache/logs/` with descriptive names

### Requirement 2: File Migration and Code Updates

**User Story:** As a developer, I want existing output files moved to the cache directory and all code updated to use the new paths, so that the system continues to function correctly with the new organization.

#### Acceptance Criteria

1. THE NSA System SHALL move existing checkpoint files from `small_transformer_based/results/` to `cache/checkpoints/small_transformer_based/results/`
2. THE NSA System SHALL move existing JSON files (`full_trans.json`, `vocab.json`, `dataset_cache.txt`, `training_data_summary.json`) to `cache/data/`
3. THE NSA System SHALL move existing log files (`training_test.log`) to `cache/logs/`
4. THE NSA System SHALL update all file path references in Python code to use the new cache directory structure
5. THE NSA System SHALL update Makefile targets to reference the new cache directory paths

### Requirement 3: Pytest Test Suite for Core Workflows

**User Story:** As a developer, I want pytest integration tests that exercise the core data generation, training, and evaluation workflows, so that I can verify functionality and identify dead code through coverage analysis.

#### Acceptance Criteria

1. THE NSA System SHALL organize all test files within a `tests/` directory at the project root
2. THE NSA System SHALL provide an integration test that generates a minimal dataset using the quick-generate workflow
3. THE NSA System SHALL provide an integration test that trains a model for minimal epochs using the quick-train workflow
4. THE NSA System SHALL provide an integration test that evaluates a trained model using the quick-eval workflow
5. THE NSA System SHALL configure pytest-cov to generate coverage reports in HTML and terminal formats
6. THE NSA System SHALL exclude test files, migrations, and virtual environments from coverage analysis
7. THE NSA System SHALL use the existing pytest-cov dependency without requiring additional package installations

### Requirement 4: Dead Code Identification

**User Story:** As a developer, I want a coverage report that identifies untested code paths, so that I can safely remove PyTorch residue and unused code.

#### Acceptance Criteria

1. THE NSA System SHALL generate a coverage report showing percentage coverage for each Python module
2. THE NSA System SHALL identify Python files with zero coverage as candidates for removal
3. THE NSA System SHALL highlight functions and classes that are never executed during tests
4. THE NSA System SHALL provide a summary report listing potential dead code files
5. THE NSA System SHALL exclude intentionally unused code (examples, deprecated modules) from dead code analysis

### Requirement 5: Makefile Integration

**User Story:** As a developer, I want Makefile targets for running tests and generating coverage reports, so that I can easily execute the test suite and review results.

#### Acceptance Criteria

1. THE NSA System SHALL provide a `make test` target that runs the full pytest suite
2. THE NSA System SHALL provide a `make test-quick` target that runs only fast integration tests
3. THE NSA System SHALL provide a `make coverage` target that generates and displays coverage reports
4. THE NSA System SHALL provide a `make coverage-html` target that opens the HTML coverage report in a browser
5. THE NSA System SHALL update the `make clean` target to remove cache directory contents

### Requirement 6: Documentation Updates

**User Story:** As a developer, I want updated documentation that explains the new cache directory structure and testing workflow, so that team members understand the changes.

#### Acceptance Criteria

1. THE NSA System SHALL update the README with a section describing the cache directory structure
2. THE NSA System SHALL document the purpose of each cache subdirectory
3. THE NSA System SHALL provide examples of running tests and generating coverage reports
4. THE NSA System SHALL document the process for identifying and removing dead code
5. THE NSA System SHALL update steering files to reflect the new organization patterns
