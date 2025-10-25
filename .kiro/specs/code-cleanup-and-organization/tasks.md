# Implementation Plan

- [x] 1. Create PathConfig utility class
  - Create PathConfig class in utils.py with centralized path management
  - Implement cache directory creation methods
  - Implement simple path resolution methods (no fallback logic)
  - Add clear error messages for missing files
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Create file migration script
  - Create scripts/migrate_to_cache.py for one-time file migration
  - Implement file discovery for legacy locations
  - Implement safe file moving with error handling
  - Generate migration report JSON
  - Add dry-run mode for testing
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 3. Update training module for cache paths
  - Update flax_train.py to import PathConfig
  - Replace hardcoded checkpoint paths with PathConfig.get_checkpoint_path()
  - Replace hardcoded data paths (vocab.json, dataset_cache.txt) with PathConfig.get_data_path()
  - Update save_checkpoint function to use cache directory
  - Verify training still works with new paths
  - _Requirements: 2.4, 1.3, 1.4_

- [x] 4. Update evaluation module for cache paths
  - Update flax_eval.py to import PathConfig
  - Replace checkpoint loading paths with PathConfig methods
  - Update results file paths (arga_*.json) to use cache/data/
  - Update TTA directory to use cache/tta/
  - Update proposed_transformations.txt path
  - _Requirements: 2.4, 1.3, 1.4_

- [x] 5. Update data generation module for cache paths
  - Update generate_transformation.py to use PathConfig
  - Update full_trans.json output path to cache/data/
  - Ensure generated data folders remain at root (final_data8, etc.) for now
  - _Requirements: 2.4, 1.4_

- [x] 6. Create test directory structure and configuration
  - Create tests/ directory at project root
  - Create tests/__init__.py
  - Create tests/conftest.py with shared fixtures
  - Create .coveragerc configuration file
  - Add pytest configuration to pyproject.toml
  - _Requirements: 3.1, 3.5, 3.6, 3.7_

- [x] 7. Implement data generation integration test
  - Create tests/test_integration_data.py
  - Implement test_data_generation_minimal() that generates 10 samples
  - Verify output files are created in correct cache location
  - Verify JSON structure and required fields
  - Add assertions for data validity
  - _Requirements: 3.2_

- [x] 8. Implement training integration test
  - Create tests/test_integration_train.py
  - Implement test_training_minimal() that trains for 1 epoch
  - Use minimal dataset (10 samples, batch_size=2)
  - Verify checkpoint is saved to cache/checkpoints/
  - Verify vocab.json is created in cache/data/
  - Add memory usage checks
  - _Requirements: 3.3_

- [x] 9. Implement evaluation integration test
  - Create tests/test_integration_eval.py
  - Implement test_evaluation_minimal() that evaluates 2 tasks
  - Use pre-generated checkpoint or train minimal model
  - Verify results file is created in cache/data/
  - Verify predictions are generated
  - _Requirements: 3.4_

- [x] 10. Update Makefile with test targets
  - Add 'make test' target for full test suite with coverage
  - Add 'make test-quick' target for fast integration tests
  - Add 'make coverage' target for coverage report generation
  - Add 'make coverage-html' target to open HTML report
  - Update 'make clean' to remove cache/data/ and test artifacts
  - Update 'make clean-all' to remove entire cache/ directory
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 11. Run file migration and verify functionality
  - Execute migration script to move existing files
  - Verify all files moved successfully
  - Run quick-train to verify training works with new paths
  - Run quick-eval to verify evaluation works with new paths
  - Verify no errors with new cache structure
  - _Requirements: 2.1, 2.2, 2.3, 2.5_

- [x] 12. Clean up legacy directories and update data generation defaults
  - Remove empty small_transformer_based/results/ directory structure
  - Run 'make clean' to remove temporary data generation folders (final_data8/, generated_llm_data_two_trans1/)
  - Update generate_transformation.py default output folders to use cache/generated_samples/
  - Change --one_trans_folder default from "final_data8" to "cache/generated_samples/one_trans"
  - Change --two_trans_folder default from "generated_llm_data_two_trans1" to "cache/generated_samples/two_trans"
  - Update Makefile clean target to remove cache/generated_samples/ instead of root-level folders
  - Verify .gitignore properly excludes cache/ directory (already present)
  - _Requirements: 2.1, 2.2, 2.4, 2.5_

- [x] 13. Generate coverage report and identify dead code
  - Run 'make coverage' to generate full coverage report
  - Review HTML coverage report in browser
  - Identify files with 0% coverage as dead code candidates
  - Check for PyTorch imports in uncovered files (small_transformer_based/train.py, small_transformer_based/eval.py)
  - Create dead_code_candidates.md document with findings and recommendations
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 14. Update README.md documentation
  - Add "Cache Directory Structure" section after Abstract
  - Document purpose of each cache subdirectory (checkpoints/, data/, logs/, tta/)
  - Add "Testing" section with examples of running tests and coverage
  - Update code examples to reference cache paths (cache/data/full_trans.json)
  - Add "Development Workflow" section with make commands
  - _Requirements: 6.1, 6.2, 6.3_

- [x] 15. Update steering documentation
  - Update .kiro/steering/guide.md with testing guidelines
  - Add section on running tests and interpreting coverage reports
  - Document dead code identification process
  - Update .kiro/steering/structure.md with cache directory structure
  - Document PathConfig usage patterns
  - _Requirements: 6.4, 6.5_

- [x] 16. Final verification and cleanup
  - Run 'make format' to ensure code style compliance
  - Run 'make lint' to check for any remaining issues
  - Run full test suite with 'make test' and verify all tests pass
  - Review coverage report for any critical gaps in core modules
  - Verify all cache paths are using PathConfig correctly
  - Run 'make train-quick' to verify training still works
  - Run 'make eval-quick' to verify evaluation still works
  - _Requirements: All_
