#!/usr/bin/env python3
"""One-time migration script to move legacy files to cache directory structure.

This script discovers files in legacy locations and moves them to the new
centralized cache directory structure. It includes dry-run mode for testing
and generates a detailed migration report.

Usage:
    python scripts/migrate_to_cache.py              # Perform migration
    python scripts/migrate_to_cache.py --dry-run    # Test without moving files
    python scripts/migrate_to_cache.py --report     # Show report only
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


class FileMigrator:
    """Handles migration of files from legacy locations to cache directory."""

    # Define legacy file mappings: source -> destination
    LEGACY_MAPPINGS = {
        # Data files at root
        "full_trans.json": "cache/data/full_trans.json",
        "vocab.json": "cache/data/vocab.json",
        "dataset_cache.txt": "cache/data/dataset_cache.txt",
        "training_data_summary.json": "cache/data/training_data_summary.json",
        # Log files at root
        "training_test.log": "cache/logs/training_test.log",
        # Checkpoint directories
        "small_transformer_based/results/": "cache/checkpoints/small_transformer_based/results/",
    }

    def __init__(self, dry_run: bool = False) -> None:
        """Initialize the file migrator.

        Args:
            dry_run: If True, simulate migration without moving files
        """
        self.dry_run = dry_run
        self.migration_results: list[dict[str, Any]] = []
        self.errors: list[str] = []

    def discover_files(self) -> list[tuple[str, str]]:
        """Discover files that need to be migrated.

        Returns:
            List of (source_path, destination_path) tuples for files that exist
        """
        files_to_migrate: list[tuple[str, str]] = []

        for source, destination in self.LEGACY_MAPPINGS.items():
            source_path = Path(source)

            # Handle directory migrations
            if source.endswith("/"):
                if source_path.exists() and source_path.is_dir():
                    # Find all files recursively in the directory
                    for file_path in source_path.rglob("*"):
                        if file_path.is_file():
                            # Calculate relative path within source directory
                            relative_path = file_path.relative_to(source_path)
                            dest_path = Path(destination) / relative_path
                            files_to_migrate.append((str(file_path), str(dest_path)))
            else:
                # Handle single file migrations
                if source_path.exists() and source_path.is_file():
                    files_to_migrate.append((source, destination))

        return files_to_migrate

    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes.

        Args:
            file_path: Path to the file

        Returns:
            File size in bytes, or 0 if file doesn't exist
        """
        try:
            return os.path.getsize(file_path)
        except OSError:
            return 0

    def move_file(self, source: str, destination: str) -> bool:
        """Move a file from source to destination with error handling.

        Args:
            source: Source file path
            destination: Destination file path

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create destination directory if it doesn't exist
            dest_path = Path(destination)
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            if self.dry_run:
                # In dry-run mode, just check if we can read the source
                with open(source, "rb"):
                    pass
                print(f"[DRY RUN] Would move: {source} -> {destination}")
                return True
            else:
                # Actually move the file
                shutil.move(source, destination)
                print(f"Moved: {source} -> {destination}")
                return True

        except PermissionError as e:
            error_msg = f"Permission denied: {source} - {e}"
            self.errors.append(error_msg)
            print(f"ERROR: {error_msg}")
            return False

        except OSError as e:
            error_msg = f"OS error moving {source}: {e}"
            self.errors.append(error_msg)
            print(f"ERROR: {error_msg}")
            return False

        except Exception as e:
            error_msg = f"Unexpected error moving {source}: {e}"
            self.errors.append(error_msg)
            print(f"ERROR: {error_msg}")
            return False

    def check_disk_space(self, required_bytes: int) -> bool:
        """Check if sufficient disk space is available.

        Args:
            required_bytes: Required space in bytes

        Returns:
            True if sufficient space available, False otherwise
        """
        try:
            stat = shutil.disk_usage(".")
            available_bytes = stat.free
            # Add 10% buffer for safety
            required_with_buffer = required_bytes * 1.1

            if available_bytes < required_with_buffer:
                error_msg = (
                    f"Insufficient disk space. Required: {required_with_buffer / 1024 / 1024:.2f} MB, "
                    f"Available: {available_bytes / 1024 / 1024:.2f} MB"
                )
                self.errors.append(error_msg)
                print(f"ERROR: {error_msg}")
                return False

            return True

        except Exception as e:
            error_msg = f"Error checking disk space: {e}"
            self.errors.append(error_msg)
            print(f"WARNING: {error_msg}")
            # Continue anyway if we can't check
            return True

    def migrate(self) -> dict[str, Any]:
        """Perform the migration of all discovered files.

        Returns:
            Migration report dictionary
        """
        print("=" * 70)
        print("File Migration to Cache Directory")
        print("=" * 70)
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE MIGRATION'}")
        print()

        # Discover files to migrate
        print("Discovering files to migrate...")
        files_to_migrate = self.discover_files()

        if not files_to_migrate:
            print("No files found to migrate.")
            return self._generate_report(files_to_migrate)

        print(f"Found {len(files_to_migrate)} file(s) to migrate.")
        print()

        # Calculate total size
        total_size = sum(self.get_file_size(source) for source, _ in files_to_migrate)
        print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
        print()

        # Check disk space (only for live migration)
        if not self.dry_run:
            if not self.check_disk_space(total_size):
                print("Migration aborted due to insufficient disk space.")
                return self._generate_report(files_to_migrate)

        # Perform migration
        print("Starting migration...")
        print("-" * 70)

        for source, destination in files_to_migrate:
            size_bytes = self.get_file_size(source)
            success = self.move_file(source, destination)

            self.migration_results.append(
                {
                    "source": source,
                    "destination": destination,
                    "size_bytes": size_bytes,
                    "success": success,
                }
            )

        print("-" * 70)
        print()

        # Generate and display report
        report = self._generate_report(files_to_migrate)
        self._display_summary(report)

        # Save report to file
        if not self.dry_run:
            report_path = "cache/data/migration_report.json"
            Path("cache/data").mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nMigration report saved to: {report_path}")

        return report

    def _generate_report(self, files_to_migrate: list[tuple[str, str]]) -> dict[str, Any]:
        """Generate migration report.

        Args:
            files_to_migrate: List of files that were attempted to migrate

        Returns:
            Report dictionary with migration details
        """
        successful = [r for r in self.migration_results if r["success"]]
        failed = [r for r in self.migration_results if not r["success"]]
        total_size_mb = sum(r["size_bytes"] for r in successful) / 1024 / 1024

        return {
            "timestamp": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "files_moved": self.migration_results,
            "errors": self.errors,
            "summary": {
                "total_files": len(files_to_migrate),
                "successful": len(successful),
                "failed": len(failed),
                "total_size_mb": round(total_size_mb, 2),
            },
        }

    def _display_summary(self, report: dict[str, Any]) -> None:
        """Display migration summary.

        Args:
            report: Migration report dictionary
        """
        summary = report["summary"]

        print("Migration Summary")
        print("=" * 70)
        print(f"Total files:     {summary['total_files']}")
        print(f"Successful:      {summary['successful']}")
        print(f"Failed:          {summary['failed']}")
        print(f"Total size:      {summary['total_size_mb']:.2f} MB")

        if report["errors"]:
            print()
            print("Errors encountered:")
            for error in report["errors"]:
                print(f"  - {error}")

        print("=" * 70)


def main() -> None:
    """Main entry point for the migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate legacy files to cache directory structure"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate migration without moving files",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Show existing migration report",
    )

    args = parser.parse_args()

    if args.report:
        # Display existing report
        report_path = Path("cache/data/migration_report.json")
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            print(json.dumps(report, indent=2))
        else:
            print("No migration report found at cache/data/migration_report.json")
        return

    # Perform migration
    migrator = FileMigrator(dry_run=args.dry_run)
    migrator.migrate()

    if args.dry_run:
        print()
        print("This was a dry run. No files were actually moved.")
        print("Run without --dry-run to perform the actual migration.")


if __name__ == "__main__":
    main()
