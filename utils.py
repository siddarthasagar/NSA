import os
from enum import Enum
from pathlib import Path


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP_LEFT = "UP_LEFT"
    UP_RIGHT = "UP_RIGHT"
    DOWN_LEFT = "DOWN_LEFT"
    DOWN_RIGHT = "DOWN_RIGHT"


class Rotation(str, Enum):
    CW = "CW"
    CCW = "CCW"
    CW2 = "CW2"


class Mirror(str, Enum):
    VERTICAL = "VERTICAL"
    HORIZONTAL = "HORIZONTAL"
    DIAGONAL_LEFT = "DIAGONAL_LEFT"  # \
    DIAGONAL_RIGHT = "DIAGONAL_RIGHT"  # /


class ImagePoints(str, Enum):
    TOP = "TOP"
    BOTTOM = "BOTTOM"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    TOP_LEFT = "TOP_LEFT"
    TOP_RIGHT = "TOP_RIGHT"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"


class RelativePosition(str, Enum):
    SOURCE = "SOURCE"
    TARGET = "TARGET"
    MIDDLE = "MIDDLE"


class ObjectProperty(Enum):
    SYMMETRICAL = 0
    HOLLOW = 1


class PathConfig:
    """Centralized path configuration for cache directory."""

    # Base directories
    CACHE_ROOT = "cache"
    CHECKPOINTS_DIR = "cache/checkpoints"
    DATA_DIR = "cache/data"
    LOGS_DIR = "cache/logs"
    TTA_DIR = "cache/tta"

    @staticmethod
    def ensure_cache_dirs() -> None:
        """Create cache directory structure if it doesn't exist.

        Creates the following directories:
        - cache/checkpoints/
        - cache/data/
        - cache/logs/
        - cache/tta/

        Raises:
            OSError: If directory creation fails due to permissions or disk space.
        """
        directories = [
            PathConfig.CACHE_ROOT,
            PathConfig.CHECKPOINTS_DIR,
            PathConfig.DATA_DIR,
            PathConfig.LOGS_DIR,
            PathConfig.TTA_DIR,
        ]

        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise OSError(
                    f"Failed to create cache directory '{directory}': {e}. "
                    f"Check permissions and available disk space."
                ) from e

    @staticmethod
    def get_checkpoint_dir(model_size: str) -> str:
        """Get checkpoint directory path for a given model size.

        Args:
            model_size: Model size identifier (e.g., "1.5M", "3.0M")

        Returns:
            Path to checkpoint directory (e.g., "cache/checkpoints/small_transformer_based/results/1.5M")

        Raises:
            ValueError: If model_size is empty or invalid format.
        """
        if not model_size or not isinstance(model_size, str):
            raise ValueError(
                f"Invalid model_size: '{model_size}'. Expected non-empty string (e.g., '1.5M')."
            )

        checkpoint_path = os.path.join(
            PathConfig.CHECKPOINTS_DIR, "small_transformer_based", "results", model_size
        )

        # Ensure the directory exists
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        return checkpoint_path

    @staticmethod
    def get_checkpoint_path(
        model_size: str, epoch: int, final: bool = True, batch: int | None = None
    ) -> str:
        """Get full checkpoint file path.

        Args:
            model_size: Model size identifier (e.g., "1.5M", "3.0M")
            epoch: Epoch number
            final: If True, returns final checkpoint path; if False, returns emergency checkpoint
            batch: Batch number for emergency checkpoints (required if final=False)

        Returns:
            Full path to checkpoint file

        Raises:
            ValueError: If parameters are invalid or batch is missing for emergency checkpoint.
            FileNotFoundError: If checkpoint file doesn't exist at expected location.
        """
        if epoch < 0:
            raise ValueError(f"Invalid epoch: {epoch}. Expected non-negative integer.")

        checkpoint_dir = PathConfig.get_checkpoint_dir(model_size)

        if final:
            checkpoint_file = f"checkpoint_epoch{epoch}_final.msgpack"
        else:
            if batch is None:
                raise ValueError(
                    "batch parameter is required for emergency checkpoints (final=False)."
                )
            checkpoint_file = f"emergency_epoch{epoch}_batch{batch}.msgpack"

        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

        return checkpoint_path

    @staticmethod
    def get_data_path(filename: str) -> str:
        """Get data file path in cache/data/.

        Args:
            filename: Name of the data file (e.g., "vocab.json", "full_trans.json")

        Returns:
            Full path to data file in cache directory

        Raises:
            ValueError: If filename is empty or contains path separators.
        """
        if not filename or not isinstance(filename, str):
            raise ValueError(f"Invalid filename: '{filename}'. Expected non-empty string.")

        if os.path.sep in filename or "/" in filename or "\\" in filename:
            raise ValueError(
                f"Invalid filename: '{filename}'. Filename should not contain path separators."
            )

        # Ensure data directory exists
        Path(PathConfig.DATA_DIR).mkdir(parents=True, exist_ok=True)

        return os.path.join(PathConfig.DATA_DIR, filename)

    @staticmethod
    def get_log_path(filename: str) -> str:
        """Get log file path in cache/logs/.

        Args:
            filename: Name of the log file (e.g., "training_test.log")

        Returns:
            Full path to log file in cache directory

        Raises:
            ValueError: If filename is empty or contains path separators.
        """
        if not filename or not isinstance(filename, str):
            raise ValueError(f"Invalid filename: '{filename}'. Expected non-empty string.")

        if os.path.sep in filename or "/" in filename or "\\" in filename:
            raise ValueError(
                f"Invalid filename: '{filename}'. Filename should not contain path separators."
            )

        # Ensure logs directory exists
        Path(PathConfig.LOGS_DIR).mkdir(parents=True, exist_ok=True)

        return os.path.join(PathConfig.LOGS_DIR, filename)

    @staticmethod
    def get_tta_dir(task_id: str) -> str:
        """Get TTA (Test-Time Adaptation) directory path for a specific task.

        Args:
            task_id: Task identifier (e.g., "00576224", "009d5c81")

        Returns:
            Path to TTA directory for the task (e.g., "cache/tta/00576224")

        Raises:
            ValueError: If task_id is empty or contains invalid characters.
        """
        if not task_id or not isinstance(task_id, str):
            raise ValueError(f"Invalid task_id: '{task_id}'. Expected non-empty string.")

        if os.path.sep in task_id or "/" in task_id or "\\" in task_id:
            raise ValueError(
                f"Invalid task_id: '{task_id}'. Task ID should not contain path separators."
            )

        tta_path = os.path.join(PathConfig.TTA_DIR, task_id)

        # Ensure the directory exists
        Path(tta_path).mkdir(parents=True, exist_ok=True)

        return tta_path

    @staticmethod
    def get_tta_task_file(task_id: str) -> str:
        """Get TTA task JSON file path for a specific task.

        Args:
            task_id: Task identifier (e.g., "00576224", "009d5c81")

        Returns:
            Path to TTA task JSON file (e.g., "cache/tta/00576224.json")

        Raises:
            ValueError: If task_id is empty or contains invalid characters.
        """
        if not task_id or not isinstance(task_id, str):
            raise ValueError(f"Invalid task_id: '{task_id}'. Expected non-empty string.")

        if os.path.sep in task_id or "/" in task_id or "\\" in task_id:
            raise ValueError(
                f"Invalid task_id: '{task_id}'. Task ID should not contain path separators."
            )

        # Ensure TTA directory exists
        Path(PathConfig.TTA_DIR).mkdir(parents=True, exist_ok=True)

        return os.path.join(PathConfig.TTA_DIR, f"{task_id}.json")
