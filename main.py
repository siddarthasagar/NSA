# main.py

from task import *
import sys
import json
from multiprocessing import Process, Queue


def solve_task_id(task_file, task_type="training", time_limit=1800, save_images=False, q=None):
    """
    Solves a given task and returns the solution.
    This function is intended to be run in a separate process.
    """
    try:
        # Determine the data path based on task_type
        if task_type == "training":
            data_path = "dataset/training/"
        else:
            data_path = "dataset/evaluation/"

        # Initialize the Task
        # task = Task(data_path + task_file, proposed_transformations = ["truncate"])
        # task = Task(data_path + task_file, proposed_transformations = ["upscale_grid"])
        task = Task(data_path + task_file)

        # Solve the task
        (
            abstraction,
            solution_apply_call,
            error,
            train_error,
            solving_time,
            nodes_explored,
        ) = task.solve(
            shared_frontier=True,
            time_limit=time_limit,
            do_constraint_acquisition=True,
            save_images=save_images,
        )

        # Prepare the solution dictionary
        solution = {
            "abstraction": abstraction,
            "apply_call": solution_apply_call,
            "train_error": train_error,
            "test_error": error,
            "time": solving_time,
            "nodes_explored": nodes_explored,
        }

        # Put the solution into the queue
        if q is not None:
            q.put(solution)
    except Exception as e:
        # In case of exception, put the exception into the queue
        if q is not None:
            q.put(e)


def main():
    if __name__ == "__main__":
        if len(sys.argv) < 3:
            print(
                "Usage: python -m main <task_file.json> <task_type> [time_limit_in_seconds] [save_images]"
            )
            sys.exit(1)

        task_file = str(sys.argv[1])
        task_type = str(sys.argv[2])

        time_limit = 1800

        # Optional: Specify whether to save images via command-line argument
        if len(sys.argv) >= 5:
            save_images = sys.argv[4].lower() in ["true", "1", "yes"]
        else:
            save_images = False

        # Create a Queue to receive the result
        q = Queue()

        # Create a Process to run solve_task_id
        p = Process(
            target=solve_task_id,
            args=(task_file, task_type, time_limit, save_images, q),
        )

        # Start the process
        p.start()

        # Wait for the process to finish within the time limit
        p.join(time_limit)

        if p.is_alive():
            print(f"Time limit of {time_limit} seconds exceeded. Terminating the process.")
            p.terminate()
            p.join()
            # Optionally, handle the timeout case as needed
            solution = {
                "abstraction": None,
                "apply_call": None,
                "train_error": None,
                "test_error": None,
                "time": time_limit,
                "nodes_explored": None,
                "status": "timeout",
            }
        else:
            try:
                result = q.get_nowait()
                if isinstance(result, Exception):
                    print(f"An error occurred in the worker process: {result}")
                    solution = {
                        "abstraction": None,
                        "apply_call": None,
                        "train_error": None,
                        "test_error": None,
                        "time": time_limit,
                        "nodes_explored": None,
                        "status": "error",
                        "error": str(result),
                    }
                else:
                    solution = result
                    solution["status"] = "success"
            except Exception as e:
                print("Failed to retrieve the solution from the process.")
                solution = {
                    "abstraction": None,
                    "apply_call": None,
                    "train_error": None,
                    "test_error": None,
                    "time": time_limit,
                    "nodes_explored": None,
                    "status": "error",
                    "error": str(e),
                }

        # Print the solution
        print(json.dumps(solution, indent=4))


if __name__ == "__main__":
    main()
