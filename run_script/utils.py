import os
import inspect

def prepare_results_directory():
    """
    Prepares a results directory that mirrors the two parent folders of the caller script.
    Creates a numbered subdirectory for the current test run.
    """
    # Get caller file path (not this file)
    caller_frame = inspect.stack()[1]
    caller_file = os.path.abspath(caller_frame.filename)

    file_dir = os.path.dirname(caller_file)
    parent_name = os.path.basename(file_dir)
    grandparent_name = os.path.basename(os.path.dirname(file_dir))

    project_root = os.path.abspath(os.path.join(file_dir, '..', '..', '..'))
    results_parent = os.path.join(project_root, 'results', grandparent_name, parent_name)
    os.makedirs(results_parent, exist_ok=True)

    existing = [
        d for d in os.listdir(results_parent)
        if os.path.isdir(os.path.join(results_parent, d)) and d.isdigit()
    ]
    next_idx = max(map(int, existing)) + 1 if existing else 1

    TEST_DIR = os.path.join(results_parent, f"{next_idx:03d}")
    os.makedirs(TEST_DIR, exist_ok=True)

    return TEST_DIR