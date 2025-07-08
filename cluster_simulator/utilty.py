import os

def find_project_dir(start_path=None):
    """Finds the project directory by locating the first parent that contains .gitignore."""
    if start_path is None:
        start_path = os.getcwd()  # Default to current working directory

    current_dir = os.path.abspath(start_path)

    while current_dir != os.path.dirname(current_dir):  # Stop at the root "/"
        if os.path.exists(os.path.join(current_dir, ".gitignore")):
            return current_dir
        current_dir = os.path.dirname(current_dir)  # Move up one level

    return None  # Return None if .gitignore is not found