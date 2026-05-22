import os

def get_project_root() -> str:
    """
    Returns the absolute path to the project root directory.
    Assumes this file is located at <project_root>/utils/path_utils.py.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resolve_path(rel_or_abs: str) -> str:
    """
    Resolves a path to an absolute path.
    If the path is already absolute, it is returned as is.
    If it is relative, it is resolved relative to the project root directory.
    """
    if not rel_or_abs:
        return ""
    if os.path.isabs(rel_or_abs):
        return os.path.normpath(rel_or_abs)
    # Strip leading dots and separators to resolve relative to root
    cleaned = rel_or_abs.lstrip('.').lstrip('/').lstrip('\\')
    return os.path.normpath(os.path.join(get_project_root(), cleaned))
