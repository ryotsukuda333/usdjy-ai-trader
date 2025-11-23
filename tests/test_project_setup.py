"""Tests for project structure setup."""

import os
from pathlib import Path


def test_required_directories_exist():
    """
    Given: Project root directory
    When: Project structure is initialized
    Then: All required directories should exist
    """
    project_root = Path(__file__).parent.parent
    required_dirs = ['data', 'features', 'model', 'backtest', 'trader', 'utils', 'tests']

    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"Directory '{dir_name}' does not exist"
        assert dir_path.is_dir(), f"'{dir_name}' is not a directory"


def test_gitignore_exists():
    """
    Given: Project root directory
    When: Project structure is initialized
    Then: .gitignore file should exist
    """
    project_root = Path(__file__).parent.parent
    gitignore_path = project_root / '.gitignore'
    assert gitignore_path.exists(), ".gitignore file does not exist"
    assert gitignore_path.is_file(), ".gitignore is not a file"


def test_gitignore_contains_required_patterns():
    """
    Given: .gitignore file exists
    When: File is read
    Then: It should contain required exclusion patterns
    """
    project_root = Path(__file__).parent.parent
    gitignore_path = project_root / '.gitignore'

    required_patterns = [
        '__pycache__/',
        '*.pyc',
        '*.json',
        '*.csv',
        'venv/'
    ]

    with open(gitignore_path, 'r') as f:
        gitignore_content = f.read()

    for pattern in required_patterns:
        assert pattern in gitignore_content, f"Pattern '{pattern}' not found in .gitignore"


def test_requirements_txt_exists():
    """
    Given: Project root directory
    When: Project structure is initialized
    Then: requirements.txt should exist
    """
    project_root = Path(__file__).parent.parent
    req_path = project_root / 'requirements.txt'
    assert req_path.exists(), "requirements.txt does not exist"
    assert req_path.is_file(), "requirements.txt is not a file"


def test_requirements_txt_contains_dependencies():
    """
    Given: requirements.txt file exists
    When: File is read
    Then: It should contain all required dependencies with versions
    """
    project_root = Path(__file__).parent.parent
    req_path = project_root / 'requirements.txt'

    required_packages = ['pandas', 'numpy', 'xgboost', 'scikit-learn', 'yfinance', 'ta', 'matplotlib']

    with open(req_path, 'r') as f:
        requirements_content = f.read()

    for package in required_packages:
        assert package in requirements_content, f"Package '{package}' not found in requirements.txt"

    # Verify version specifications exist (format: package==version)
    lines = [line.strip() for line in requirements_content.split('\n') if line.strip() and not line.startswith('#')]
    for line in lines:
        if any(pkg in line for pkg in required_packages):
            assert '==' in line, f"Version not specified for: {line}"


def test_utils_errors_module_exists():
    """
    Given: utils directory exists
    When: Project structure is initialized
    Then: utils/errors.py should exist
    """
    project_root = Path(__file__).parent.parent
    errors_path = project_root / 'utils' / 'errors.py'
    assert errors_path.exists(), "utils/errors.py does not exist"
    assert errors_path.is_file(), "utils/errors.py is not a file"


def test_utils_errors_contains_base_exception_class():
    """
    Given: utils/errors.py exists
    When: Module is imported
    Then: It should define TraderError base exception class
    """
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    try:
        from utils.errors import TraderError

        # Verify TraderError is an Exception subclass
        assert issubclass(TraderError, Exception), "TraderError is not an Exception subclass"

        # Verify __init__ accepts required parameters
        import inspect
        sig = inspect.signature(TraderError.__init__)
        params = list(sig.parameters.keys())

        assert 'error_code' in params, "TraderError.__init__ missing 'error_code' parameter"
        assert 'user_message' in params, "TraderError.__init__ missing 'user_message' parameter"
        assert 'technical_message' in params, "TraderError.__init__ missing 'technical_message' parameter"
    finally:
        sys.path.pop(0)


def test_utils_errors_contains_error_subclasses():
    """
    Given: utils/errors.py exists
    When: Module is imported
    Then: It should define all required error subclasses
    """
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    try:
        from utils.errors import (
            TraderError,
            DataError,
            FeatureEngineeringError,
            ModelError,
            BacktestError,
            VisualizationError
        )

        # Verify each is a TraderError subclass
        error_classes = [
            DataError,
            FeatureEngineeringError,
            ModelError,
            BacktestError,
            VisualizationError
        ]

        for error_class in error_classes:
            assert issubclass(error_class, TraderError), \
                f"{error_class.__name__} is not a TraderError subclass"
    finally:
        sys.path.pop(0)
