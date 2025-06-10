# Setup.py Template for bartMachine Python Package

This document provides a template and guidelines for updating the setup.py file for the bartMachine Python package.

## Current Setup.py

The current setup.py file is located at `/Users/mark/Documents/Cline/bartMachine_port/bartmachine_py/setup.py`. It may need to be updated to include all necessary metadata, dependencies, and package data.

## Template

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from setuptools import setup, find_packages

# Get the long description from the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Get the version from the package
with open(os.path.join("bartmachine", "__init__.py"), encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.strip().split("=")[1].strip(" \"'")
            break
    else:
        version = "0.1.0"  # Default version if not found

# Define package data (non-Python files)
package_data = {
    "bartmachine": [
        "java/*.jar",
        "java/README.md",
        "java/compile_wrapper.sh",
    ],
}

# Define dependencies
install_requires = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "py4j>=0.10.9",
    "matplotlib>=3.4.0",
]

# Define development dependencies
extras_require = {
    "dev": [
        "pytest>=6.0.0",
        "pytest-cov>=2.12.0",
        "black>=21.5b2",
        "flake8>=3.9.0",
        "mypy>=0.812",
        "tox>=3.24.0",
    ],
    "docs": [
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=0.5.2",
        "myst-parser>=0.15.0",
    ],
}

# Define entry points (if needed)
entry_points = {
    "console_scripts": [
        # "bartmachine=bartmachine.cli:main",
    ],
}

# Define classifiers
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Statistics",
]

setup(
    name="bartmachine",
    version=version,
    description="Python implementation of the bartMachine package for Bayesian Additive Regression Trees",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/username/bartmachine",
    packages=find_packages(),
    package_data=package_data,
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points=entry_points,
    classifiers=classifiers,
    python_requires=">=3.8",
    keywords="bayesian, additive, regression, trees, bart, machine learning",
    license="MIT",
    platforms=["any"],
    zip_safe=False,  # Due to Java JAR files
)
```

## Key Components to Update

1. **Package Name**: Update the name to "bartmachine" or another appropriate name.

2. **Version**: Ensure the version is correctly extracted from the `__init__.py` file.

3. **Description**: Provide a clear and concise description of the package.

4. **Author Information**: Update with the correct author name and email.

5. **URL**: Update with the correct repository URL.

6. **Package Data**: Ensure all necessary Java JAR files and other non-Python files are included.

7. **Dependencies**: Update the dependencies with the correct versions.

8. **Classifiers**: Update the classifiers to accurately reflect the package's status, audience, and compatibility.

9. **License**: Update with the correct license.

## Additional Files

In addition to updating setup.py, you may need to create or update the following files:

1. **MANIFEST.in**: To include non-Python files in the source distribution.

```
include LICENSE
include README.md
include requirements.txt
recursive-include bartmachine/java *.jar
recursive-include bartmachine/java *.md
recursive-include bartmachine/java *.sh
recursive-include bartmachine/examples *.py
recursive-include bartmachine/tests *.py
```

2. **pyproject.toml**: To specify build system requirements.

```toml
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"
```

3. **setup.cfg**: For additional configuration.

```ini
[metadata]
license_files = LICENSE

[bdist_wheel]
universal = 0

[flake8]
max-line-length = 100
exclude = .git,__pycache__,build,dist

[tool:pytest]
testpaths = bartmachine/tests
python_files = test_*.py
python_functions = test_*
```

## Testing the Setup

After updating the setup.py file, you should test the package installation:

1. Create a source distribution:
   ```
   python setup.py sdist
   ```

2. Create a wheel distribution:
   ```
   python setup.py bdist_wheel
   ```

3. Install the package in development mode:
   ```
   pip install -e .
   ```

4. Test importing the package:
   ```python
   import bartmachine
   print(bartmachine.__version__)
   ```

5. Test the package with tox:
   ```
   tox
