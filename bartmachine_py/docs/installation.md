# Installation Guide

This document provides instructions for installing the bartMachine Python package.

## Prerequisites

Before installing bartMachine, you need to have the following prerequisites:

1. **Python 3.8 or higher**
   - The package is compatible with Python 3.8 and above.
   - You can check your Python version with `python3 --version`.

2. **Java 8 or higher**
   - The package requires a Java Development Kit (JDK) to be installed.
   - You can check your Java version with `java -version`.
   - Make sure the JAVA_HOME environment variable is set correctly.

## Installation Methods

### Installing from PyPI

The easiest way to install bartMachine is from PyPI:

```bash
pip install bartmachine
```

This will install the package and all its dependencies.

### Installing from Source

To install from source:

1. Clone the repository:
   ```bash
   git clone https://github.com/username/bartmachine.git
   ```

2. Navigate to the package directory:
   ```bash
   cd bartmachine
   ```

3. Install the package:
   ```bash
   pip install .
   ```

### Development Installation

For development purposes, you can install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

This will install the package in development mode, along with additional dependencies for development and testing.

## Verifying the Installation

To verify that the package is installed correctly, you can run a simple test:

```python
import bartmachine
print(bartmachine.__version__)
```

## Dependencies

The package has the following dependencies:

- **Core Dependencies**:
  - numpy>=1.19.0
  - pandas>=1.0.0
  - jpype1>=1.3.0
  - matplotlib>=3.3.0
  - scikit-learn>=0.23.0

- **Optional Dependencies**:
  - rpy2>=3.4.0 (for R integration and equivalence testing)

## Troubleshooting

### Java Issues

If you encounter issues with Java, make sure:

1. Java is installed and the JAVA_HOME environment variable is set correctly.
2. You have the correct Java version (Java 8 or higher).
3. The Java installation is accessible from your Python environment.

You can test the Java integration with:

```python
from bartmachine import initialize_jvm, is_jvm_running

initialize_jvm()
print(is_jvm_running())  # Should print True
```

### Package Import Issues

If you encounter issues importing the package, make sure:

1. The package is installed in your current Python environment.
2. You are using the correct Python interpreter.
3. The package name is spelled correctly (case-sensitive).

### Other Issues

For other issues, please check the [GitHub repository](https://github.com/username/bartmachine/issues) for known issues or to report a new issue.
