#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup script for the bartmachine package.

This package provides a Python implementation of the bartMachine R package,
which implements Bayesian Additive Regression Trees (BART).
"""

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
        "java/*.md",
        "java/*.sh",
        "java/*.class",
        "java/src/**/*.java",
        "java/bartMachine/**/*.class",
        "java/AlgorithmTesting/**/*.class",
        "java/CustomLogging/**/*.class",
        "java/OpenSourceExtensions/**/*.class",
    ],
}

# Define dependencies
install_requires = [
    "numpy>=1.19.0",
    "pandas>=1.0.0",
    "jpype1>=1.3.0",
    "matplotlib>=3.3.0",
    "scikit-learn>=0.23.0",
]

# Define development dependencies
extras_require = {
    "dev": [
        "pytest>=6.0.0",
        "pytest-cov>=2.12.0",
        "pytest-xdist>=2.3.0",
        "black>=21.5b2",
        "flake8>=3.9.0",
        "isort>=5.9.1",
        "mypy>=0.812",
        "tox>=3.24.0",
    ],
    "test": [
        "pytest>=6.0.0",
        "pytest-cov>=2.12.0",
        "pytest-xdist>=2.3.0",
        "rpy2>=3.4.0",
    ],
    "docs": [
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=0.5.2",
        "myst-parser>=0.15.0",
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
    author="Mark",
    author_email="mark@example.com",
    url="https://github.com/username/bartmachine",
    packages=find_packages(),
    package_data=package_data,
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=classifiers,
    python_requires=">=3.8",
    keywords="bayesian, additive, regression, trees, bart, machine learning",
    license="MIT",
    platforms=["any"],
    zip_safe=False,  # Due to Java JAR files
)
