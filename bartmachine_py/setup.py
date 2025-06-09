"""
Setup script for the bartmachine package.
"""

from setuptools import setup, find_packages

setup(
    name="bartmachine",
    version="0.1.0",
    author="Mark",
    author_email="mark@example.com",
    description="Python port of bartMachine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/bartmachine_py",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "jpype1>=1.3.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.23.0",
    ],
    extras_require={
        "test": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "pytest-xdist>=2.3.0",
            "rpy2>=3.4.0",
        ],
    },
    package_data={
        "bartmachine": ["java/*.jar"],
    },
)
