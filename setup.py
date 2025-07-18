"""
Setup script for the MAZE reinforcement learning library.

This script configures the package for installation via pip, including
dependencies, metadata, and entry points.
"""

from setuptools import setup, find_packages
import os

# Read the README file for the long description
def read_readme():
    """Read README.md file for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements from requirements.txt if it exists
def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'numpy>=1.20.0',
        'pygame>=2.0.0',
        'matplotlib>=3.3.0',
        'tqdm>=4.60.0'
    ]

setup(
    name="maze-rl",
    version="1.0.0",
    author="V1centFaure",
    author_email="",
    description="A reinforcement learning library for maze environments with interactive visualization",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/V1centFaure/maze",
    project_urls={
        "Bug Reports": "https://github.com/V1centFaure/maze/issues",
        "Source": "https://github.com/V1centFaure/maze",
        "Documentation": "https://github.com/V1centFaure/maze/blob/main/README.md",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
        "Topic :: Games/Entertainment :: Puzzle Games",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "maze-demo=maze.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "reinforcement learning",
        "maze",
        "q-learning",
        "sarsa",
        "epsilon-greedy",
        "artificial intelligence",
        "machine learning",
        "pygame",
        "visualization",
        "pathfinding",
        "navigation",
        "agent",
        "environment",
    ],
    zip_safe=False,
)
