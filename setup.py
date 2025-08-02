#!/usr/bin/env python3
"""
Setup script for CSV Embedding and FAISS Indexing package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="csv-embedding-faiss",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for loading CSV data, computing embeddings using OpenAI API, and creating FAISS indexes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/csv-embedding-faiss",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "embed-csv=embed_csv:main",
            "search-faiss=search_index:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 