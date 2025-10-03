#!/usr/bin/env python3
"""
Setup script for Gonfanolier - Comprehensive Validation Framework
================================================================

Validation framework for the Borgia molecular computing system with
comprehensive analysis of fuzzy molecular representations, S-entropy
coordinates, and oscillatory cheminformatics.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Gonfanolier: Comprehensive Validation Framework for Borgia"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="gonfanolier",
    version="1.0.0",
    author="Borgia Framework Team",
    author_email="team@borgia.dev",
    description="Comprehensive Validation Framework for Fuzzy Molecular Representations and S-Entropy Coordinates",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/borgia-framework/gonfanolier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research", 
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=3.0',
            'black>=22.0',
            'flake8>=4.0',
            'mypy>=0.950',
            'pre-commit>=2.15',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=1.0',
            'sphinxcontrib-napoleon>=0.7',
        ],
        'viz': [
            'plotly>=5.0',
            'bokeh>=2.4',
            'dash>=2.0',
            'streamlit>=1.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'gonfanolier-validate=gonfanolier.run_all_validations:main',
            'gonfanolier-viz=gonfanolier.generate_scientific_visualizations:main',
            'gonfanolier-setup=gonfanolier.setup_environment:main',
        ],
    },
    package_data={
        'gonfanolier': [
            'public/agrafiotis-smarts-tar/*',
            'public/ahmed-smarts-tar/*', 
            'public/daylight-smarts-tar/*',
            'public/hann-smarts-tar/*',
            'public/walters-smarts-tar/*',
            'results/*',
            'src/*/*.py',
        ],
    },
    include_package_data=True,
    keywords="cheminformatics molecular-computing fuzzy-representations s-entropy oscillatory-mechanics validation",
    project_urls={
        "Bug Reports": "https://github.com/borgia-framework/gonfanolier/issues",
        "Source": "https://github.com/borgia-framework/gonfanolier",
        "Documentation": "https://gonfanolier.readthedocs.io/",
        "Research Paper": "https://arxiv.org/abs/xxxx.xxxx",
    },
)
