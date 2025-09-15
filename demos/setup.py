"""
Simplified Borgia validation framework setup
"""

from setuptools import setup, find_packages

setup(
    name='borgia_validation',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'matplotlib>=3.5.0',
        'pytest>=7.0.0'
    ],
    extras_require={
        'advanced': [
            'scipy>=1.9.0; python_version<"3.13"'
        ]
    },
    author='Borgia Development Team',
    description='Simplified validation framework for the Borgia Rust implementation',
    python_requires='>=3.8',
)