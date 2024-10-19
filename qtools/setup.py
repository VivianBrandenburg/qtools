#!python
from setuptools import find_packages, setup

setup(
    name='qtools',
    version = "1.0.0",
    description="Toolbox to prepare data for siamese and quartet models, track this training and explore the trained models with in silico mutagenesis",
    url="github-url",
    author="Vivian Brandenburg",
    author_email="qtools@vivian-brandenburg.de",
    
    classifiers=[
    "Intended Audience :: Researchers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3"],
    
    packages=find_packages(),
    python_requires=">=3.6, <4",
)
