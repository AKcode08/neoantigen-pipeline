from setuptools import setup, find_packages

setup(
    name="neoantigens",
    version="0.1.0",
    description="Neoantigen discovery and validation pipeline",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
)
