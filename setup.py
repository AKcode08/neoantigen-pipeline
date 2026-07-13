from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent


def read_requirements():
    """Parse requirements.txt, skipping comments and blank lines."""
    req_file = ROOT / "requirements.txt"
    if not req_file.exists():
        return []
    return [
        line.strip()
        for line in req_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


setup(
    name="neoantigen-pipeline",
    version="0.1.0",
    description=(
        "Meta-analysis pipeline for neoantigen candidate prioritization: "
        "5-layer prediction ensemble with LLM synthesis, validated on ITSNdb"
    ),
    long_description=(ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else "",
    long_description_content_type="text/markdown",
    author="Aman Kumar",
    url="https://github.com/AKcode08/neoantigen-pipeline",
    license="MIT",

    packages=find_packages(include=["src", "src.*", "validation", "validation.*"]),

    # MHCflurry requires the `pipes` stdlib module, removed in Python 3.13.
    # Tested on 3.11 only. Do not widen this range without testing.
    python_requires=">=3.11,<3.13",

    install_requires=read_requirements(),

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
