"""
setup.py — Nexa Package Installer
==================================
Makes `nexa` an installable Python package.

Usage:
    pip install -e .          # Editable install (recommended for development)
    pip install .             # Standard install

After installing, anywhere in your Python environment you can do:
    from nexa.utils import get_logger
    from nexa.models import NexaTransformer
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read dependencies from requirements.txt, skipping comments and blank lines
def parse_requirements(filename: str) -> list[str]:
    reqs = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments, blank lines, and inline comments
            if not line or line.startswith("#"):
                continue
            # Strip inline comments (e.g.  "torch>=2.2.0  # PyTorch")
            dep = line.split("#")[0].strip()
            if dep:
                reqs.append(dep)
    return reqs


setup(
    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------
    name="nexa",
    version="0.1.0",
    author="Nexa Team",
    description="Nexa — An independent AI system built from scratch with PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/nexa",

    # -------------------------------------------------------------------------
    # Package Discovery
    # -------------------------------------------------------------------------
    # find_packages() automatically finds all directories containing __init__.py
    packages=find_packages(exclude=["tests*", "scripts*", "docs*"]),
    python_requires=">=3.10",

    # -------------------------------------------------------------------------
    # Dependencies
    # -------------------------------------------------------------------------
    install_requires=parse_requirements("requirements.txt"),

    # -------------------------------------------------------------------------
    # CLI Entry Points (added in later phases)
    # -------------------------------------------------------------------------
    entry_points={
        "console_scripts": [
            # Phase 4+: nexa-train — launch training from the command line
            # "nexa-train=nexa.training.trainer:main",
            # Phase 5+: nexa-generate — run text generation from CLI
            # "nexa-generate=nexa.inference.generator:main",
        ]
    },

    # -------------------------------------------------------------------------
    # Metadata / PyPI classifiers
    # -------------------------------------------------------------------------
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
