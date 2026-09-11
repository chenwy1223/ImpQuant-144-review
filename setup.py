from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="impquant",
    version="0.1.0",
    description="Importance-aware quantization for vision-language models",
    long_description=Path(__file__).with_name("README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["qmllm", "qmllm.*"]),
    python_requires=">=3.10",
)
