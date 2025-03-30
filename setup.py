from setuptools import setup, find_packages

setup(
    name="graphrag_patch",
    version="0.1.0",
    description="Ollama integration for GraphRAG",
    author="GraphRAG User",
    packages=find_packages(),
    install_requires=["httpx"],
    python_requires=">=3.9",
) 