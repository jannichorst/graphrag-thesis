#!/usr/bin/env python3
"""
Activate the Ollama integration for GraphRAG CLI commands.

This script:
1. Installs the graphrag_patch package to the current Python environment
2. Adds an import hook to ensure it's loaded when GraphRAG runs
3. Tests that the Ollama model type is registered

After running this script, GraphRAG CLI commands will recognize the 'ollama_chat' model type.
"""

import os
import sys
import subprocess
import site
import importlib
from pathlib import Path


def install_patch_package():
    """Install the graphrag_patch package to the current Python environment."""
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    patch_dir = current_dir / "graphrag_patch"
    
    if not patch_dir.exists():
        print(f"Error: graphrag_patch directory not found at {patch_dir}")
        return False
    
    print(f"Installing graphrag_patch from {current_dir}")
    
    # First, try to install required dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx"])
    
    # Install the package in development mode
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        print("Successfully installed graphrag_patch in development mode")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install in development mode, trying setup.py...")
        
    # If that fails, try creating a setup.py file and installing it
    setup_py = current_dir / "setup.py"
    if not setup_py.exists():
        with open(setup_py, "w") as f:
            f.write("""
from setuptools import setup, find_packages

setup(
    name="graphrag_patch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["httpx"],
)
""")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        print("Successfully installed graphrag_patch package")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing graphrag_patch package: {e}")
        return False


def create_sitecustomize():
    """Create a sitecustomize.py file to auto-import our patch on startup."""
    site_packages = site.getsitepackages()[0]
    sitecustomize_path = Path(site_packages) / "sitecustomize.py"
    
    # Check if file exists already and has our import
    if sitecustomize_path.exists():
        with open(sitecustomize_path, "r") as f:
            content = f.read()
            if "import graphrag_patch" in content:
                print("sitecustomize.py already contains our import")
                return True
    
    # Create or append to the file
    try:
        mode = "a" if sitecustomize_path.exists() else "w"
        with open(sitecustomize_path, mode) as f:
            f.write("""
# Auto-import graphrag_patch for Ollama integration
try:
    import graphrag_patch
except ImportError:
    pass
""")
        print(f"Updated {sitecustomize_path} to auto-import graphrag_patch")
        return True
    except Exception as e:
        print(f"Error updating sitecustomize.py: {e}")
        return False


def test_registration():
    """Test if the Ollama model is registered with GraphRAG."""
    try:
        from graphrag.config.enums import ModelType
        from graphrag.language_model.factory import ModelFactory
        
        # Import our package to trigger registration
        import graphrag_patch
        
        # Check if model is registered
        if ModelFactory.is_supported_chat_model(ModelType.OllamaChat):
            print(f"✅ Successfully registered {ModelType.OllamaChat} model type!")
            print("Available models:", ModelFactory.get_chat_models())
            return True
        else:
            print(f"❌ {ModelType.OllamaChat} model type is not registered.")
            print("Available models:", ModelFactory.get_chat_models())
            return False
    except Exception as e:
        print(f"Error testing registration: {e}")
        return False


def main():
    """Main function to activate the Ollama integration."""
    print("Activating Ollama integration for GraphRAG CLI...")
    
    success = install_patch_package()
    if not success:
        print("Failed to install graphrag_patch package.")
        return False
    
    success = create_sitecustomize()
    if not success:
        print("Failed to create sitecustomize.py.")
        return False
    
    success = test_registration()
    if not success:
        print("Failed to register Ollama model type.")
        return False
    
    print("\n✅ Ollama integration is now active!")
    print("\nYou can now use 'ollama_chat' as a model type in your settings.yaml")
    print("and run GraphRAG CLI commands like:")
    print("  graphrag index --root ./ragtest")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 