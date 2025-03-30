"""Install necessary dependencies for Ollama integration."""

import subprocess
import sys

def install_dependencies():
    """Install required packages for Ollama integration."""
    dependencies = ["httpx"]
    
    print(f"Installing dependencies: {', '.join(dependencies)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + dependencies)
        print("Successfully installed dependencies.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    return True

if __name__ == "__main__":
    install_dependencies() 