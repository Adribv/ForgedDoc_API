import os
import sys
import requests
import winreg
from pathlib import Path
import subprocess

def download_tesseract():
    """Download the latest Tesseract installer"""
    print("\nDownloading Tesseract...")
    url = "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe"
    
    response = requests.get(url, stream=True)
    installer_path = "tesseract_installer.exe"
    
    with open(installer_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return installer_path

def install_tesseract(installer_path):
    """Install Tesseract"""
    print("\nInstalling Tesseract...")
    print("Please follow the installation wizard when it appears.")
    print("Make sure to:")
    print("1. Install for all users (if possible)")
    print("2. Note the installation directory")
    print("3. Add Tesseract to system PATH when prompted")
    
    try:
        subprocess.run([installer_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running installer: {e}")
        return False
    
    return True

def verify_installation():
    """Verify Tesseract installation"""
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, 
                              text=True)
        if result.returncode == 0:
            print("\nTesseract installation verified!")
            print(f"Version info:\n{result.stdout}")
            return True
    except Exception:
        pass
    
    print("\nTesseract verification failed.")
    print("Please ensure Tesseract is properly installed and added to PATH")
    return False

def main():
    try:
        # Download installer
        installer_path = download_tesseract()
        
        # Install Tesseract
        if install_tesseract(installer_path):
            print("\nTesseract installation completed.")
            print("\nIMPORTANT: You need to restart your terminal/IDE for the PATH changes to take effect.")
        else:
            print("\nTesseract installation may have failed.")
            print("Please try installing manually from: https://github.com/UB-Mannheim/tesseract/wiki")
        
        # Clean up
        if os.path.exists(installer_path):
            os.remove(installer_path)
        
        # Verify installation
        if verify_installation():
            print("\nSetup complete! You can now use Tesseract OCR.")
        
    except Exception as e:
        print(f"\nSetup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 