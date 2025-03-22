import os
import sys
import requests
import zipfile
import winreg
import subprocess
from pathlib import Path

def check_poppler_in_path():
    """Check if poppler is already in PATH"""
    paths = os.environ.get('PATH', '').split(';')
    print("\nChecking PATH for poppler:")
    found_paths = []
    for path in paths:
        if 'poppler' in path.lower():
            found_paths.append(path)
            pdftoppm_path = os.path.join(path, 'pdftoppm.exe')
            if os.path.exists(pdftoppm_path):
                print(f"Found poppler in: {path}")
                print(f"pdftoppm.exe exists: Yes")
                return True
            else:
                print(f"Found poppler-like path but pdftoppm.exe not found in: {path}")
    
    if not found_paths:
        print("No poppler directories found in PATH")
    return False

def download_poppler():
    """Download the latest poppler release"""
    print("\nDownloading poppler...")
    url = "https://github.com/oschwartz10612/poppler-windows/releases/download/v24.02.0-0/Release-24.02.0-0.zip"
    response = requests.get(url, stream=True)
    
    # Save the zip file
    with open("poppler.zip", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def get_user_install_dir():
    """Get user-specific installation directory"""
    # Try to use Local AppData first
    local_appdata = os.environ.get('LOCALAPPDATA')
    if local_appdata:
        return os.path.join(local_appdata, 'Programs', 'poppler')
    
    # Fallback to user's home directory
    return os.path.join(os.path.expanduser('~'), 'poppler')

def update_user_path(bin_path):
    """Update user's PATH environment variable"""
    try:
        # Open the registry key for the user's PATH
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment', 0, winreg.KEY_ALL_ACCESS)
        
        try:
            path = winreg.QueryValueEx(key, 'Path')[0]
        except WindowsError:
            path = ''
        
        # Add poppler bin if not already in PATH
        if bin_path not in path:
            new_path = f"{path};{bin_path}" if path else bin_path
            winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, new_path)
            
        winreg.CloseKey(key)
        print("Added poppler to user PATH")
        
        # Also update current process PATH
        os.environ['PATH'] = f"{os.environ.get('PATH', '')};{bin_path}"
        return True
        
    except Exception as e:
        print(f"Error updating PATH: {e}")
        return False

def install_poppler():
    """Extract poppler and add to PATH"""
    # Get user-specific installation directory
    poppler_dir = get_user_install_dir()
    
    print(f"\nInstalling poppler to {poppler_dir}...")
    
    try:
        # Create installation directory if it doesn't exist
        os.makedirs(poppler_dir, exist_ok=True)
        
        # Extract the zip file
        with zipfile.ZipFile("poppler.zip", 'r') as zip_ref:
            zip_ref.extractall(poppler_dir)
        
        # The zip file contains a nested directory structure, find the bin directory
        bin_path = None
        for root, dirs, files in os.walk(poppler_dir):
            if 'pdftoppm.exe' in files:
                bin_path = root
                break
        
        if not bin_path:
            raise Exception("Could not find pdftoppm.exe in extracted files")
        
        print(f"Found poppler binaries in: {bin_path}")
        
        # Try to update user's PATH
        path_updated = update_user_path(bin_path)
        
        if not path_updated:
            print("\nPlease manually add the following directory to your PATH:")
            print(bin_path)
            print("\nYou can do this by:")
            print("1. Press Win + R")
            print("2. Type 'systempropertiesadvanced' and press Enter")
            print("3. Click 'Environment Variables'")
            print("4. Under 'User variables', edit 'Path'")
            print("5. Click 'New' and add the above directory")
        
        # Clean up
        os.remove("poppler.zip")
        
        return bin_path
        
    except Exception as e:
        print(f"\nError during installation: {e}")
        print("\nAlternative installation method:")
        print("1. Download poppler manually from: https://github.com/oschwartz10612/poppler-windows/releases")
        print("2. Extract the zip file to a location of your choice")
        print("3. Add the 'bin' directory to your PATH environment variable")
        raise

def verify_installation(bin_path):
    """Verify poppler installation"""
    try:
        pdftoppm_path = os.path.join(bin_path, 'pdftoppm.exe')
        if os.path.exists(pdftoppm_path):
            print("\nPoppler files found successfully!")
            # Try to run pdftoppm
            try:
                result = subprocess.run([pdftoppm_path, '-v'], 
                                     capture_output=True, 
                                     text=True)
                if result.returncode == 0:
                    print("pdftoppm test: Success!")
                    return True
                else:
                    print(f"pdftoppm test failed: {result.stderr}")
            except Exception as e:
                print(f"Error running pdftoppm: {e}")
    except Exception:
        pass
    return False

def main():
    try:
        print("Checking current poppler installation...")
        if check_poppler_in_path():
            print("\nPoppler is already installed and in PATH!")
            sys.exit(0)
            
        print("\nNo working poppler installation found. Installing...")
        download_poppler()
        bin_path = install_poppler()
        
        if verify_installation(bin_path):
            print("\nPoppler installation complete!")
            print("\nIMPORTANT: You need to restart your terminal/IDE for the PATH changes to take effect.")
            print("However, you can try running your PDF analysis now - the PATH has been updated for the current process.")
            print(f"\nVerify installation by running: {bin_path}\\pdftoppm -v")
        else:
            print("\nWarning: Installation may not have completed successfully.")
            print("Please follow the manual installation instructions above.")
        
    except Exception as e:
        print(f"\nSetup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 