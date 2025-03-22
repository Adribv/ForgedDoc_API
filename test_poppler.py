import os
import sys
from pdf2image import pdfinfo_from_path
from pathlib import Path

def test_poppler():
    print("\nTesting Poppler Installation:")
    print("-" * 50)
    
    # Print current PATH
    print("\nCurrent PATH:")
    paths = os.environ.get('PATH', '').split(';')
    for path in paths:
        if path.strip():
            print(f"  {path}")
    
    # Look for poppler in PATH
    print("\nSearching for poppler in PATH:")
    found = False
    for path in paths:
        if 'poppler' in path.lower():
            pdftoppm_path = os.path.join(path, 'pdftoppm.exe')
            print(f"Found poppler-like directory: {path}")
            print(f"pdftoppm.exe exists: {os.path.exists(pdftoppm_path)}")
            found = True
    
    if not found:
        print("No poppler directories found in PATH")
    
    # Test with a PDF file
    print("\nTesting PDF processing:")
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if os.path.exists(pdf_path):
            try:
                info = pdfinfo_from_path(pdf_path)
                print(f"\nSuccess! PDF info retrieved:")
                print(f"  Pages: {info['Pages']}")
                print(f"  Author: {info.get('Author', 'Not specified')}")
                print(f"  Creator: {info.get('Creator', 'Not specified')}")
            except Exception as e:
                print(f"\nError processing PDF: {e}")
        else:
            print(f"\nError: PDF file not found: {pdf_path}")
    else:
        print("\nNo PDF file specified for testing")
        print("Usage: python test_poppler.py <pdf_file>")

if __name__ == "__main__":
    test_poppler() 