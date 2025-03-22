from document_forgery_detector import DocumentForgeryDetector
import sys
import json
import logging
import numpy as np
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def format_results(results):
    """Format the analysis results for display"""
    output = []
    
    # Handle error case
    if 'error' in results:
        return f"Error: {results['error']}"
    
    # Get file type
    file_type = results['analysis_details'].get('file_type', 'image')
    
    if file_type == 'pdf':
        # PDF-specific formatting
        output.append("\nPDF Analysis Results:")
        output.append("-" * 50)
        
        # Metadata
        output.append("\nMetadata Analysis:")
        metadata = results['analysis_details']['metadata']
        for key, value in metadata.items():
            output.append(f"  {key}: {value}")
        
        # Page Analysis
        output.append("\nPage Analysis:")
        for i, page in enumerate(results['analysis_details']['pages']):
            output.append(f"\nPage {page['analysis_details']['page_number']}:")
            output.append("  Image Analysis:")
            for key, value in page['analysis_details'].items():
                if key not in ['page_number', 'text_analysis']:
                    output.append(f"    {key}: {value}")
            
            output.append("  Text Analysis:")
            text_analysis = page['analysis_details']['text_analysis']
            for key, value in text_analysis.items():
                output.append(f"    {key}: {value}")
        
        # Inconsistencies
        if results['analysis_details']['inconsistencies']:
            output.append("\nDetected Inconsistencies:")
            for issue in results['analysis_details']['inconsistencies']:
                output.append(f"  - {issue}")
    else:
        # Image-specific formatting
        output.append("\nImage Analysis Results:")
        output.append("-" * 50)
        
        for key, value in results['analysis_details'].items():
            if isinstance(value, dict):
                output.append(f"\n{key}:")
                for subkey, subvalue in value.items():
                    output.append(f"  {subkey}: {subvalue}")
            else:
                output.append(f"{key}: {value}")
    
    # Final verdict
    output.append("\nFinal Verdict:")
    output.append("-" * 50)
    output.append(f"Is Forged: {bool(results['is_forged'])}")
    output.append(f"Confidence: {float(results['confidence']):.2%}")
    
    return "\n".join(output)

def main():
    try:
        if len(sys.argv) != 2:
            print("Usage: python example.py <path_to_file>")
            print("Supported formats: PDF, JPEG, PNG, BMP, TIFF, GIF")
            sys.exit(1)

        file_path = sys.argv[1]
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' does not exist")
            sys.exit(1)

        logger.debug(f"Initializing detector for file: {file_path}")
        
        detector = DocumentForgeryDetector()
        logger.debug("Detector initialized successfully")
        
        print(f"Analyzing file: {file_path}")
        results = detector.analyze_document(file_path)
        
        # Format and display results
        print(format_results(results))
        
        # Save detailed results to JSON file
        output_file = "analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"\nDetailed results saved to: {output_file}")
    
    except Exception as e:
        logger.exception("An error occurred during execution:")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 