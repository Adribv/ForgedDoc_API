from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from document_forgery_detector import DocumentForgeryDetector
import tempfile
import gc
import logging
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'pdf_processing')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def cleanup_temp_files():
    """Clean up temporary files in the upload folder"""
    try:
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error cleaning temp directory: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal, cleaning up...")
    cleanup_temp_files()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return jsonify({
        "status": "active",
        "message": "Document Forgery Detection API",
        "endpoints": {
            "/analyze": "POST - Analyze document for forgery (send file in form-data with key 'document')"
        }
    })

@app.route('/analyze', methods=['POST'])
def analyze_document():
    gc.collect()  # Force garbage collection before processing
    
    try:
        # Check if a file was sent
        if 'document' not in request.files:
            return jsonify({
                "error": "No document file provided",
                "message": "Please send a file with key 'document'"
            }), 400

        file = request.files['document']

        # Check if a file was selected
        if file.filename == '':
            return jsonify({
                "error": "No file selected",
                "message": "Please select a file to analyze"
            }), 400

        # Check if the file type is allowed
        if not allowed_file(file.filename):
            return jsonify({
                "error": "Invalid file type",
                "message": f"Allowed file types are: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

        # Save the file temporarily with a unique name
        filename = secure_filename(file.filename)
        unique_filename = f"{os.getpid()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            file.save(filepath)
            
            # Initialize detector and analyze document
            detector = DocumentForgeryDetector()
            results = detector.analyze_document(filepath)

            # Clean up
            del detector
            gc.collect()

            return jsonify({
                "filename": filename,
                "analysis_results": results
            })

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return jsonify({
                "error": "Analysis failed",
                "message": str(e)
            }), 500

        finally:
            # Clean up the temporary file
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                logger.error(f"Error cleaning up file {filepath}: {e}")

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({
            "error": "Server error",
            "message": str(e)
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "error": "File too large",
        "message": "The file size exceeds the maximum allowed size (8MB)"
    }), 413

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port) 