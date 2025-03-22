from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from document_forgery_detector import DocumentForgeryDetector
import tempfile
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

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

        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
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
            if os.path.exists(filepath):
                os.remove(filepath)

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
        "message": "The file size exceeds the maximum allowed size (10MB)"
    }), 413

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 