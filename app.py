from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from document_forgery_detector import DocumentForgeryDetector
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
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

    try:
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Initialize detector and analyze document
        detector = DocumentForgeryDetector()
        results = detector.analyze_document(filepath)

        # Clean up the temporary file
        os.remove(filepath)

        # Return results
        return jsonify({
            "filename": filename,
            "analysis_results": results
        })

    except Exception as e:
        # Clean up the temporary file if it exists
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            "error": "Analysis failed",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))