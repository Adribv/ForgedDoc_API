# Document Forgery Detection System

This system provides a comprehensive solution for detecting forged or manipulated documents and images using multiple analysis techniques:

1. Deep Learning Analysis using pre-trained models
2. Error Level Analysis (ELA)
3. Metadata Analysis
4. Noise Pattern Analysis

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

You can use the system in two ways:

### 1. Using the example script

```bash
python example.py path/to/your/image.jpg
```

### 2. Using the DocumentForgeryDetector class in your code

```python
from document_forgery_detector import DocumentForgeryDetector

detector = DocumentForgeryDetector()
results = detector.analyze_image("path/to/your/image.jpg")

# Access the results
print(f"Is forged: {results['is_forged']}")
print(f"Confidence: {results['confidence']}")
print(f"Detailed analysis: {results['analysis_details']}")
```

## Analysis Methods

The system uses multiple techniques to detect forgery:

1. **Deep Learning Analysis**: Uses a pre-trained model from Hugging Face specifically trained for document forgery detection.

2. **Error Level Analysis (ELA)**: Examines the compression artifacts in the image to identify areas that may have been modified.

3. **Metadata Analysis**: Checks for inconsistencies in image metadata, such as modification dates and software used.

4. **Noise Pattern Analysis**: Analyzes the noise patterns across the image to detect inconsistencies that might indicate manipulation.

## Results Interpretation

The system provides:
- A binary decision (`is_forged`)
- A confidence score (0-1)
- Detailed analysis results from each method

The final decision is weighted based on:
- Deep Learning Analysis (40% weight)
- Error Level Analysis (20% weight)
- Noise Pattern Analysis (20% weight)
- Metadata Analysis (20% weight)

## Requirements

- Python 3.7+
- See requirements.txt for detailed package dependencies

## Limitations

- The system works best with high-quality document images
- Performance may vary depending on the type and quality of forgery
- Deep learning model requires internet connection for first-time download
- Some analysis methods may produce false positives on heavily compressed images

## License

MIT License 