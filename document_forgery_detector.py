import os
import cv2
import numpy as np
from PIL import Image, ImageChops
import exifread
from skimage import measure
import imghdr
import logging
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import tempfile
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentForgeryDetector:
    def __init__(self):
        logger.info("Initializing Document Forgery Detector")
        self.temp_dir = tempfile.mkdtemp()
        
        # Set poppler path - check multiple locations
        poppler_paths = [
            '/usr/bin',  # Linux default
            '/usr/local/bin',  # Alternative Linux location
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'poppler', 'Library', 'bin'),  # Windows local
            os.path.join(os.environ.get('PROGRAMFILES', ''), 'poppler', 'bin'),  # Windows program files
            os.path.join(os.environ.get('PROGRAMFILES(X86)', ''), 'poppler', 'bin'),  # Windows program files (x86)
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'poppler-24.02.0', 'Library', 'bin'),  # Specific version
        ]
        
        self.poppler_path = None
        for path in poppler_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, 'pdftoppm.exe' if os.name == 'nt' else 'pdftoppm')):
                self.poppler_path = path
                logger.info(f"Found poppler at: {path}")
                break
        
        if not self.poppler_path:
            # If no specific path is found, try using system PATH
            try:
                import subprocess
                subprocess.run(['pdftoppm', '-v'], capture_output=True)
                self.poppler_path = None  # Let pdf2image find it in system PATH
                logger.info("Using system poppler installation")
            except Exception as e:
                logger.warning("Poppler not found in system. PDF analysis may fail.")
                logger.warning("Please ensure poppler-utils is installed")
            
        # Set Tesseract path
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            os.path.join(os.environ.get('PROGRAMFILES', ''), 'Tesseract-OCR', 'tesseract.exe'),
            os.path.join(os.environ.get('PROGRAMFILES(X86)', ''), 'Tesseract-OCR', 'tesseract.exe')
        ]
        
        tesseract_found = False
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                logger.info(f"Found Tesseract at: {path}")
                tesseract_found = True
                break
                
        if not tesseract_found:
            logger.warning("Tesseract not found in common locations. Text analysis may fail.")
            logger.warning("Please install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")

    def analyze_document(self, file_path):
        """
        Main function to analyze a document (image or PDF) for potential forgery
        Returns a dictionary with analysis results and confidence scores
        """
        results = {
            'is_forged': False,
            'confidence': 0.0,
            'analysis_details': {}
        }

        # Check if file exists
        if not os.path.exists(file_path):
            return {'error': 'File does not exist'}

        # Determine file type
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.pdf':
            results = self._analyze_pdf(file_path)
        else:
            results = self._analyze_image(file_path)

        return results

    def _analyze_pdf(self, pdf_path):
        """
        Analyze PDF document for potential forgery
        """
        results = {
            'is_forged': False,
            'confidence': 0.0,
            'analysis_details': {
                'file_type': 'pdf',
                'pages': [],
                'metadata': {},
                'inconsistencies': []
            }
        }

        try:
            # Read PDF
            pdf = PdfReader(pdf_path)
            
            # Analyze PDF metadata
            results['analysis_details']['metadata'] = self._analyze_pdf_metadata(pdf)
            
            # Convert PDF pages to images for analysis
            images = convert_from_path(pdf_path, poppler_path=self.poppler_path)
            
            # Analyze each page
            page_results = []
            for i, image in enumerate(images):
                # Save temporary image
                temp_image_path = os.path.join(self.temp_dir, f'page_{i}.png')
                image.save(temp_image_path, 'PNG')
                
                # Analyze the page image
                page_result = self._analyze_image(temp_image_path)
                page_result['analysis_details']['page_number'] = i + 1
                
                # Add text analysis for PDF
                page_result['analysis_details']['text_analysis'] = self._analyze_pdf_text(image)
                
                page_results.append(page_result)
                
                # Clean up temporary file
                os.remove(temp_image_path)

            # Combine results from all pages
            results['analysis_details']['pages'] = page_results
            
            # Check for cross-page inconsistencies
            inconsistencies = self._check_cross_page_inconsistencies(page_results)
            results['analysis_details']['inconsistencies'] = inconsistencies
            
            # Calculate final confidence score
            final_score = self._calculate_pdf_confidence(results)
            results['confidence'] = final_score
            results['is_forged'] = final_score > 0.5

        except Exception as e:
            logger.error(f"Error in PDF analysis: {e}")
            results['error'] = str(e)

        return results

    def _analyze_pdf_metadata(self, pdf):
        """
        Enhanced PDF metadata analysis with better forgery detection
        """
        metadata = {}
        critical_issues = []
        try:
            # Extract basic metadata
            metadata['author'] = pdf.metadata.get('/Author', '')
            metadata['creator'] = pdf.metadata.get('/Creator', '')
            metadata['producer'] = pdf.metadata.get('/Producer', '')
            metadata['creation_date'] = pdf.metadata.get('/CreationDate', '')
            metadata['modification_date'] = pdf.metadata.get('/ModDate', '')
            
            # Check for suspicious software with expanded list
            suspicious_software = [
                'photoshop', 'gimp', 'paint', 'editor', 'canva', 'illustrator',
                'pixlr', 'inkscape', 'affinity', 'sketch', 'figma'
            ]
            metadata['suspicious_software_detected'] = any(
                software in str(metadata).lower() 
                for software in suspicious_software
            )
            
            # Enhanced date analysis
            if metadata['creation_date'] or metadata['modification_date']:
                creation = self._parse_pdf_date(metadata['creation_date'])
                modification = self._parse_pdf_date(metadata['modification_date'])
                current_time = datetime.now()
                
                if creation and modification:
                    # Check for impossible time sequence
                    if modification < creation:
                        critical_issues.append("Modification time before creation time")
                        metadata['impossible_time_sequence'] = True
                    
                    # Check for future dates
                    if creation > current_time:
                        critical_issues.append("Creation date in the future")
                        metadata['future_creation_date'] = True
                    if modification > current_time:
                        critical_issues.append("Modification date in the future")
                        metadata['future_modification_date'] = True
                        
                    # Check for suspiciously close timestamps
                    time_diff = abs((modification - creation).total_seconds())
                    if 0 < time_diff < 5:  # Less than 5 seconds difference
                        critical_issues.append("Suspiciously close creation and modification times")
                        metadata['suspicious_timestamps'] = True
            
            # Additional checks
            metadata['is_encrypted'] = pdf.is_encrypted
            metadata['has_acroform'] = bool(pdf.get_form_text_fields())
            metadata['has_xfa'] = '/XFA' in pdf.get_fields() if pdf.get_fields() else False
            
            # Store critical issues
            metadata['critical_issues'] = critical_issues
            
        except Exception as e:
            logger.error(f"Error in PDF metadata analysis: {e}")
            metadata['error'] = str(e)
        
        return metadata

    def _analyze_pdf_text(self, image):
        """
        Analyze text in PDF for inconsistencies
        """
        text_analysis = {
            'font_inconsistencies': False,
            'spacing_irregularities': False,
            'character_confidence': 0.0
        }
        
        try:
            # Extract text and confidence scores using Tesseract
            text_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Analyze confidence scores
            confidences = [int(conf) for conf in text_data['conf'] if conf != '-1']
            if confidences:
                text_analysis['character_confidence'] = sum(confidences) / len(confidences) / 100.0
            
            # Analyze font sizes for inconsistencies
            heights = [int(height) for height in text_data['height'] if height != '-1']
            if heights:
                height_std = np.std(heights)
                height_mean = np.mean(heights)
                text_analysis['font_inconsistencies'] = (height_std / height_mean) > 0.5
            
            # Analyze spacing between words
            lefts = [int(left) for left in text_data['left'] if left != '-1']
            if len(lefts) > 1:
                spaces = np.diff(lefts)
                space_std = np.std(spaces)
                space_mean = np.mean(spaces)
                text_analysis['spacing_irregularities'] = (space_std / space_mean) > 0.5
                
        except Exception as e:
            logger.error(f"Error in PDF text analysis: {e}")
            text_analysis['error'] = str(e)
            
        return text_analysis

    def _check_cross_page_inconsistencies(self, page_results):
        """
        Check for inconsistencies across PDF pages
        """
        inconsistencies = []
        
        try:
            # Compare noise patterns across pages
            noise_patterns = [page['analysis_details']['noise_inconsistency'] 
                            for page in page_results if 'noise_inconsistency' in page['analysis_details']]
            if noise_patterns:
                noise_std = np.std(noise_patterns)
                if noise_std > 0.2:  # Threshold for noise pattern consistency
                    inconsistencies.append("Inconsistent noise patterns across pages")
            
            # Compare text confidence scores
            text_confidences = [page['analysis_details']['text_analysis']['character_confidence'] 
                              for page in page_results if 'text_analysis' in page['analysis_details']]
            if text_confidences:
                conf_std = np.std(text_confidences)
                if conf_std > 0.2:  # Threshold for text confidence consistency
                    inconsistencies.append("Inconsistent text quality across pages")
            
        except Exception as e:
            logger.error(f"Error in cross-page analysis: {e}")
            inconsistencies.append(f"Error in cross-page analysis: {str(e)}")
            
        return inconsistencies

    def _calculate_pdf_confidence(self, results):
        """
        Enhanced confidence calculation with better weighting for obvious forgery signs
        """
        scores = []
        weights = {
            'critical_metadata': 0.4,  # Increased weight for critical issues
            'metadata': 0.2,
            'page_analysis': 0.3,
            'cross_page': 0.1
        }
        
        try:
            # Critical metadata score (obvious signs of forgery)
            metadata = results['analysis_details']['metadata']
            critical_score = 0.0
            
            # Check for critical issues
            critical_issues = metadata.get('critical_issues', [])
            if critical_issues:
                # Each critical issue adds significant confidence
                critical_score = min(1.0, len(critical_issues) * 0.5)
            
            scores.append(critical_score * weights['critical_metadata'])
            
            # Regular metadata score
            metadata_score = 0.0
            if metadata.get('suspicious_software_detected', False):
                metadata_score += 0.3
            if metadata.get('is_encrypted', False):
                metadata_score += 0.2
            if metadata.get('has_xfa', False):
                metadata_score += 0.2
            scores.append(metadata_score * weights['metadata'])
            
            # Page analysis score with noise emphasis
            page_scores = []
            for page in results['analysis_details']['pages']:
                page_score = 0.0
                # Higher weight for noise inconsistency
                if page['analysis_details'].get('noise_inconsistency', 0) > 0.5:
                    page_score += 0.4
                if page['analysis_details'].get('copy_move_score', 0) > 0.3:
                    page_score += 0.3
                if page['analysis_details'].get('metadata_issues'):
                    page_score += 0.2
                page_scores.append(min(1.0, page_score))
            
            if page_scores:
                scores.append(max(page_scores) * weights['page_analysis'])
            
            # Cross-page inconsistency score
            cross_page_score = len(results['analysis_details']['inconsistencies']) * 0.3
            scores.append(min(cross_page_score, 1.0) * weights['cross_page'])
            
            # Calculate final score with emphasis on critical issues
            final_score = sum(scores)
            
            # Boost score if critical issues are found
            if critical_issues:
                final_score = max(final_score, 0.7)  # Minimum 70% confidence if critical issues exist
            
            return min(1.0, final_score)
            
        except Exception as e:
            logger.error(f"Error in PDF confidence calculation: {e}")
            return 0.5

    def _parse_pdf_date(self, date_string):
        """
        Parse PDF date string to datetime object
        """
        try:
            # Remove D: prefix and timezone if present
            date_string = date_string.replace("D:", "")[:14]
            return datetime.strptime(date_string, "%Y%m%d%H%M%S")
        except:
            return None

    def _analyze_image(self, image_path):
        """
        Enhanced image analysis with focus on obvious forgery signs
        """
        results = {
            'is_forged': False,
            'confidence': 0.0,
            'analysis_details': {
                'critical_issues': [],
                'visual_inconsistencies': [],
                'text_anomalies': []
            }
        }

        try:
            # Get image format
            image_format = imghdr.what(image_path)
            results['analysis_details']['image_format'] = image_format

            # Load image in different formats for various analyses
            cv_img = cv2.imread(image_path)
            pil_img = Image.open(image_path)
            gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            # 1. Text Analysis
            text_issues = self._analyze_text_consistency(cv_img)
            results['analysis_details']['text_anomalies'].extend(text_issues)

            # 2. Edge Detection for Cut-Paste
            edge_score = self._detect_edge_anomalies(gray_img)
            if edge_score > 0.6:
                results['analysis_details']['critical_issues'].append(
                    "Detected sharp edge transitions indicative of cut-paste"
                )

            # 3. Enhanced Copy-Move Detection
            copy_move_score = self._enhanced_copy_move_detection(gray_img)
            if copy_move_score > 0.5:
                results['analysis_details']['critical_issues'].append(
                    "Multiple identical regions detected"
                )

            # 4. Noise Pattern Analysis
            noise_score = self._analyze_noise_patterns(image_path)
            if noise_score > 0.7:
                results['analysis_details']['critical_issues'].append(
                    "Inconsistent noise patterns detected"
                )

            # 5. Color Consistency Check
            color_issues = self._check_color_consistency(cv_img)
            results['analysis_details']['visual_inconsistencies'].extend(color_issues)

            # 6. Compression Artifact Analysis
            compression_score = self._analyze_compression_artifacts(cv_img)
            if compression_score > 0.6:
                results['analysis_details']['visual_inconsistencies'].append(
                    "Inconsistent compression artifacts detected"
                )

            # Calculate final confidence with emphasis on critical issues
            results = self._calculate_image_confidence(results)

        except Exception as e:
            logger.error(f"Error during enhanced image analysis: {e}")
            results['error'] = str(e)

        return results

    def _validate_file(self, image_path):
        """Validate if the file is a valid image"""
        if not os.path.exists(image_path):
            return False
        
        # Check if it's a valid image using imghdr
        image_type = imghdr.what(image_path)
        if image_type is None:
            # Fallback to extension check
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
            return os.path.splitext(image_path)[1].lower() in valid_extensions
        return True

    def _error_level_analysis(self, image_path, quality=90):
        """
        Perform Error Level Analysis (ELA) to detect image manipulation
        """
        try:
            # Save image at a known quality level
            original = Image.open(image_path).convert('RGB')  # Convert to RGB to ensure JPEG compatibility
            temp_path = "temp_ela.jpg"
            original.save(temp_path, quality=quality)
            
            # Open both images and calculate difference
            saved = Image.open(temp_path)
            ela_image = ImageChops.difference(original, saved)
            extrema = ela_image.getextrema()
            
            # Calculate ELA score
            max_diff = max([ex[1] for ex in extrema])
            return max_diff / 255.0  # Normalize to 0-1 range
            
        except Exception as e:
            logger.error(f"Error in ELA analysis: {e}")
            return 0.0
        finally:
            if os.path.exists("temp_ela.jpg"):
                os.remove("temp_ela.jpg")

    def _detect_copy_move(self, image_path):
        """
        Detect copy-move forgery using keypoint matching
        """
        try:
            # Read image in grayscale
            img = cv2.imread(image_path, 0)
            if img is None:
                return 0.0

            # Initialize SIFT detector
            sift = cv2.SIFT_create()

            # Find keypoints and descriptors
            keypoints, descriptors = sift.detectAndCompute(img, None)
            
            if descriptors is None or len(descriptors) < 2:
                return 0.0

            # Match descriptors with themselves
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            matches = flann.knnMatch(descriptors, descriptors, k=2)
            
            # Apply ratio test and filter out self-matches
            good_matches = []
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance and m.queryIdx != m.trainIdx:
                    good_matches.append(m)

            # Calculate score based on number of good matches
            score = len(good_matches) / len(keypoints) if keypoints else 0
            return min(score, 1.0)  # Normalize to 0-1

        except Exception as e:
            logger.error(f"Error in copy-move detection: {e}")
            return 0.0

    def _analyze_metadata(self, image_path):
        """
        Analyze image metadata for inconsistencies
        """
        issues = []
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f)
                
            if not tags:
                issues.append("No metadata found")
                return issues

            # Check for common manipulation software tags
            software_tags = ['Image Software', 'Processing Software']
            for tag in software_tags:
                if tag in tags:
                    issues.append(f"Image processed with {tags[tag]}")

            # Check modification date vs creation date
            if 'EXIF DateTimeOriginal' in tags and 'EXIF DateTimeModified' in tags:
                if tags['EXIF DateTimeOriginal'].values != tags['EXIF DateTimeModified'].values:
                    issues.append("Creation and modification dates don't match")

        except Exception as e:
            logger.error(f"Error in metadata analysis: {e}")
            issues.append(f"Metadata analysis error: {str(e)}")

        return issues

    def _analyze_noise_patterns(self, image_path):
        """
        Analyze noise patterns for inconsistencies
        """
        try:
            image = cv2.imread(image_path, 0)  # Read as grayscale
            if image is None:
                return 1.0

            # Apply noise extraction filter
            noise = cv2.fastNlMeansDenoising(image) - image
            
            # Calculate local noise statistics
            local_std = []
            for i in range(0, noise.shape[0], 50):
                for j in range(0, noise.shape[1], 50):
                    patch = noise[i:min(i+50, noise.shape[0]), 
                               j:min(j+50, noise.shape[1])]
                    local_std.append(np.std(patch))

            # Calculate variation in noise patterns
            noise_variation = np.std(local_std) / np.mean(local_std) if np.mean(local_std) != 0 else 1.0
            return min(noise_variation, 1.0)  # Normalize to 0-1

        except Exception as e:
            logger.error(f"Error in noise analysis: {e}")
            return 1.0

    def _analyze_jpeg_quality(self, image_path):
        """
        Estimate JPEG quality and look for inconsistencies
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return 0.0

            # Convert to YCrCb color space
            img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            
            # Calculate DCT coefficients
            y_channel = img_ycrcb[:,:,0]
            h, w = y_channel.shape
            h = h - h % 8
            w = w - w % 8
            y_channel = y_channel[:h,:w]
            
            # Analyze 8x8 blocks
            quality_scores = []
            for i in range(0, h, 8):
                for j in range(0, w, 8):
                    block = y_channel[i:i+8, j:j+8].astype(np.float32)
                    dct = cv2.dct(block)
                    quality_scores.append(np.mean(np.abs(dct)))
            
            # Calculate quality score
            quality_variation = np.std(quality_scores) / np.mean(quality_scores) if np.mean(quality_scores) != 0 else 1.0
            return min(quality_variation, 1.0)  # Normalize to 0-1

        except Exception as e:
            logger.error(f"Error in JPEG quality analysis: {e}")
            return 0.0

    def _make_final_decision(self, results):
        """
        Combine all analysis results to make final decision
        """
        scores = []
        weights = {
            'ela': 0.25,
            'noise': 0.25,
            'jpeg': 0.15,
            'copy_move': 0.25,
            'metadata': 0.10
        }
        
        # Add ELA score if available
        if results['analysis_details'].get('ela_score') is not None:
            scores.append(results['analysis_details']['ela_score'] * weights['ela'])
        
        # Add noise score
        if 'noise_inconsistency' in results['analysis_details']:
            scores.append(results['analysis_details']['noise_inconsistency'] * weights['noise'])
        
        # Add JPEG quality score if available
        if results['analysis_details'].get('jpeg_quality') is not None:
            scores.append(results['analysis_details']['jpeg_quality'] * weights['jpeg'])
        
        # Add copy-move score
        if 'copy_move_score' in results['analysis_details']:
            scores.append(results['analysis_details']['copy_move_score'] * weights['copy_move'])
        
        # Add metadata score
        metadata_score = len(results['analysis_details'].get('metadata_issues', [])) * 0.1
        scores.append(min(metadata_score, 1.0) * weights['metadata'])

        # Calculate final confidence score
        final_score = sum(scores) / sum(weight for weight in weights.values() 
                                      if any(score * (1/weight) <= 1.0 for score in scores))
        
        results['is_forged'] = final_score > 0.5
        results['confidence'] = final_score

        return results 

    def _analyze_text_consistency(self, image):
        """
        Analyze text for inconsistencies in font, size, and alignment
        """
        issues = []
        try:
            # Convert to grayscale for text detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extract text and data using Tesseract
            text_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            
            # Analyze font sizes
            heights = [int(h) for h in text_data['height'] if h != '-1']
            if heights:
                height_std = np.std(heights)
                height_mean = np.mean(heights)
                if height_std / height_mean > 0.3:
                    issues.append("Multiple inconsistent font sizes detected")
            
            # Analyze text alignment
            lefts = [int(left) for left in text_data['left'] if left != '-1']
            if lefts:
                left_std = np.std(lefts)
                if left_std > 20:  # Threshold for alignment consistency
                    issues.append("Inconsistent text alignment detected")
            
            # Check for mixed font confidence
            confs = [int(conf) for conf in text_data['conf'] if conf != '-1']
            if confs:
                conf_std = np.std(confs)
                if conf_std > 15:  # High variance in confidence suggests mixed fonts
                    issues.append("Multiple font types detected")
            
            # Check for suspicious text patterns
            text = ' '.join([word for word in text_data['text'] if word.strip()])
            if re.search(r'[A-Za-z]+\s+[0-9]+\s+[A-Za-z]+', text):  # Suspicious mixing of text and numbers
                issues.append("Suspicious text patterns detected")
                
        except Exception as e:
            logger.error(f"Error in text consistency analysis: {e}")
            
        return issues

    def _detect_edge_anomalies(self, gray_image):
        """
        Detect unnatural edges indicating cut-paste operations
        """
        try:
            # Apply Canny edge detection
            edges = cv2.Canny(gray_image, 100, 200)
            
            # Look for rectangular patterns
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rectangular_score = 0
            for contour in contours:
                # Approximate the contour to a polygon
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's rectangular (4 points)
                if len(approx) == 4:
                    rectangular_score += 1
            
            return min(1.0, rectangular_score / 10)  # Normalize score
            
        except Exception as e:
            logger.error(f"Error in edge anomaly detection: {e}")
            return 0.0

    def _enhanced_copy_move_detection(self, gray_image):
        """
        Enhanced copy-move forgery detection
        """
        try:
            # Apply SIFT detection
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray_image, None)
            
            if descriptors is None or len(descriptors) < 2:
                return 0.0
            
            # Use FLANN matcher with stricter parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=100)  # Increased checks
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            matches = flann.knnMatch(descriptors, descriptors, k=2)
            
            # Apply stricter ratio test
            good_matches = []
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.6 * n.distance and m.queryIdx != m.trainIdx:  # Stricter ratio
                    good_matches.append(m)
            
            # Calculate score with higher weight for multiple matches
            score = len(good_matches) / len(keypoints) if keypoints else 0
            return min(score * 1.5, 1.0)  # Increase sensitivity
            
        except Exception as e:
            logger.error(f"Error in enhanced copy-move detection: {e}")
            return 0.0

    def _check_color_consistency(self, image):
        """
        Check for color inconsistencies that might indicate manipulation
        """
        issues = []
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Split into regions
            h, w = hsv.shape[:2]
            regions = []
            for i in range(0, h, h//4):
                for j in range(0, w, w//4):
                    region = hsv[i:i+h//4, j:j+w//4]
                    regions.append(region)
            
            # Compare color statistics between regions
            stats = []
            for region in regions:
                mean_hsv = np.mean(region, axis=(0,1))
                stats.append(mean_hsv)
            
            # Check for significant variations
            stats = np.array(stats)
            std_hsv = np.std(stats, axis=0)
            
            if std_hsv[0] > 30:  # High hue variation
                issues.append("Inconsistent color patterns detected")
            if std_hsv[1] > 50:  # High saturation variation
                issues.append("Suspicious color saturation variations")
                
        except Exception as e:
            logger.error(f"Error in color consistency check: {e}")
            
        return issues

    def _analyze_compression_artifacts(self, image):
        """
        Analyze compression artifacts for inconsistencies
        """
        try:
            # Convert to YCrCb
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            
            # Analyze each channel separately
            scores = []
            for channel in cv2.split(ycrcb):
                # Calculate DCT coefficients
                dct = cv2.dct(np.float32(channel))
                
                # Analyze high-frequency components
                high_freq = dct[5:, 5:]
                score = np.std(high_freq) / np.mean(np.abs(high_freq)) if np.mean(np.abs(high_freq)) != 0 else 1
                scores.append(score)
            
            return min(1.0, np.mean(scores))
            
        except Exception as e:
            logger.error(f"Error in compression artifact analysis: {e}")
            return 0.0

    def _calculate_image_confidence(self, results):
        """
        Calculate final confidence score with emphasis on critical issues
        """
        try:
            # Base weights
            weights = {
                'critical_issues': 0.4,
                'visual_inconsistencies': 0.3,
                'text_anomalies': 0.3
            }
            
            scores = []
            
            # Critical issues score (highest priority)
            critical_score = len(results['analysis_details']['critical_issues']) * 0.4
            scores.append(min(1.0, critical_score) * weights['critical_issues'])
            
            # Visual inconsistencies score
            visual_score = len(results['analysis_details']['visual_inconsistencies']) * 0.3
            scores.append(min(1.0, visual_score) * weights['visual_inconsistencies'])
            
            # Text anomalies score
            text_score = len(results['analysis_details']['text_anomalies']) * 0.3
            scores.append(min(1.0, text_score) * weights['text_anomalies'])
            
            # Calculate final confidence
            final_score = sum(scores)
            
            # Boost confidence if multiple types of issues are found
            if (len(results['analysis_details']['critical_issues']) > 0 and
                len(results['analysis_details']['visual_inconsistencies']) > 0 and
                len(results['analysis_details']['text_anomalies']) > 0):
                final_score = max(final_score, 0.9)  # At least 90% confidence if all types of issues are found
            
            # Ensure 100% confidence for obvious forgeries
            if len(results['analysis_details']['critical_issues']) >= 2:
                final_score = 1.0  # 100% confidence if multiple critical issues
            
            results['confidence'] = final_score
            results['is_forged'] = final_score > 0.5
            
            return results
            
        except Exception as e:
            logger.error(f"Error in confidence calculation: {e}")
            results['confidence'] = 0.5
            results['is_forged'] = False
            return results 