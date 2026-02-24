# Flask Backend for SRNet Steganography Suite
# File: backend/app.py

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import io
import base64
from PIL import Image
import numpy as np
import torch
import cv2
from datetime import datetime

# Import algorithm modules
from algorithms.lsb import LSBSteganography
from algorithms.dct import DCTSteganography
from algorithms.dwt import DWTSteganography
from algorithms.pvd import PVDSteganography
from algorithms.edge_adaptive import EdgeAdaptiveSteganography
from algorithms.detector import SteganographyDetector
from models.srnet import SRNet

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('weights', exist_ok=True)

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
srnet_model = None

def load_srnet_model():
    """Load pre-trained SRNet model"""
    global srnet_model
    try:
        srnet_model = SRNet().to(device)
        # Use absolute path relative to this script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(script_dir, 'weights', 'srnet_best.pth')
        if os.path.exists(weight_path):
            srnet_model.load_state_dict(torch.load(weight_path, map_location=device))
            srnet_model.eval()
            print(f"✓ SRNet model loaded from {weight_path}")
        else:
            print(f"⚠ No pre-trained weights found. Using untrained model.")
            print(f"  Place trained weights at: {weight_path}")
    except Exception as e:
        print(f"⚠ Error loading SRNet: {e}")

# Initialize algorithms
lsb_algo = LSBSteganography()
dct_algo = DCTSteganography()
dwt_algo = DWTSteganography()
pvd_algo = PVDSteganography()
edge_algo = EdgeAdaptiveSteganography()
detector = SteganographyDetector()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    img = Image.fromarray(image_array.astype('uint8'))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def base64_to_image(base64_string):
    """Convert base64 string to numpy array"""
    img_data = base64.b64decode(base64_string.split(',')[1] if ',' in base64_string else base64_string)
    img = Image.open(io.BytesIO(img_data))
    return np.array(img)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'device': str(device),
        'srnet_loaded': srnet_model is not None,
        'algorithms': ['lsb', 'dct', 'dwt', 'pvd', 'edge']
    })

@app.route('/api/detect', methods=['POST'])
def detect_steganography():
    """Detect steganography in uploaded image using SRNet"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Read image
        img = Image.open(file.stream).convert('RGB')
        img_array = np.array(img)
        
        start_time = datetime.now()
        
        # Run SRNet detection if model is loaded and trained
        if srnet_model is not None:
            with torch.no_grad():
                # Convert to grayscale and ensure 256x256
                gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                gray_img = cv2.resize(gray_img, (256, 256))
                
                # Preprocess image - SRNet expects raw grayscale values [0,255]
                img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(0).float()
                img_tensor = img_tensor.to(device)
                
                # Run inference
                output = srnet_model(img_tensor)
                probability = torch.softmax(output, dim=1)[0, 1].item()  # Probability of stego
                
                # Only use SRNet if confidence is high, otherwise fall back to statistical
                if probability > 0.6 or probability < 0.4:  # Confident prediction
                    is_stego = probability > 0.5
                    confidence = probability * 100 if is_stego else (1 - probability) * 100
                else:
                    # Fall back to statistical detection
                    detection_results = detector.detect(img_array)
                    is_stego = detection_results['is_stego']
                    confidence = detection_results['confidence']
        else:
            # Use statistical detection
            detection_results = detector.detect(img_array)
            is_stego = detection_results['is_stego']
            confidence = detection_results['confidence']
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Statistical analysis for technique detection
        stats = detector.analyze_techniques(img_array)
        
        result = {
            'is_stego': is_stego,
            'confidence': round(confidence, 2),
            'processing_time': round(processing_time, 2),
            'detected_technique': stats['likely_technique'] if is_stego else None,
            'embedding_strength': stats['embedding_strength'] if is_stego else None,
            'analysis': {
                'chi_square': stats['chi_square'],
                'rs_analysis': stats['rs_analysis'],
                'histogram_score': stats['histogram_score']
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/encode', methods=['POST'])
def encode_message():
    """Encode secret message into image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        message = request.form.get('message', '')
        technique = request.form.get('technique', 'lsb')
        password = request.form.get('password', '')
        strength = int(request.form.get('strength', 5))
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Read image
        img = Image.open(file.stream).convert('RGB')
        cover_image = np.array(img)
        
        start_time = datetime.now()
        
        # Select algorithm
        algo_map = {
            'lsb': lsb_algo,
            'dct': dct_algo,
            'dwt': dwt_algo,
            'pvd': pvd_algo,
            'edge': edge_algo
        }
        
        algo = algo_map.get(technique, lsb_algo)
        
        # Encode message
        stego_image = algo.encode(cover_image, message, password, strength)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate quality metrics
        psnr = detector.calculate_psnr(cover_image, stego_image)
        mse = detector.calculate_mse(cover_image, stego_image)
        
        # Convert images to base64
        cover_base64 = image_to_base64(cover_image)
        stego_base64 = image_to_base64(stego_image)
        
        result = {
            'success': True,
            'cover_image': f'data:image/png;base64,{cover_base64}',
            'stego_image': f'data:image/png;base64,{stego_base64}',
            'technique': technique,
            'message_length': len(message),
            'encrypted': bool(password),
            'psnr': round(psnr, 2),
            'mse': round(mse, 4),
            'processing_time': round(processing_time, 2),
            'cover_size': cover_image.nbytes,
            'stego_size': stego_image.nbytes
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/decode', methods=['POST'])
def decode_message():
    """Decode hidden message from stego image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        technique = request.form.get('technique', 'auto')
        password = request.form.get('password', '')
        
        # Read image
        img = Image.open(file.stream).convert('RGB')
        stego_image = np.array(img)
        
        start_time = datetime.now()
        
        # Auto-detect technique if requested
        if technique == 'auto':
            stats = detector.analyze_techniques(stego_image)
            technique = stats['likely_technique']
        
        # Select algorithm
        algo_map = {
            'lsb': lsb_algo,
            'dct': dct_algo,
            'dwt': dwt_algo,
            'pvd': pvd_algo,
            'edge': edge_algo
        }
        
        algo = algo_map.get(technique, lsb_algo)
        
        # Decode message
        message = algo.decode(stego_image, password)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'success': True,
            'message': message,
            'technique': technique,
            'message_length': len(message),
            'decrypted': bool(password),
            'processing_time': round(processing_time, 2)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-detect', methods=['POST'])
def batch_detect():
    """Batch detection for multiple images"""
    try:
        files = request.files.getlist('images')
        results = []
        
        for file in files:
            if allowed_file(file.filename):
                img = Image.open(file.stream).convert('RGB')
                img_array = np.array(img)
                
                detection = detector.detect(img_array)
                results.append({
                    'filename': secure_filename(file.filename),
                    'is_stego': detection['is_stego'],
                    'confidence': round(detection['confidence'], 2)
                })
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🔒 SRNet Steganography Suite - Backend Server")
    print("=" * 60)
    print(f"Device: {device}")
    print("Loading models...")
    load_srnet_model()
    print("=" * 60)
    print("✓ Server ready!")
    print("  API Endpoint: http://localhost:5000")
    print("  Algorithms: LSB, DCT, DWT, PVD, Edge-Adaptive")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
