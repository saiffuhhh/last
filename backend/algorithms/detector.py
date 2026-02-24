import numpy as np
import cv2
from scipy import stats

class SteganographyDetector:
    """Detection algorithms for steganography"""
    
    def calculate_psnr(self, img1, img2):
        """Calculate PSNR between two images"""
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return 100
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def calculate_mse(self, img1, img2):
        """Calculate MSE"""
        return np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    
    def chi_square_attack(self, image):
        """Chi-square attack for LSB detection"""
        histogram = np.histogram(image.flatten(), bins=256, range=(0, 256))[0]
        
        chi_square = 0
        for i in range(0, 256, 2):
            if histogram[i] + histogram[i+1] > 0:
                expected = (histogram[i] + histogram[i+1]) / 2
                chi_square += ((histogram[i] - expected) ** 2 + 
                              (histogram[i+1] - expected) ** 2) / expected
        
        return chi_square
    
    def rs_analysis(self, image):
        """RS analysis for steganography detection"""
        # Simplified RS analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Calculate smoothness
        dx = np.diff(gray, axis=1)
        dy = np.diff(gray, axis=0)
        
        smoothness = np.sum(np.abs(dx)) + np.sum(np.abs(dy[:, :-1]))
        total_pixels = gray.shape[0] * gray.shape[1]
        
        rs_score = smoothness / total_pixels
        return rs_score
    
    def detect(self, image):
        """Detect steganography"""
        chi_square = self.chi_square_attack(image)
        rs_score = self.rs_analysis(image)
        
        # Thresholds based on dataset statistics
        # Stego images typically have lower chi-square values
        chi_threshold = 3000  # Below this suggests steganography
        rs_threshold = 210   # Above this suggests steganography
        
        is_stego = chi_square < chi_threshold or rs_score > rs_threshold
        confidence = min(100, max(0, (chi_threshold - chi_square) / chi_threshold * 50 + (rs_score - rs_threshold) / rs_threshold * 50))
        
        return {
            'is_stego': is_stego,
            'confidence': confidence,
            'chi_square': chi_square,
            'rs_analysis': rs_score
        }
    
    def analyze_techniques(self, image):
        """Analyze which technique was likely used"""
        chi_square = self.chi_square_attack(image)
        rs_score = self.rs_analysis(image)
        
        # Heuristic detection - lower chi_square indicates steganography
        if chi_square < 200:
            technique = "LSB"
            strength = "High"
        elif rs_score > 60:
            technique = "DCT/DWT"
            strength = "Medium"
        elif rs_score > 40:
            technique = "PVD"
            strength = "Low"
        else:
            technique = "Edge-based/Advanced"
            strength = "Very Low"
        
        return {
            'likely_technique': technique,
            'embedding_strength': strength,
            'chi_square': chi_square,
            'rs_analysis': rs_score,
            'histogram_score': np.std(np.histogram(image.flatten(), bins=256)[0])
        }