from algorithms.detector import SteganographyDetector
import cv2
import numpy as np

detector = SteganographyDetector()

# Test on 50 images to understand the chi-square distribution
cover_chi_squares = []
stego_chi_squares = []

for i in range(1, 51):
    cover_path = f'../dataset/bossbase_raw/{i}.pgm'
    cover = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    
    stego_path = f'../dataset/stego/{i}.png'
    stego = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
    
    if cover is not None and stego is not None:
        result_cover = detector.detect(cover)
        result_stego = detector.detect(stego)
        
        cover_chi_squares.append(result_cover["chi_square"])
        stego_chi_squares.append(result_stego["chi_square"])

cover_chi_squares = np.array(cover_chi_squares)
stego_chi_squares = np.array(stego_chi_squares)

print("Cover Chi-Square Statistics:")
print(f"  Min: {cover_chi_squares.min():.1f}")
print(f"  Max: {cover_chi_squares.max():.1f}")
print(f"  Mean: {cover_chi_squares.mean():.1f}")
print(f"  Median: {np.median(cover_chi_squares):.1f}")
print(f"  Std: {cover_chi_squares.std():.1f}")

print("\nStego Chi-Square Statistics:")
print(f"  Min: {stego_chi_squares.min():.1f}")
print(f"  Max: {stego_chi_squares.max():.1f}")
print(f"  Mean: {stego_chi_squares.mean():.1f}")
print(f"  Median: {np.median(stego_chi_squares):.1f}")
print(f"  Std: {stego_chi_squares.std():.1f}")

# Find optimal threshold
print("\nAnalysis:")
print(f"  Cover max: {cover_chi_squares.max():.1f}")
print(f"  Stego min: {stego_chi_squares.min():.1f}")
print(f"  Suggested threshold: {(cover_chi_squares.max() + stego_chi_squares.min()) / 2:.1f}")
