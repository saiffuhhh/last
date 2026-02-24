from algorithms.detector import SteganographyDetector
import cv2
import numpy as np

detector = SteganographyDetector()

# Test on 100 images to find optimal threshold
cover_pv = []
stego_pv = []

for i in range(1, 101):
    cover_path = f'../dataset/bossbase_raw/{i}.pgm'
    cover = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    
    stego_path = f'../dataset/stego/{i}.png'
    stego = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
    
    if cover is not None and stego is not None:
        result_cover = detector.detect(cover)
        result_stego = detector.detect(stego)
        
        cover_pv.append(result_cover["pair_variance"])
        stego_pv.append(result_stego["pair_variance"])

cover_pv = np.array(cover_pv)
stego_pv = np.array(stego_pv)

print("Cover Pair Variance Statistics:")
print(f"  Min: {cover_pv.min():.1f}, Max: {cover_pv.max():.1f}")
print(f"  Mean: {cover_pv.mean():.1f}, Median: {np.median(cover_pv):.1f}, Std: {cover_pv.std():.1f}")
print(f"  Q1: {np.percentile(cover_pv, 25):.1f}, Q3: {np.percentile(cover_pv, 75):.1f}")

print("\nStego Pair Variance Statistics:")
print(f"  Min: {stego_pv.min():.1f}, Max: {stego_pv.max():.1f}")
print(f"  Mean: {stego_pv.mean():.1f}, Median: {np.median(stego_pv):.1f}, Std: {stego_pv.std():.1f}")
print(f"  Q1: {np.percentile(stego_pv, 25):.1f}, Q3: {np.percentile(stego_pv, 75):.1f}")

# Find optimal threshold by testing different values
print("\n" + "="*60)
print("Testing different thresholds:")
print("="*60)
print("Threshold | Cover as Stego | Stego as Cover | Accuracy")
print("-" * 60)

best_threshold = 0
best_accuracy = 0

for thresh in [30, 40, 50, 60, 70, 80, 90, 100]:
    cover_wrong = np.sum(cover_pv < thresh)
    stego_wrong = np.sum(stego_pv >= thresh)
    accuracy = (100 - cover_wrong + 100 - stego_wrong) / 2
    print(f"{thresh:9d} | {cover_wrong:14d} | {stego_wrong:14d} | {accuracy:.1f}%")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = thresh

print(f"\nBest threshold: {best_threshold} with accuracy {best_accuracy:.1f}%")
