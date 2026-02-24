from algorithms.detector import SteganographyDetector
import cv2
import numpy as np

detector = SteganographyDetector()

# Test on 100 images
print("Image | Cover Pair-V | Stego Pair-V | Cover > Stego?")
print("-" * 60)

correct = 0
total = 0

for i in range(1, 101):
    cover_path = f'../dataset/bossbase_raw/{i}.pgm'
    cover = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    
    stego_path = f'../dataset/stego/{i}.png'
    stego = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
    
    if cover is not None and stego is not None:
        result_cover = detector.detect(cover)
        result_stego = detector.detect(stego)
        
        cover_pv = result_cover["pair_variance"]
        stego_pv = result_stego["pair_variance"]
        is_correct = cover_pv > stego_pv
        
        if i <= 10 or i % 10 == 0:
            print(f"{i:3d} | {cover_pv:11.1f} | {stego_pv:12.1f} | {is_correct}")
        
        if is_correct:
            correct += 1
        total += 1

print(f"\nAccuracy: {correct}/{total} = {100*correct/total:.1f}%")

# Now test individual classification
print("\n" + "="*60)
print("Classification Test (single image basis):")
print("="*60)
print("Image | Cover is_stego | Stego is_stego | Correct?")
print("-" * 60)

correct_classification = 0

for i in range(1, 31):
    cover_path = f'../dataset/bossbase_raw/{i}.pgm'
    cover = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    
    stego_path = f'../dataset/stego/{i}.png'
    stego = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
    
    if cover is not None and stego is not None:
        result_cover = detector.detect(cover)
        result_stego = detector.detect(stego)
        
        cover_correct = not result_cover["is_stego"]
        stego_correct = result_stego["is_stego"]
        both_correct = cover_correct and stego_correct
        
        print(f"{i:3d} | {str(result_cover['is_stego']):14} | {str(result_stego['is_stego']):14} | {both_correct}")
        
        if both_correct:
            correct_classification += 1

print(f"\nClassification Accuracy: {correct_classification}/30 = {100*correct_classification/30:.1f}%")
