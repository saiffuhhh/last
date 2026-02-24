from algorithms.detector import SteganographyDetector
import cv2
import numpy as np

detector = SteganographyDetector()

# Test on more images to find clear separation
print("Image | Cover Chi-Sq | Stego Chi-Sq | Cover > Stego?")
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
        
        cover_chi = result_cover["chi_square"]
        stego_chi = result_stego["chi_square"]
        is_correct = cover_chi > stego_chi
        
        if i <= 10 or i % 10 == 0:
            print(f"{i:3d} | {cover_chi:11.1f} | {stego_chi:12.1f} | {is_correct}")
        
        if is_correct:
            correct += 1
        total += 1

print(f"\nAccuracy: {correct}/{total} = {100*correct/total:.1f}%")
