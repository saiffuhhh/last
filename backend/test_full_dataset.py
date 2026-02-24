from algorithms.detector import SteganographyDetector
import cv2

detector = SteganographyDetector()

# Test on 100 images
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
        
        cover_correct = not result_cover["is_stego"]
        stego_correct = result_stego["is_stego"]
        
        if cover_correct and stego_correct:
            correct += 1
        total += 1

print(f"Detection Accuracy on 100 images: {correct}/{total} = {100*correct/total:.1f}%")

# Show some examples
print("\nExamples:")
print("Image | Cover Correct? | Stego Correct? | Pair Var (C) | Pair Var (S)")
print("-" * 70)
for i in [1, 5, 10, 20, 50, 100]:
    cover = cv2.imread(f'../dataset/bossbase_raw/{i}.pgm', cv2.IMREAD_GRAYSCALE)
    stego = cv2.imread(f'../dataset/stego/{i}.png', cv2.IMREAD_GRAYSCALE)
    
    result_c = detector.detect(cover)
    result_s = detector.detect(stego)
    
    cover_ok = not result_c["is_stego"]
    stego_ok = result_s["is_stego"]
    
    print(f"{i:5d} | {str(cover_ok):14} | {str(stego_ok):14} | {result_c['pair_variance']:11.1f} | {result_s['pair_variance']:11.1f}")
