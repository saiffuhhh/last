from algorithms.detector import SteganographyDetector
import cv2

detector = SteganographyDetector()

# Test on multiple images
for i in [1, 10, 50, 100, 500]:
    # Load cover image
    cover_path = f'../dataset/bossbase_raw/{i}.pgm'
    cover = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    
    # Load stego image
    stego_path = f'../dataset/stego/{i}.png'
    stego = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
    
    if cover is not None and stego is not None:
        result_cover = detector.detect(cover)
        result_stego = detector.detect(stego)
        
        print(f'Image {i}:')
        print(f'  Cover: chi_square={result_cover["chi_square"]:.1f}, rs={result_cover["rs_analysis"]:.1f}, is_stego={result_cover["is_stego"]}')
        print(f'  Stego: chi_square={result_stego["chi_square"]:.1f}, rs={result_stego["rs_analysis"]:.1f}, is_stego={result_stego["is_stego"]}')
        print()
