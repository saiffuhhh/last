# backend/random.py

import os
import cv2
import random
import string

from algorithms.lsb import LSBSteganography   # adjust if module name differs


# ====== PATHS ======
# Change ONLY if your project root is not E:\last
COVER_DIR = r"E:\last\dataset\cover"   # resized cover images
STEGO_DIR = r"E:\last\dataset\stego"   # stego images will be saved here
os.makedirs(STEGO_DIR, exist_ok=True)


stego_algo = LSBSteganography()


def random_message(min_len: int = 20, max_len: int = 80) -> str:
    """Generate a random ASCII secret message."""
    length = random.randint(min_len, max_len)
    chars = string.ascii_letters + string.digits + " .,;!?-_:"
    return "".join(random.choice(chars) for _ in range(length))


def main():
    # Accept common image extensions
    files = [
        f for f in os.listdir(COVER_DIR)
        if f.lower().endswith((".pgm", ".png", ".jpg", ".jpeg", ".bmp"))
    ]

    if not files:
        print("No images found in", COVER_DIR)
        return

    print(f"Found {len(files)} cover images in {COVER_DIR}")

    for fname in files:
        in_path = os.path.join(COVER_DIR, fname)

        # Read as COLOR (H, W, 3) because LSB implementation expects 3 channels
        img = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img is None:
            print("Failed to read:", in_path)
            continue

        msg = random_message()

        # Encode with LSB
        try:
            stego_img = stego_algo.encode(img, msg)
            if stego_img is None or not hasattr(stego_img, "shape") or stego_img.size == 0:
                print(f"Invalid stego for {fname}, skipping.")
                continue
        except Exception as e:
            print(f"Error encoding {fname}: {e}")
            continue

        # Save stego as PNG (avoid PGM grayscale issue)
        name_root, _ = os.path.splitext(fname)
        out_path = os.path.join(STEGO_DIR, name_root + ".png")

        ok = cv2.imwrite(out_path, stego_img)
        if not ok:
            print("Failed to write:", out_path)
        else:
            print("Saved stego:", out_path)

    print("\nStego generation complete.")


if __name__ == "__main__":
    main()
