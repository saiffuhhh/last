import numpy as np
from algorithms.lsb import LSBSteganography
from algorithms.dct import DCTSteganography
from algorithms.dwt import DWTSteganography
from algorithms.pvd import PVDSteganography
from algorithms.edge_adaptive import EdgeAdaptiveSteganography

# Create a test image
test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Test message
message = "Hello World! This is a test message for steganography."

algorithms = [
    ("LSB", LSBSteganography()),
    ("DCT", DCTSteganography()),
    ("DWT", DWTSteganography()),
    ("PVD", PVDSteganography()),
    ("Edge-Adaptive", EdgeAdaptiveSteganography())
]

for name, algo in algorithms:
    try:
        print(f"\nTesting {name}...")
        # Encode
        stego = algo.encode(test_image, message)
        print(f"  Encoding successful for {name}")

        # Decode
        decoded = algo.decode(stego)
        print(f"  Decoding successful for {name}")

        # Check
        if decoded == message:
            print(f"  ✓ {name} PASSED")
        else:
            print(f"  ✗ {name} FAILED: Expected '{message}', got '{decoded}'")

    except Exception as e:
        print(f"  ✗ {name} ERROR: {e}")
        # Try to fix or report