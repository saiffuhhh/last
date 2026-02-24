import numpy as np
from algorithms.pvd import PVDSteganography

# Create a test image
test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Test message
message = "Hello"

pvd = PVDSteganography()

# Encode
stego = pvd.encode(test_image, message)
print("Encoding successful")
print(f"Image shape: {stego.shape}")
print(f"Message length: {len(message)}")
print(f"Delimiter: {pvd.delimiter}")

# Check if data is actually embedded
flat_stego = stego.reshape(-1)
embedded_bits = []
for i in range(0, min(50, len(flat_stego))):
    embedded_bits.append(flat_stego[i] & 1)
print(f"First 50 LSBs: {embedded_bits}")

# Try decode
try:
    decoded = pvd.decode(stego)
    print(f"Decoded: {decoded}")
except Exception as e:
    print(f"Decode error: {e}")
