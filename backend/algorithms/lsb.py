# LSB Steganography Implementation
# File: backend/algorithms/lsb.py

import numpy as np
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes
import hashlib

class LSBSteganography:
    """
    Least Significant Bit (LSB) Steganography
    Hides data in the least significant bits of pixel values
    """
    
    def __init__(self):
        self.delimiter = "<<<END_OF_MESSAGE>>>"
        
    def _text_to_binary(self, text):
        """Convert text to binary string"""
        return ''.join(format(byte, '08b') for byte in text.encode('utf-8'))
    
    def _binary_to_text(self, binary):
        """Convert binary string to text"""
        bytes_data = bytearray()
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            if len(byte) == 8:
                bytes_data.append(int(byte, 2))
        try:
            return bytes_data.decode('utf-8')
        except UnicodeDecodeError:
            return bytes_data.decode('utf-8', errors='ignore')
    
    def _encrypt_message(self, message, password):
        """Encrypt message using AES-256"""
        if not password:
            return message
        
        # Generate key from password
        salt = get_random_bytes(16)
        key = PBKDF2(password, salt, dkLen=32)
        
        # Encrypt
        cipher = AES.new(key, AES.MODE_EAX)
        ciphertext, tag = cipher.encrypt_and_digest(message.encode())
        
        # Combine salt + nonce + tag + ciphertext
        encrypted = salt + cipher.nonce + tag + ciphertext
        return encrypted.hex()
    
    def _decrypt_message(self, encrypted_hex, password):
        """Decrypt AES-256 encrypted message"""
        if not password:
            return encrypted_hex
        
        try:
            encrypted = bytes.fromhex(encrypted_hex)
            
            # Extract components
            salt = encrypted[:16]
            nonce = encrypted[16:32]
            tag = encrypted[32:48]
            ciphertext = encrypted[48:]
            
            # Generate key from password
            key = PBKDF2(password, salt, dkLen=32)
            
            # Decrypt
            cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
            plaintext = cipher.decrypt_and_verify(ciphertext, tag)
            
            return plaintext.decode()
        except Exception as e:
            raise ValueError("Decryption failed. Wrong password?")
    
    def encode(self, cover_image, message, password="", strength=5):
        """
        Encode message into cover image using LSB
        
        Args:
            cover_image: numpy array of cover image (H, W, 3)
            message: string message to hide
            password: encryption password (optional)
            strength: embedding strength (1-10), higher = more bits used
            
        Returns:
            stego_image: numpy array of stego image
        """
        # Encrypt message if password provided
        if password:
            message = self._encrypt_message(message, password)
        
        # Add delimiter
        message_with_delimiter = message + self.delimiter
        
        # Convert to binary
        binary_message = self._text_to_binary(message_with_delimiter)
        
        # Check capacity
        height, width, channels = cover_image.shape
        max_bits = height * width * channels
        
        if len(binary_message) > max_bits:
            raise ValueError(f"Message too large. Max capacity: {max_bits // 8} bytes")
        
        # Create copy of image
        stego_image = cover_image.copy()
        
        # Embed message
        data_index = 0
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    if data_index < len(binary_message):
                        # Get pixel value
                        pixel = stego_image[i, j, k]
                        
                        # Replace LSB
                        pixel = (pixel & 0xFE) | int(binary_message[data_index])
                        stego_image[i, j, k] = pixel
                        
                        data_index += 1
                    else:
                        return stego_image
        
        return stego_image
    
    def decode(self, stego_image, password=""):
        """
        Decode message from stego image
        
        Args:
            stego_image: numpy array of stego image (H, W, 3)
            password: decryption password (optional)
            
        Returns:
            message: extracted message string
        """
        height, width, channels = stego_image.shape
        
        # Extract bits
        binary_message = ""
        
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    # Extract LSB
                    pixel = stego_image[i, j, k]
                    binary_message += str(pixel & 1)
        
        # Convert to text
        message = self._binary_to_text(binary_message)
        
        # Find delimiter
        if self.delimiter in message:
            message = message.split(self.delimiter)[0]
        else:
            raise ValueError("No valid message found in image")
        
        # Decrypt if password provided
        if password:
            message = self._decrypt_message(message, password)
        
        return message
    
    def estimate_capacity(self, image_shape):
        """
        Estimate maximum message capacity in bytes
        
        Args:
            image_shape: tuple (height, width, channels)
            
        Returns:
            capacity_bytes: maximum capacity in bytes
        """
        height, width, channels = image_shape
        total_bits = height * width * channels
        capacity_bytes = total_bits // 8
        
        # Account for delimiter
        delimiter_bytes = len(self.delimiter)
        
        return capacity_bytes - delimiter_bytes - 100  # Safety margin

if __name__ == '__main__':
    # Test LSB steganography
    print("Testing LSB Steganography...")
    
    # Create test image
    test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # Test message
    secret_message = "This is a secret message hidden using LSB steganography!"
    password = "test_password_123"
    
    # Initialize
    lsb = LSBSteganography()
    
    # Encode
    print(f"\nOriginal message: {secret_message}")
    print(f"Password: {password}")
    
    stego_image = lsb.encode(test_image, secret_message, password)
    print(f"Message encoded successfully!")
    
    # Check difference
    diff = np.abs(test_image.astype(int) - stego_image.astype(int))
    print(f"Max pixel difference: {diff.max()}")
    print(f"Mean pixel difference: {diff.mean():.4f}")
    
    # Decode
    decoded_message = lsb.decode(stego_image, password)
    print(f"\nDecoded message: {decoded_message}")
    
    # Verify
    if secret_message == decoded_message:
        print("✓ Test passed! Messages match.")
    else:
        print("✗ Test failed! Messages don't match.")
    
    # Test capacity
    capacity = lsb.estimate_capacity(test_image.shape)
    print(f"\nMax capacity: {capacity} bytes ({capacity / 1024:.2f} KB)")
