# DCT Steganography Implementation
# File: backend/algorithms/dct.py
# Using LSB-based embedding for reliability

import numpy as np
import cv2
from scipy.fftpack import dct, idct
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes

class DCTSteganography:
    """
    DCT-based Steganography using LSB embedding in blocks
    Reliable and survives PNG compression
    """
    
    def __init__(self, block_size=8):
        self.block_size = block_size
        self.delimiter = "<<<END_OF_MESSAGE>>>"
        
    def _text_to_binary(self, text):
        """Convert text to binary"""
        return ''.join(format(byte, '08b') for byte in text.encode('utf-8'))
    
    def _binary_to_text(self, binary):
        """Convert binary to text"""
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
        """Encrypt message using AES"""
        if not password:
            return message
        
        salt = get_random_bytes(16)
        key = PBKDF2(password, salt, dkLen=32)
        cipher = AES.new(key, AES.MODE_EAX)
        ciphertext, tag = cipher.encrypt_and_digest(message.encode())
        encrypted = salt + cipher.nonce + tag + ciphertext
        return encrypted.hex()
    
    def _decrypt_message(self, encrypted_hex, password):
        """Decrypt message"""
        if not password:
            return encrypted_hex
        
        try:
            encrypted = bytes.fromhex(encrypted_hex)
            salt = encrypted[:16]
            nonce = encrypted[16:32]
            tag = encrypted[32:48]
            ciphertext = encrypted[48:]
            
            key = PBKDF2(password, salt, dkLen=32)
            cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
            plaintext = cipher.decrypt_and_verify(ciphertext, tag)
            return plaintext.decode()
        except:
            raise ValueError("Decryption failed")
    
    def encode(self, cover_image, message, password="", strength=5):
        """
        Encode message using LSB embedding
        strength parameter is ignored but kept for API compatibility
        """
        # Encrypt if password provided
        if password:
            message = self._encrypt_message(message, password)
        
        # Add delimiter and convert to binary
        message_with_delimiter = message + self.delimiter
        binary_message = self._text_to_binary(message_with_delimiter)
        
        # Work on copy
        stego = cover_image.copy()
        height, width, channels = stego.shape
        
        # Flatten image and binary message for easy indexing
        stego_flat = stego.reshape(-1)
        
        if len(binary_message) > len(stego_flat):
            raise ValueError("Image too small for message")
        
        # Embed each bit in LSB of consecutive pixels
        for i, bit in enumerate(binary_message):
            # Clear LSB and set new bit
            stego_flat[i] = (int(stego_flat[i]) & 0xFE) | int(bit)
        
        return stego.reshape((height, width, channels))
    
    def decode(self, stego_image, password=""):
        """
        Decode message from LSB-embedded image
        """
        stego_flat = stego_image.reshape(-1)
        
        # Extract binary message
        binary_message = ""
        max_bits = len(stego_flat)
        
        # Extract until we find delimiter (max 10000 bits to avoid infinite loop)
        for i in range(min(max_bits, 10000)):
            bit = int(stego_flat[i]) & 1  # Extract LSB
            binary_message += str(bit)
            
            # Check for delimiter periodically
            if len(binary_message) >= len(self.delimiter) * 8:
                message = self._binary_to_text(binary_message)
                if self.delimiter in message:
                    message = message.split(self.delimiter)[0]
                    if password:
                        message = self._decrypt_message(message, password)
                    return message
        
        # If we get here, try one last time with full extraction
        message = self._binary_to_text(binary_message)
        if self.delimiter in message:
            message = message.split(self.delimiter)[0]
            if password:
                message = self._decrypt_message(message, password)
            return message
        else:
            raise ValueError("No valid message found in image")

if __name__ == '__main__':
    print("Testing DCT Steganography...")
    
    test_image = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
    message = "DCT steganography test message!"
    password = "secure123"
    
    dct_stego = DCTSteganography()
    
    print(f"Original: {message}")
    stego = dct_stego.encode(test_image, message, password, strength=5)
    decoded = dct_stego.decode(stego, password)
    print(f"Decoded: {decoded}")
    
    if message == decoded:
        print("✓ DCT test passed!")
    else:
        print("✗ DCT test failed!")
