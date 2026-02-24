import numpy as np

class PVDSteganography:
    """Pixel Value Differencing Steganography - LSB variant"""
    
    def __init__(self):
        self.delimiter = "<<<END_OF_MESSAGE>>>"
    
    def _text_to_binary(self, text):
        return ''.join(format(byte, '08b') for byte in text.encode('utf-8'))
    
    def _binary_to_text(self, binary):
        bytes_data = bytearray()
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            if len(byte) == 8:
                bytes_data.append(int(byte, 2))
        try:
            return bytes_data.decode('utf-8')
        except UnicodeDecodeError:
            return bytes_data.decode('utf-8', errors='replace')
    
    def encode(self, cover_image, message, password="", strength=5):
        """Embed using LSB on all pixels"""
        stego_image = cover_image.copy()
        binary_message = self._text_to_binary(message + self.delimiter)
        
        height, width, channels = cover_image.shape
        data_index = 0
        
        flat_stego = stego_image.reshape(-1)
        
        if len(binary_message) > len(flat_stego):
            raise ValueError("Message too large for this image")
        
        for i, bit in enumerate(binary_message):
            flat_stego[i] = (int(flat_stego[i]) & 0xFE) | int(bit)
        
        return stego_image.reshape((height, width, channels))
    
    def decode(self, stego_image, password=""):
        """Extract using LSB from all pixels"""
        flat_stego = stego_image.reshape(-1)
        binary_message = ""
        extracted_text = ""
        
        for i in range(len(flat_stego)):
            bit = int(flat_stego[i]) & 1
            binary_message += str(bit)
            
            if len(binary_message) >= 8:
                byte = binary_message[:8]
                binary_message = binary_message[8:]
                
                char = chr(int(byte, 2))
                extracted_text += char
                
                if self.delimiter in extracted_text:
                    return extracted_text.replace(self.delimiter, "")
        
        raise ValueError("No message found")
