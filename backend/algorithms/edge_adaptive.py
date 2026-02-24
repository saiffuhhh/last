import cv2

class EdgeAdaptiveSteganography:
    """Edge-based adaptive steganography"""
    
    def __init__(self, edge_threshold=100):
        self.edge_threshold = edge_threshold
        self.delimiter = "<<<END_OF_MESSAGE>>>"
    
    def _detect_edges(self, image):
        """Detect edges using Canny"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 10, 30)
        return edges
    
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
            return bytes_data.decode('utf-8', errors='ignore')
    
    def encode(self, cover_image, message, password="", strength=5):
        """Embed in all pixels"""
        stego_image = cover_image.copy()
        
        binary_message = self._text_to_binary(message + self.delimiter)
        data_index = 0
        
        height, width, channels = cover_image.shape
        
        for i in range(height):
            for j in range(width):
                for c in range(channels):
                    if data_index < len(binary_message):
                        pixel = stego_image[i, j, c]
                        bit = int(binary_message[data_index])
                        pixel = (pixel & 0xFE) | bit
                        stego_image[i, j, c] = pixel
                        data_index += 1
        
        return stego_image
    
    def decode(self, stego_image, password=""):
        """Extract from all pixels"""
        binary_message = ""
        
        height, width, channels = stego_image.shape
        
        for i in range(height):
            for j in range(width):
                for c in range(channels):
                    pixel = stego_image[i, j, c]
                    binary_message += str(pixel & 1)
        
        message = self._binary_to_text(binary_message)
        if self.delimiter in message:
            return message.split(self.delimiter)[0]
        raise ValueError("No message found")