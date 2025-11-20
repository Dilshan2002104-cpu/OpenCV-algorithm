"""
Circuit Symbol Detection Module
Detects various circuit components using OpenCV
"""

import cv2
import numpy as np
from typing import List, Tuple
import os

class CircuitSymbolDetector:
    def __init__(self):
        self.templates = {}
        self.load_templates()
    
    def load_templates(self):
        """Load template images for different circuit symbols"""
        # Note: In a real implementation, you would load actual template images
        # For now, we'll create simple synthetic templates
        
        # Create basic resistor template (zigzag pattern)
        resistor = np.zeros((20, 60), dtype=np.uint8)
        cv2.rectangle(resistor, (10, 8), (50, 12), 255, -1)
        self.templates['resistor'] = resistor
        
        # Create basic capacitor template (two parallel lines)
        capacitor = np.zeros((30, 20), dtype=np.uint8)
        cv2.line(capacitor, (8, 5), (8, 25), 255, 2)
        cv2.line(capacitor, (12, 5), (12, 25), 255, 2)
        self.templates['capacitor'] = capacitor
        
        # Create basic ground symbol template
        ground = np.zeros((25, 25), dtype=np.uint8)
        cv2.line(ground, (12, 5), (12, 15), 255, 2)
        cv2.line(ground, (8, 15), (16, 15), 255, 2)
        cv2.line(ground, (10, 18), (14, 18), 255, 2)
        cv2.line(ground, (11, 21), (13, 21), 255, 2)
        self.templates['ground'] = ground
    
    def detect_by_template_matching(self, image: np.ndarray, symbol_type: str, threshold: float = 0.7) -> List[Tuple[int, int, int, int]]:
        """
        Detect symbols using template matching
        
        Args:
            image: Input circuit image
            symbol_type: Type of symbol to detect ('resistor', 'capacitor', etc.)
            threshold: Matching threshold (0.0 to 1.0)
        
        Returns:
            List of bounding boxes (x, y, width, height)
        """
        if symbol_type not in self.templates:
            return []
        
        template = self.templates[symbol_type]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Multi-scale template matching to handle different sizes
        found_symbols = []
        scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        
        for scale in scales:
            # Resize template
            width = int(template.shape[1] * scale)
            height = int(template.shape[0] * scale)
            
            if width < 10 or height < 10 or width > gray.shape[1] or height > gray.shape[0]:
                continue
                
            resized_template = cv2.resize(template, (width, height))
            
            # Template matching
            result = cv2.matchTemplate(gray, resized_template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            
            for pt in zip(*locations[::-1]):
                x, y = pt
                # Check for overlapping detections
                overlap = False
                for existing in found_symbols:
                    if self._boxes_overlap((x, y, width, height), existing):
                        overlap = True
                        break
                
                if not overlap:
                    found_symbols.append((x, y, width, height))
        
        return found_symbols
    
    def detect_by_contours(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect symbols using contour analysis
        
        Args:
            image: Input circuit image
        
        Returns:
            List of bounding boxes (x, y, width, height)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Adaptive thresholding for better results
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        symbols = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (adjust based on your images)
            if 50 < area < 5000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio and size
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5.0 and w > 10 and h > 10:
                    symbols.append((x, y, w, h))
        
        return symbols
    
    def detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions in the circuit diagram
        
        Args:
            image: Input circuit image
        
        Returns:
            List of text bounding boxes (x, y, width, height)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Use MSER (Maximally Stable Extremal Regions) for text detection
        mser = cv2.MSER_create(
            _delta=5,
            _min_area=20,
            _max_area=2000,
            _max_variation=0.25,
            _min_diversity=0.2,
            _max_evolution=200,
            _area_threshold=1.01,
            _min_margin=0.003,
            _edge_blur_size=5
        )
        
        regions, _ = mser.detectRegions(gray)
        
        text_boxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            
            # Filter text-like regions
            aspect_ratio = w / h if h > 0 else 0
            if 0.1 < aspect_ratio < 10.0 and w > 5 and h > 5:
                text_boxes.append((x, y, w, h))
        
        return text_boxes
    
    def detect_all_symbols(self, image: np.ndarray) -> dict:
        """
        Detect all types of symbols in the image
        
        Args:
            image: Input circuit image
        
        Returns:
            Dictionary with symbol types as keys and bounding boxes as values
        """
        all_symbols = {}
        
        # Template-based detection
        for symbol_type in self.templates.keys():
            symbols = self.detect_by_template_matching(image, symbol_type)
            if symbols:
                all_symbols[symbol_type] = symbols
        
        # Contour-based detection for general components
        contour_symbols = self.detect_by_contours(image)
        if contour_symbols:
            all_symbols['general_components'] = contour_symbols
        
        # Text detection
        text_regions = self.detect_text_regions(image)
        if text_regions:
            all_symbols['text'] = text_regions
        
        return all_symbols
    
    def _boxes_overlap(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> bool:
        """Check if two bounding boxes overlap"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
    
    def visualize_detections(self, image: np.ndarray, symbols_dict: dict) -> np.ndarray:
        """
        Draw detected symbols on the image for visualization
        
        Args:
            image: Original image
            symbols_dict: Dictionary of detected symbols
        
        Returns:
            Image with drawn bounding boxes
        """
        result = image.copy()
        colors = {
            'resistor': (255, 0, 0),      # Red
            'capacitor': (0, 255, 0),     # Green
            'ground': (0, 0, 255),        # Blue
            'general_components': (255, 255, 0),  # Yellow
            'text': (255, 0, 255)         # Magenta
        }
        
        for symbol_type, symbols in symbols_dict.items():
            color = colors.get(symbol_type, (128, 128, 128))
            
            for x, y, w, h in symbols:
                cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
                cv2.putText(result, symbol_type, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result

# Example usage
if __name__ == "__main__":
    detector = CircuitSymbolDetector()
    
    # Test with a sample image
    # image = cv2.imread('circuit_diagram.png')
    # symbols = detector.detect_all_symbols(image)
    # result = detector.visualize_detections(image, symbols)
    # cv2.imshow('Detected Symbols', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()