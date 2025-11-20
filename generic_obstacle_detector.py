"""
Generic Obstacle Detection for Circuit Diagrams
Uses binarization and dilation to detect ALL components (not just specific symbols)
Works with hand-drawn, black & white circuit diagrams
"""

import cv2
import numpy as np
from typing import Tuple

class GenericObstacleDetector:
    """
    Detects obstacles using generic computer vision techniques
    Does NOT rely on knowing what specific components look like
    """
    
    def __init__(self):
        self.obstacle_map = None
        self.original_binary = None
    
    def create_obstacle_map(self, image: np.ndarray, 
                           dilation_size: int = 5,
                           safety_padding: int = 3) -> np.ndarray:
        """
        Create obstacle map using binarization + dilation technique
        
        This is the technique Gemini AI described:
        1. Binarization (Thresholding) - Convert to pure black/white
        2. Morphological Dilation - "Fatten" all lines to close gaps
        3. Safety Padding - Add extra margin around obstacles
        
        Args:
            image: Input circuit image (RGB or grayscale)
            dilation_size: How much to "thicken" the lines (higher = thicker blobs)
            safety_padding: Extra pixels of clearance around obstacles
        
        Returns:
            Binary obstacle map (255 = obstacle, 0 = free space)
        """
        
        # Step 1: Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 2: BINARIZATION (Thresholding)
        # For black & white circuit diagrams, use adaptive thresholding
        # This handles varying lighting/contrast better than simple threshold
        binary = cv2.adaptiveThreshold(
            gray, 
            255,                                    # Max value (white)
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,        # Adaptive method
            cv2.THRESH_BINARY_INV,                 # Invert: black lines → white
            blockSize=11,                           # Neighborhood size
            C=2                                     # Constant subtracted
        )
        
        # Alternative: Simple threshold (works if image has uniform lighting)
        # _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        self.original_binary = binary.copy()
        
        # Step 3: MORPHOLOGICAL DILATION (The "Fat Marker" Technique)
        # This is THE KEY STEP that makes wires avoid components!
        # 
        # What it does:
        # - Thin zigzag resistor line → Thick solid blob
        # - Gaps inside components → Filled up
        # - Pathfinding sees solid wall instead of maze
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (dilation_size, dilation_size)
        )
        
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Step 4: SAFETY PADDING (Inflation)
        # Add extra margin so wires don't touch components visually
        if safety_padding > 0:
            padding_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (safety_padding * 2 + 1, safety_padding * 2 + 1)
            )
            dilated = cv2.dilate(dilated, padding_kernel, iterations=1)
        
        self.obstacle_map = dilated
        return dilated
    
    def create_advanced_obstacle_map(self, image: np.ndarray,
                                    remove_thin_lines: bool = True,
                                    min_component_area: int = 100) -> np.ndarray:
        """
        Advanced obstacle detection with noise filtering
        
        Additional features:
        - Removes very thin lines (likely existing wires, not components)
        - Filters out small noise/dots
        - Keeps only substantial components
        
        Args:
            image: Input circuit image
            remove_thin_lines: If True, tries to remove existing thin wires
            min_component_area: Minimum size to be considered a component
        
        Returns:
            Refined obstacle map
        """
        
        # Start with basic obstacle map
        obstacle_map = self.create_obstacle_map(image, dilation_size=3, safety_padding=2)
        
        if remove_thin_lines:
            # Use morphological opening to remove thin lines
            # This removes wires but keeps thicker components
            line_removal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            obstacle_map = cv2.morphologyEx(
                obstacle_map, 
                cv2.MORPH_OPEN, 
                line_removal_kernel,
                iterations=1
            )
        
        if min_component_area > 0:
            # Remove small blobs (noise)
            contours, _ = cv2.findContours(
                obstacle_map, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Create clean map
            clean_map = np.zeros_like(obstacle_map)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_component_area:
                    cv2.drawContours(clean_map, [contour], -1, 255, -1)
            
            obstacle_map = clean_map
        
        # Apply safety padding again after filtering
        padding_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        obstacle_map = cv2.dilate(obstacle_map, padding_kernel, iterations=1)
        
        self.obstacle_map = obstacle_map
        return obstacle_map
    
    def visualize_obstacle_map(self, original_image: np.ndarray) -> np.ndarray:
        """
        Overlay obstacle map on original image for debugging
        
        Args:
            original_image: Original circuit image
        
        Returns:
            Image with obstacle overlay (obstacles shown in red)
        """
        if self.obstacle_map is None:
            return original_image
        
        # Convert to color if needed
        if len(original_image.shape) == 2:
            result = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            result = original_image.copy()
        
        # Create red overlay for obstacles
        red_overlay = np.zeros_like(result)
        red_overlay[:, :, 2] = self.obstacle_map  # Red channel
        
        # Blend with original
        result = cv2.addWeighted(result, 0.7, red_overlay, 0.3, 0)
        
        return result
    
    def get_free_space_map(self) -> np.ndarray:
        """
        Get inverse of obstacle map (where wires CAN go)
        
        Returns:
            Binary map where 255 = free space, 0 = obstacle
        """
        if self.obstacle_map is None:
            raise ValueError("No obstacle map created yet. Call create_obstacle_map() first.")
        
        return cv2.bitwise_not(self.obstacle_map)
    
    def check_path_clearance(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """
        Check if a straight line between two points is obstacle-free
        
        Args:
            start: Starting point (x, y)
            end: End point (x, y)
        
        Returns:
            True if path is clear, False if it hits obstacles
        """
        if self.obstacle_map is None:
            return True
        
        # Use Bresenham's line algorithm to check all pixels along the line
        x1, y1 = start
        x2, y2 = end
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x_step = 1 if x1 < x2 else -1
        y_step = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        
        while True:
            # Check if current pixel is an obstacle
            if (0 <= y < self.obstacle_map.shape[0] and 
                0 <= x < self.obstacle_map.shape[1]):
                if self.obstacle_map[y, x] == 255:
                    return False  # Hit obstacle
            
            if x == x2 and y == y2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += x_step
            if e2 < dx:
                err += dx
                y += y_step
        
        return True  # Path is clear


# Example usage and testing
if __name__ == "__main__":
    # Test with a sample circuit image
    detector = GenericObstacleDetector()
    
    # Load your circuit diagram
    image = cv2.imread('c3140abe-a115-4cdf-a27d-07953f06ec65.jpg')
    
    if image is not None:
        # Create obstacle map
        obstacle_map = detector.create_obstacle_map(
            image, 
            dilation_size=7,      # Adjust this: higher = thicker blobs
            safety_padding=5       # Adjust this: higher = more clearance
        )
        
        # Visualize
        overlay = detector.visualize_obstacle_map(image)
        
        # Save results
        cv2.imwrite('obstacle_map.png', obstacle_map)
        cv2.imwrite('obstacle_overlay.png', overlay)
        
        print("✓ Obstacle map created successfully!")
        print(f"  Obstacle pixels: {np.count_nonzero(obstacle_map)}")
        print(f"  Free space pixels: {np.count_nonzero(obstacle_map == 0)}")
        print("  Saved: obstacle_map.png, obstacle_overlay.png")
        
        # Display
        cv2.imshow('Original', image)
        cv2.imshow('Obstacle Map (White = Obstacle)', obstacle_map)
        cv2.imshow('Overlay (Red = Obstacle)', overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Could not load image")
