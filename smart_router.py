"""
Smart Wire Router
Combines obstacle detection and pathfinding for intelligent circuit wire routing
Now uses GENERIC OBSTACLE DETECTION (works with hand-drawn circuits!)
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from generic_obstacle_detector import GenericObstacleDetector
from pathfinding import PathFinder, ObstacleMapGenerator

class SmartWireRouter:
    def __init__(self, use_generic_detection: bool = True):
        """
        Initialize smart wire router
        
        Args:
            use_generic_detection: Uses generic obstacle detection (works with hand-drawn circuits)
        """
        self.use_generic_detection = use_generic_detection
        self.obstacle_detector = GenericObstacleDetector()
        self.current_obstacle_map = None
        self.current_symbols = {}
        
    def analyze_circuit(self, image: np.ndarray, 
                       dilation_size: int = 7,
                       safety_padding: int = 5) -> dict:
        """
        Analyze circuit image and create obstacle map
        
        NOW USES GENERIC DETECTION (Gemini AI's approach):
        1. Binarization - Convert to pure black/white
        2. Dilation - "Fatten" lines to close gaps in components
        3. Safety padding - Add clearance margin
        
        Args:
            image: Circuit diagram image
            dilation_size: How much to thicken lines (higher = thicker blobs)
            safety_padding: Extra clearance around obstacles (pixels)
        
        Returns:
            Dictionary containing obstacle map and detection info
        """
        
        if self.use_generic_detection:
            # NEW APPROACH: Generic obstacle mapping (works with hand-drawn!)
            print("Using Generic Obstacle Detection (Gemini AI technique)")
            
            self.current_obstacle_map = self.obstacle_detector.create_obstacle_map(
                image,
                dilation_size=dilation_size,
                safety_padding=safety_padding
            )
            
            # Count obstacle pixels
            obstacle_pixels = np.count_nonzero(self.current_obstacle_map)
            free_pixels = np.count_nonzero(self.current_obstacle_map == 0)
            
            return {
                'method': 'generic',
                'obstacle_map': self.current_obstacle_map,
                'obstacle_pixels': int(obstacle_pixels),
                'free_pixels': int(free_pixels),
                'coverage': round(obstacle_pixels / (obstacle_pixels + free_pixels) * 100, 2)
            }
        else:
            # OLD APPROACH: Symbol detection (only for perfect schematics)
            print("Using Symbol Detection (traditional approach)")
            
            self.current_symbols = self.symbol_detector.detect_all_symbols(image)
            
            # Create obstacle map from detected symbols
            self.current_obstacle_map = ObstacleMapGenerator.create_obstacle_map(
                image.shape[:2], self.current_symbols, padding=15
            )
            
            # Expand obstacles for safety margin
            self.current_obstacle_map = ObstacleMapGenerator.expand_obstacles(
                self.current_obstacle_map, expansion=5
            )
            
            symbols_count = sum(len(symbols) for symbols in self.current_symbols.values())
            
            return {
                'method': 'symbol_detection',
                'symbols': self.current_symbols,
                'obstacle_map': self.current_obstacle_map,
                'symbols_count': symbols_count
            }
    
    def route_wire(self, start_point: Tuple[int, int], end_point: Tuple[int, int], 
                   routing_style: str = 'manhattan') -> List[Tuple[int, int]]:
        """
        Route a wire between two points avoiding obstacles
        
        Args:
            start_point: Starting point (x, y)
            end_point: End point (x, y)
            routing_style: 'manhattan' or 'astar'
        
        Returns:
            List of waypoints for the wire path
        """
        if self.current_obstacle_map is None:
            raise ValueError("Circuit not analyzed. Call analyze_circuit() first.")
        
        # Create pathfinder
        pathfinder = PathFinder(self.current_obstacle_map)
        
        # Find path
        if routing_style == 'manhattan':
            path = pathfinder.manhattan_routing(start_point, end_point)
            if not path:  # Fallback to A* if Manhattan fails
                path = pathfinder.a_star(start_point, end_point)
        else:
            path = pathfinder.a_star(start_point, end_point)
        
        # Optimize path
        if path:
            path = pathfinder.optimize_path(path)
        
        return path
    
    def get_connection_suggestions(self, point: Tuple[int, int], radius: int = 20) -> List[Tuple[int, int]]:
        """
        Suggest good connection points near a given point
        
        Args:
            point: Reference point (x, y)
            radius: Search radius in pixels
        
        Returns:
            List of suggested connection points
        """
        suggestions = []
        x, y = point
        
        # Look for component boundaries in the area
        for symbol_type, symbols in self.current_symbols.items():
            for sx, sy, sw, sh in symbols:
                # Check if symbol is near the point
                symbol_center_x = sx + sw // 2
                symbol_center_y = sy + sh // 2
                
                distance = np.sqrt((symbol_center_x - x)**2 + (symbol_center_y - y)**2)
                
                if distance <= radius:
                    # Suggest connection points on symbol edges
                    edge_points = [
                        (sx, symbol_center_y),      # Left edge
                        (sx + sw, symbol_center_y), # Right edge
                        (symbol_center_x, sy),      # Top edge
                        (symbol_center_x, sy + sh)  # Bottom edge
                    ]
                    suggestions.extend(edge_points)
        
        return suggestions
    
    def visualize_routing(self, image: np.ndarray, path: List[Tuple[int, int]], 
                         wire_color: Tuple[int, int, int] = (0, 255, 0), 
                         wire_thickness: int = 2) -> np.ndarray:
        """
        Visualize the wire routing on the image
        
        Args:
            image: Original circuit image
            path: Wire path waypoints
            wire_color: Color for drawing the wire (B, G, R)
            wire_thickness: Thickness of the wire line
        
        Returns:
            Image with drawn wire path
        """
        result = image.copy()
        
        if len(path) < 2:
            return result
        
        # Draw the wire path
        for i in range(len(path) - 1):
            cv2.line(result, path[i], path[i + 1], wire_color, wire_thickness)
        
        # Draw waypoints
        for i, point in enumerate(path):
            if i == 0:  # Start point
                cv2.circle(result, point, 5, (255, 0, 0), -1)  # Blue
            elif i == len(path) - 1:  # End point
                cv2.circle(result, point, 5, (0, 0, 255), -1)  # Red
            else:  # Waypoints
                cv2.circle(result, point, 3, (0, 255, 255), -1)  # Yellow
        
        return result
    
    def visualize_analysis(self, image: np.ndarray) -> np.ndarray:
        """
        Visualize the circuit analysis (detected symbols and obstacle map)
        
        Args:
            image: Original circuit image
        
        Returns:
            Image with visualization overlay
        """
        result = image.copy()
        
        # Draw detected symbols
        if self.current_symbols:
            result = self.symbol_detector.visualize_detections(result, self.current_symbols)
        
        # Overlay obstacle map (semi-transparent)
        if self.current_obstacle_map is not None:
            # Create colored obstacle overlay
            obstacle_colored = cv2.applyColorMap(self.current_obstacle_map, cv2.COLORMAP_JET)
            
            # Make it semi-transparent
            mask = self.current_obstacle_map > 0
            result[mask] = cv2.addWeighted(result[mask], 0.7, obstacle_colored[mask], 0.3, 0)
        
        return result
    
    def get_routing_statistics(self, path: List[Tuple[int, int]]) -> dict:
        """
        Get statistics about a wire routing path
        
        Args:
            path: Wire path waypoints
        
        Returns:
            Dictionary with routing statistics
        """
        if len(path) < 2:
            return {'length': 0, 'segments': 0, 'turns': 0}
        
        # Calculate total length
        total_length = 0
        turns = 0
        
        for i in range(len(path) - 1):
            # Calculate segment length
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            segment_length = np.sqrt(dx**2 + dy**2)
            total_length += segment_length
            
            # Count turns (direction changes)
            if i > 0:
                # Previous direction
                prev_dx = path[i][0] - path[i - 1][0]
                prev_dy = path[i][1] - path[i - 1][1]
                
                # Current direction
                curr_dx = path[i + 1][0] - path[i][0]
                curr_dy = path[i + 1][1] - path[i][1]
                
                # Check if direction changed
                if (prev_dx != 0 and curr_dx == 0) or (prev_dy != 0 and curr_dy == 0) or \
                   (prev_dx == 0 and curr_dx != 0) or (prev_dy == 0 and curr_dy != 0):
                    turns += 1
        
        return {
            'length': round(total_length, 2),
            'segments': len(path) - 1,
            'turns': turns,
            'waypoints': len(path)
        }
    
    def save_debug_images(self, image: np.ndarray, path: List[Tuple[int, int]], 
                         prefix: str = "debug") -> None:
        """
        Save debug images showing the routing process
        
        Args:
            image: Original circuit image
            path: Wire path
            prefix: Filename prefix for saved images
        """
        # Save original with detected symbols
        analysis_img = self.visualize_analysis(image)
        cv2.imwrite(f"{prefix}_analysis.png", analysis_img)
        
        # Save obstacle map
        if self.current_obstacle_map is not None:
            cv2.imwrite(f"{prefix}_obstacles.png", self.current_obstacle_map)
        
        # Save final routing
        routing_img = self.visualize_routing(image, path)
        cv2.imwrite(f"{prefix}_routing.png", routing_img)
        
        print(f"Debug images saved with prefix: {prefix}")

# Example usage and testing
if __name__ == "__main__":
    # Create router
    router = SmartWireRouter()
    
    # Test with a dummy image (in real use, load your circuit image)
    # image = cv2.imread('circuit_diagram.png')
    
    # For testing, create a simple test image
    test_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Add some fake components
    cv2.rectangle(test_image, (100, 100), (200, 150), (0, 0, 0), 2)  # Component 1
    cv2.rectangle(test_image, (350, 200), (450, 250), (0, 0, 0), 2)  # Component 2
    
    # Analyze circuit
    analysis = router.analyze_circuit(test_image)
    print(f"Detected {len(analysis['symbols'])} symbol types")
    
    # Route a wire
    start = (50, 125)
    end = (500, 225)
    path = router.route_wire(start, end)
    
    if path:
        print(f"Path found with {len(path)} waypoints")
        
        # Get statistics
        stats = router.get_routing_statistics(path)
        print(f"Routing statistics: {stats}")
        
        # Visualize
        result = router.visualize_routing(test_image, path)
        cv2.imshow('Smart Wire Routing', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No path found!")