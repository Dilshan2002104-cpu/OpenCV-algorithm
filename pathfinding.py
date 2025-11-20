"""
Path Finding Algorithms for Circuit Wire Routing
Implements A*, Manhattan routing, and other pathfinding algorithms
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional
import cv2

class PathFinder:
    def __init__(self, obstacle_map: np.ndarray):
        """
        Initialize pathfinder with obstacle map
        
        Args:
            obstacle_map: Binary image where 255 = obstacle, 0 = free space
        """
        self.obstacle_map = obstacle_map
        self.height, self.width = obstacle_map.shape
    
    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        A* pathfinding algorithm
        
        Args:
            start: Starting point (x, y)
            goal: Goal point (x, y)
        
        Returns:
            List of waypoints from start to goal
        """
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            # Manhattan distance heuristic (good for grid-based routing)
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def get_neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            x, y = pos
            neighbors = []
            
            # 8-directional movement (can be changed to 4-directional for simpler routing)
            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # Check bounds and obstacles
                if (0 <= nx < self.width and 
                    0 <= ny < self.height and 
                    self.obstacle_map[ny, nx] == 0):
                    neighbors.append((nx, ny))
            
            return neighbors
        
        # A* implementation
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in get_neighbors(current):
                # Calculate cost (diagonal moves cost more)
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = 1.4 if dx + dy == 2 else 1.0  # Diagonal vs straight
                
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No path found
    
    def manhattan_routing(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Manhattan routing - only horizontal and vertical movements
        Tries horizontal-first and vertical-first routes
        
        Args:
            start: Starting point (x, y)
            goal: Goal point (x, y)
        
        Returns:
            List of waypoints from start to goal
        """
        def route_horizontal_first(start_pt: Tuple[int, int], end_pt: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
            path = []
            x1, y1 = start_pt
            x2, y2 = end_pt
            
            # Go horizontal first
            if x1 != x2:
                direction = 1 if x2 > x1 else -1
                for x in range(x1, x2 + direction, direction):
                    if (x < 0 or x >= self.width or 
                        y1 < 0 or y1 >= self.height or 
                        self.obstacle_map[y1, x] == 255):
                        return None  # Route blocked
                    path.append((x, y1))
            else:
                path.append((x1, y1))
            
            # Then go vertical
            if y1 != y2:
                direction = 1 if y2 > y1 else -1
                for y in range(y1 + direction, y2 + direction, direction):
                    if (x2 < 0 or x2 >= self.width or 
                        y < 0 or y >= self.height or 
                        self.obstacle_map[y, x2] == 255):
                        return None  # Route blocked
                    path.append((x2, y))
            
            return path
        
        def route_vertical_first(start_pt: Tuple[int, int], end_pt: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
            path = []
            x1, y1 = start_pt
            x2, y2 = end_pt
            
            # Go vertical first
            if y1 != y2:
                direction = 1 if y2 > y1 else -1
                for y in range(y1, y2 + direction, direction):
                    if (x1 < 0 or x1 >= self.width or 
                        y < 0 or y >= self.height or 
                        self.obstacle_map[y, x1] == 255):
                        return None  # Route blocked
                    path.append((x1, y))
            else:
                path.append((x1, y1))
            
            # Then go horizontal
            if x1 != x2:
                direction = 1 if x2 > x1 else -1
                for x in range(x1 + direction, x2 + direction, direction):
                    if (x < 0 or x >= self.width or 
                        y2 < 0 or y2 >= self.height or 
                        self.obstacle_map[y2, x] == 255):
                        return None  # Route blocked
                    path.append((x, y2))
            
            return path
        
        # Try both routes
        route1 = route_horizontal_first(start, goal)
        route2 = route_vertical_first(start, goal)
        
        # Choose the route that works and is shorter
        if route1 and route2:
            return route1 if len(route1) <= len(route2) else route2
        elif route1:
            return route1
        elif route2:
            return route2
        else:
            # If both Manhattan routes fail, fallback to A*
            return self.a_star(start, goal)
    
    def optimize_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Optimize path by removing unnecessary waypoints
        
        Args:
            path: Original path
        
        Returns:
            Optimized path with unnecessary points removed
        """
        if len(path) < 3:
            return path
        
        optimized = [path[0]]
        
        i = 0
        while i < len(path) - 1:
            j = i + 1
            
            # Look ahead to find the farthest point we can reach in a straight line
            while j < len(path):
                if self._can_connect_directly(path[i], path[j]):
                    j += 1
                else:
                    break
            
            # Add the farthest reachable point
            if j - 1 > i:
                optimized.append(path[j - 1])
                i = j - 1
            else:
                optimized.append(path[i + 1])
                i += 1
        
        # Ensure we end at the goal
        if optimized[-1] != path[-1]:
            optimized.append(path[-1])
        
        return optimized
    
    def _can_connect_directly(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """
        Check if two points can be connected with a straight line without hitting obstacles
        
        Args:
            start: Starting point
            end: End point
        
        Returns:
            True if direct connection is possible
        """
        # Use Bresenham's line algorithm to check all points on the line
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        x_step = 1 if x0 < x1 else -1
        y_step = 1 if y0 < y1 else -1
        
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            # Check if current point is obstacle
            if (x < 0 or x >= self.width or 
                y < 0 or y >= self.height or 
                self.obstacle_map[y, x] == 255):
                return False
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            
            if e2 > -dy:
                err -= dy
                x += x_step
            
            if e2 < dx:
                err += dx
                y += y_step
        
        return True
    
    def find_best_path(self, start: Tuple[int, int], goal: Tuple[int, int], prefer_manhattan: bool = True) -> List[Tuple[int, int]]:
        """
        Find the best path using the most appropriate algorithm
        
        Args:
            start: Starting point
            goal: Goal point
            prefer_manhattan: Whether to prefer Manhattan routing over A*
        
        Returns:
            Optimized path from start to goal
        """
        if prefer_manhattan:
            # Try Manhattan routing first (better for circuits)
            path = self.manhattan_routing(start, goal)
            if not path:
                # Fallback to A* if Manhattan fails
                path = self.a_star(start, goal)
        else:
            # Use A* directly
            path = self.a_star(start, goal)
        
        # Optimize the path
        if path:
            path = self.optimize_path(path)
        
        return path

class ObstacleMapGenerator:
    """Generate obstacle maps from detected symbols"""
    
    @staticmethod
    def create_obstacle_map(image_shape: Tuple[int, int], symbols_dict: dict, padding: int = 10) -> np.ndarray:
        """
        Create obstacle map from detected symbols
        
        Args:
            image_shape: Shape of the original image (height, width)
            symbols_dict: Dictionary of detected symbols
            padding: Padding around symbols in pixels
        
        Returns:
            Binary obstacle map (255 = obstacle, 0 = free space)
        """
        obstacle_map = np.zeros(image_shape, dtype=np.uint8)
        
        for symbol_type, symbols in symbols_dict.items():
            for x, y, w, h in symbols:
                # Add padding around symbols
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image_shape[1], x + w + padding)
                y2 = min(image_shape[0], y + h + padding)
                
                cv2.rectangle(obstacle_map, (x1, y1), (x2, y2), 255, -1)
        
        return obstacle_map
    
    @staticmethod
    def expand_obstacles(obstacle_map: np.ndarray, expansion: int = 3) -> np.ndarray:
        """
        Expand obstacles to ensure minimum clearance
        
        Args:
            obstacle_map: Original obstacle map
            expansion: Number of pixels to expand
        
        Returns:
            Expanded obstacle map
        """
        if expansion <= 0:
            return obstacle_map
        
        kernel = np.ones((expansion * 2 + 1, expansion * 2 + 1), np.uint8)
        expanded = cv2.dilate(obstacle_map, kernel, iterations=1)
        return expanded

# Example usage
if __name__ == "__main__":
    # Create a test obstacle map
    obstacle_map = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(obstacle_map, (30, 30), (70, 40), 255, -1)  # Obstacle
    
    # Create pathfinder
    pathfinder = PathFinder(obstacle_map)
    
    # Find path
    start = (10, 10)
    goal = (90, 90)
    path = pathfinder.find_best_path(start, goal)
    
    print(f"Path found with {len(path)} waypoints")
    for point in path:
        print(f"  {point}")