"""
Debug why wires cut through components
"""
from PIL import Image
import numpy as np
import cv2
from smart_router import SmartWireRouter

# Load image
pil_image = Image.open('c3140abe-a115-4cdf-a27d-07953f06ec65.jpg')
cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

print("="*60)
print("TESTING SMART ROUTING")
print("="*60)

# Create router
router = SmartWireRouter()
print("\n1. Router created with generic detection: ✓")

# Analyze circuit
print("\n2. Analyzing circuit...")
analysis = router.analyze_circuit(cv_image, dilation_size=5, safety_padding=3)
print(f"   Method: {analysis['method']}")
print(f"   Coverage: {analysis['coverage']}%")
print(f"   Obstacle map: {router.current_obstacle_map is not None}")

# Test a wire that should avoid components
# Pick points that are clearly in free space
print("\n3. Testing wire routing...")
print("   Let's try multiple test points:\n")

test_cases = [
    ((50, 50), (400, 250), "Top-left to bottom-right"),
    ((100, 100), (350, 200), "Interior route"),
    ((20, 150), (440, 150), "Horizontal across"),
    ((230, 20), (230, 290), "Vertical down"),
]

for start, end, description in test_cases:
    print(f"   {description}")
    print(f"     Start: {start}, End: {end}")
    
    # Check if points are on obstacles
    if router.current_obstacle_map[start[1], start[0]] == 255:
        print(f"     ⚠️  Start point is ON obstacle!")
    if router.current_obstacle_map[end[1], end[0]] == 255:
        print(f"     ⚠️  End point is ON obstacle!")
    
    try:
        path = router.route_wire(start, end, routing_style='manhattan')
        if path:
            stats = router.get_routing_statistics(path)
            print(f"     ✓ Path found: {len(path)} waypoints, {stats['turns']} turns")
            
            # Check if path goes through obstacles
            obstacle_hits = 0
            for px, py in path:
                if router.current_obstacle_map[py, px] == 255:
                    obstacle_hits += 1
            
            if obstacle_hits > 0:
                print(f"     ⚠️  WARNING: Path crosses {obstacle_hits} obstacle pixels!")
            else:
                print(f"     ✓ Path is clear (avoids all obstacles)")
        else:
            print(f"     ✗ No path found")
    except Exception as e:
        print(f"     ✗ Error: {e}")
    
    print()

print("="*60)
print("\nCONCLUSION:")
print("If paths are found and avoid obstacles: Smart routing is working!")
print("If 'No path found': Points might be on obstacles or blocked")
print("If 'Path crosses obstacles': Bug in pathfinding algorithm")
print("="*60)
