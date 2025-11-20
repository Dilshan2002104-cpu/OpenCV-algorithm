"""Quick test of smart routing"""
from smart_router import SmartWireRouter
import cv2
import numpy as np

# Load image
img = cv2.imread('c3140abe-a115-4cdf-a27d-07953f06ec65.jpg')
print(f"Image loaded: {img.shape}")

# Create router
router = SmartWireRouter()
print("Router created with generic detection")

# Analyze
print("\nAnalyzing circuit...")
analysis = router.analyze_circuit(img, dilation_size=7, safety_padding=5)

print(f"\nResults:")
print(f"  Method: {analysis['method']}")
print(f"  Coverage: {analysis.get('coverage', 'N/A')}%")
print(f"  Obstacle map created: {router.current_obstacle_map is not None}")

if router.current_obstacle_map is not None:
    print(f"  Obstacle map shape: {router.current_obstacle_map.shape}")
    print(f"  Max value: {router.current_obstacle_map.max()}")

# Test routing
print("\nTesting wire routing...")
start = (50, 100)
end = (250, 200)

try:
    path = router.route_wire(start, end, routing_style='manhattan')
    if path:
        print(f"✓ Path found with {len(path)} waypoints")
        stats = router.get_routing_statistics(path)
        print(f"  Length: {stats['length']}")
        print(f"  Turns: {stats['turns']}")
    else:
        print("✗ No path found")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
