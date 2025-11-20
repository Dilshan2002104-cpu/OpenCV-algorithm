"""Test different dilation settings"""
from smart_router import SmartWireRouter
import cv2

img = cv2.imread('c3140abe-a115-4cdf-a27d-07953f06ec65.jpg')

# Test different settings
settings = [
    (3, 2, "Light dilation"),
    (5, 3, "Medium dilation"),
    (7, 5, "Heavy dilation"),
    (10, 7, "Very heavy dilation")
]

for dilation, padding, desc in settings:
    print(f"\n{'='*50}")
    print(f"Testing: {desc}")
    print(f"  Dilation: {dilation}, Padding: {padding}")
    
    router = SmartWireRouter()
    analysis = router.analyze_circuit(img, dilation_size=dilation, safety_padding=padding)
    
    print(f"  Coverage: {analysis['coverage']}%")
    
    # Test routing on opposite corners (should be far from components)
    start = (10, 10)
    end = (450, 300)
    
    try:
        path = router.route_wire(start, end, routing_style='manhattan')
        if path:
            print(f"  ✓ Path found: {len(path)} waypoints")
        else:
            print(f"  ✗ No path found")
            # Try A* instead
            path = router.route_wire(start, end, routing_style='astar')
            if path:
                print(f"  ✓ A* found path: {len(path)} waypoints")
            else:
                print(f"  ✗ A* also failed")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print(f"\n{'='*50}")
print("Recommendation: Use the setting that finds paths reliably")
