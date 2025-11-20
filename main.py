import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import os

# Optional dependencies for smart routing
SMART_ROUTING_AVAILABLE = False
try:
    import numpy as np
    import cv2
    from smart_router import SmartWireRouter
    SMART_ROUTING_AVAILABLE = True
    print("✓ Smart routing enabled (OpenCV and NumPy available)")
except (ImportError, Exception) as e:
    print(f"⚠ Smart routing disabled: {e}")
    print("  Install numpy and opencv-python to enable smart routing features")
    print("  Application will run with basic wire drawing features only.")

class CircuitWireDrawerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Circuit Wire Drawing Tool")
        self.root.geometry("1000x700")
        
        # Variables
        self.image = None
        self.photo = None
        self.original_image = None
        self.points = []
        self.wires = []
        self.wire_style = tk.StringVar(value="smart" if SMART_ROUTING_AVAILABLE else "orthogonal")
        self.wire_color = "red"
        self.wire_thickness = 2
        
        # Panning variables
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.space_pressed = False
        self.last_x = 0
        self.last_y = 0
        
        # Zoom variables
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        
        # Smart routing (only if dependencies available)
        if SMART_ROUTING_AVAILABLE:
            self.smart_router = SmartWireRouter()
            self.use_smart_routing = tk.BooleanVar(value=False)
            self.routing_style = tk.StringVar(value="manhattan")
            self.show_obstacles = tk.BooleanVar(value=False)  # Toggle for showing obstacles
        else:
            self.smart_router = None
            self.use_smart_routing = None
            self.routing_style = None
            self.show_obstacles = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Top Frame - Controls
        control_frame = tk.Frame(self.root, bg="#2c3e50", padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Select Image Button
        self.select_btn = tk.Button(
            control_frame, 
            text="📁 Select Image", 
            command=self.select_image,
            bg="#3498db",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=5
        )
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        # Wire Style
        tk.Label(control_frame, text="Wire Style:", bg="#2c3e50", fg="white", font=("Arial", 9)).pack(side=tk.LEFT, padx=(20, 5))
        
        style_frame = tk.Frame(control_frame, bg="#2c3e50")
        style_frame.pack(side=tk.LEFT)
        
        tk.Radiobutton(
            style_frame, 
            text="Orthogonal", 
            variable=self.wire_style, 
            value="orthogonal",
            bg="#2c3e50",
            fg="white",
            selectcolor="#34495e",
            font=("Arial", 9),
            command=self.on_wire_style_change
        ).pack(side=tk.LEFT)
        
        tk.Radiobutton(
            style_frame, 
            text="Straight", 
            variable=self.wire_style, 
            value="straight",
            bg="#2c3e50",
            fg="white",
            selectcolor="#34495e",
            font=("Arial", 9),
            command=self.on_wire_style_change
        ).pack(side=tk.LEFT)
        
        # Only show Smart option if dependencies available
        if SMART_ROUTING_AVAILABLE:
            tk.Radiobutton(
                style_frame, 
                text="Smart", 
                variable=self.wire_style, 
                value="smart",
                bg="#2c3e50",
                fg="white",
                selectcolor="#34495e",
                font=("Arial", 9)
            ).pack(side=tk.LEFT)
        
        # Color Selection
        tk.Label(control_frame, text="Color:", bg="#2c3e50", fg="white", font=("Arial", 9)).pack(side=tk.LEFT, padx=(20, 5))
        
        self.color_combo = ttk.Combobox(
            control_frame, 
            values=["red", "green", "blue", "black", "yellow", "orange"],
            state="readonly",
            width=8
        )
        self.color_combo.set("red")
        self.color_combo.pack(side=tk.LEFT, padx=5)
        self.color_combo.bind("<<ComboboxSelected>>", lambda e: self.change_color())
        
        # Thickness
        tk.Label(control_frame, text="Thickness:", bg="#2c3e50", fg="white", font=("Arial", 9)).pack(side=tk.LEFT, padx=(20, 5))
        
        self.thickness_spinbox = tk.Spinbox(
            control_frame,
            from_=1,
            to=10,
            width=5,
            command=self.change_thickness
        )
        self.thickness_spinbox.delete(0, tk.END)
        self.thickness_spinbox.insert(0, "2")
        self.thickness_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Undo Button
        self.undo_btn = tk.Button(
            control_frame,
            text="↶ Undo",
            command=self.undo_wire,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 9, "bold"),
            padx=10,
            pady=5
        )
        self.undo_btn.pack(side=tk.LEFT, padx=(20, 5))
        
        # Smart Routing Controls (only if dependencies available)
        if SMART_ROUTING_AVAILABLE:
            smart_frame = tk.Frame(control_frame, bg="#2c3e50")
            smart_frame.pack(side=tk.LEFT, padx=(20, 5))
            
            tk.Label(smart_frame, text="Smart Routing:", bg="#2c3e50", fg="white", font=("Arial", 9)).pack(side=tk.TOP)
            
            self.analyze_btn = tk.Button(
                smart_frame,
                text="🔍 Analyze",
                command=self.analyze_circuit,
                bg="#8e44ad",
                fg="white",
                font=("Arial", 8, "bold"),
                padx=8,
                pady=2
            )
            self.analyze_btn.pack(side=tk.LEFT, padx=2)
            
            # Toggle button to show/hide obstacles
            self.toggle_obstacles_btn = tk.Checkbutton(
                smart_frame,
                text="Show Components",
                variable=self.show_obstacles,
                command=self.toggle_obstacle_view,
                bg="#2c3e50",
                fg="white",
                selectcolor="#34495e",
                font=("Arial", 8)
            )
            self.toggle_obstacles_btn.pack(side=tk.LEFT, padx=2)
            
            routing_style_frame = tk.Frame(smart_frame, bg="#2c3e50")
            routing_style_frame.pack(side=tk.LEFT, padx=5)
            
            tk.Radiobutton(
                routing_style_frame,
                text="Manhattan",
                variable=self.routing_style,
                value="manhattan",
                bg="#2c3e50",
                fg="white",
                selectcolor="#34495e",
                font=("Arial", 8)
            ).pack(side=tk.TOP)
            
            tk.Radiobutton(
                routing_style_frame,
                text="A*",
                variable=self.routing_style,
                value="astar",
                bg="#2c3e50",
                fg="white",
                selectcolor="#34495e",
                font=("Arial", 8)
            ).pack(side=tk.TOP)
        
        # Clear Button
        self.clear_btn = tk.Button(
            control_frame,
            text="🗑 Clear All",
            command=self.clear_all,
            bg="#e67e22",
            fg="white",
            font=("Arial", 9, "bold"),
            padx=10,
            pady=5
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Zoom Controls
        zoom_frame = tk.Frame(control_frame, bg="#2c3e50")
        zoom_frame.pack(side=tk.LEFT, padx=(20, 5))
        
        tk.Label(zoom_frame, text="Zoom:", bg="#2c3e50", fg="white", font=("Arial", 9)).pack(side=tk.LEFT)
        
        self.zoom_out_btn = tk.Button(
            zoom_frame,
            text="🔍−",
            command=self.zoom_out,
            bg="#9b59b6",
            fg="white",
            font=("Arial", 8, "bold"),
            padx=8,
            pady=3
        )
        self.zoom_out_btn.pack(side=tk.LEFT, padx=2)
        
        self.zoom_label = tk.Label(
            zoom_frame, 
            text="100%", 
            bg="#2c3e50", 
            fg="white", 
            font=("Arial", 8),
            width=6
        )
        self.zoom_label.pack(side=tk.LEFT, padx=2)
        
        self.zoom_in_btn = tk.Button(
            zoom_frame,
            text="🔍+",
            command=self.zoom_in,
            bg="#9b59b6",
            fg="white",
            font=("Arial", 8, "bold"),
            padx=8,
            pady=3
        )
        self.zoom_in_btn.pack(side=tk.LEFT, padx=2)
        
        self.reset_zoom_btn = tk.Button(
            zoom_frame,
            text="1:1",
            command=self.reset_zoom,
            bg="#8e44ad",
            fg="white",
            font=("Arial", 8, "bold"),
            padx=8,
            pady=3
        )
        self.reset_zoom_btn.pack(side=tk.LEFT, padx=2)
        
        # Save Button
        self.save_btn = tk.Button(
            control_frame,
            text="💾 Save Image",
            command=self.save_image,
            bg="#27ae60",
            fg="white",
            font=("Arial", 9, "bold"),
            padx=10,
            pady=5
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Canvas Frame
        canvas_frame = tk.Frame(self.root, bg="#ecf0f1")
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas with Scrollbars
        self.canvas = tk.Canvas(canvas_frame, bg="white", cursor="cross")
        
        h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)
        
        # Bind key events for panning (canvas needs focus)
        self.canvas.bind("<KeyPress-space>", self.space_press)
        self.canvas.bind("<KeyRelease-space>", self.space_release)
        self.canvas.focus_set()  # Allow canvas to receive key events
        
        # Also bind to root window for better key handling
        self.root.bind("<KeyPress-space>", self.space_press)
        self.root.bind("<KeyRelease-space>", self.space_release)
        
        # Bind zoom events
        self.canvas.bind("<MouseWheel>", self.mouse_wheel_zoom)
        self.root.bind("<Control-plus>", lambda e: self.zoom_in())
        self.root.bind("<Control-equal>", lambda e: self.zoom_in())  # For + without shift
        self.root.bind("<Control-minus>", lambda e: self.zoom_out())
        self.root.bind("<Control-0>", lambda e: self.reset_zoom())
        self.root.bind("<Escape>", lambda e: self.clear_markers())
        self.root.bind("<F5>", lambda e: self.analyze_circuit())
        
        # Status Bar
        self.status_bar = tk.Label(
            self.root,
            text="Click 'Select Image' to start",
            bg="#34495e",
            fg="white",
            font=("Arial", 9),
            anchor=tk.W,
            padx=10,
            pady=5
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Circuit Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.original_image = Image.open(file_path)
                self.zoom_factor = 1.0  # Reset zoom for new image
                self.wires = []
                self.points = []
                # Clear any existing canvas content first
                self.canvas.delete("all")
                self.redraw_all()  # Use redraw_all instead of display_image
                # Ensure proper canvas setup after image load
                self.root.after(10, self.initialize_canvas_position)
                
                # Auto-analyze for smart routing if available
                if SMART_ROUTING_AVAILABLE:
                    self.root.after(100, self.auto_analyze_circuit)
                
                status_msg = f"Image loaded: {os.path.basename(file_path)} - Click 2 points to draw wire"
                if SMART_ROUTING_AVAILABLE:
                    status_msg += ". Press F5 to analyze circuit."
                self.status_bar.config(text=status_msg)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image:\n{str(e)}")
    
    def display_image(self):
        if self.image:
            # Simply display the current image (which should already have wires drawn)
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.delete("all")
            # Always create image at (0,0) for consistent coordinate system
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            # Force update of scroll region
            self.canvas.update_idletasks()
            bbox = self.canvas.bbox("all")
            if bbox:
                self.canvas.config(scrollregion=bbox)
                print(f"Display updated: bbox={bbox}, zoom={self.zoom_factor:.2f}")
            
            # Update zoom label
            self.zoom_label.config(text=f"{int(self.zoom_factor * 100)}%")
    
    def canvas_click(self, event):
        if self.image is None:
            messagebox.showwarning("No Image", "Please select an image first!")
            return
        
        # Check if we're in panning mode
        if self.space_pressed:
            self.panning = True
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.last_x = event.x
            self.last_y = event.y
            return
        
        # Don't process clicks if we were just panning
        if self.panning:
            return
        
        # Get canvas coordinates and adjust for zoom
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Debug: Check canvas scroll region
        scroll_region = self.canvas.cget('scrollregion')
        print(f"Canvas scroll region: {scroll_region}")
        
        # Check if click is within image bounds (in canvas coordinates)
        image_width = int(self.original_image.width * self.zoom_factor)
        image_height = int(self.original_image.height * self.zoom_factor)
        
        if canvas_x < 0 or canvas_y < 0 or canvas_x >= image_width or canvas_y >= image_height:
            self.status_bar.config(text="Click inside the image area")
            return
        
        # Convert to original image coordinates
        x = int(canvas_x / self.zoom_factor)
        y = int(canvas_y / self.zoom_factor)
        
        # Ensure coordinates are within original image bounds
        x = max(0, min(x, self.original_image.width - 1))
        y = max(0, min(y, self.original_image.height - 1))
        
        # Check if clicking on obstacle (if smart routing is active)
        if SMART_ROUTING_AVAILABLE and self.smart_router and self.smart_router.current_obstacle_map is not None:
            if self.smart_router.current_obstacle_map[y, x] == 255:
                self.status_bar.config(text=f"⚠️ Cannot place point on component! Click on empty space.")
                return
        
        # Add point
        self.points.append((x, y))
        
        # Draw point marker on canvas (scaled for zoom)
        marker_size = max(3, int(5 * self.zoom_factor))
        marker_width = max(2, int(3 * self.zoom_factor))
        
        # Create a more visible marker
        marker_id = self.canvas.create_oval(
            canvas_x-marker_size, canvas_y-marker_size, 
            canvas_x+marker_size, canvas_y+marker_size,
            fill="lime",
            outline="darkgreen",
            width=marker_width,
            tags="marker"
        )
        
        # Add a small text label
        self.canvas.create_text(
            canvas_x, canvas_y-marker_size-10,
            text=f"P{len(self.points)}",
            fill="red",
            font=("Arial", max(8, int(10 * self.zoom_factor)), "bold"),
            tags="marker"
        )
        
        bbox = self.canvas.bbox("all")
        print(f"CLICK DEBUG: Point {len(self.points)}")
        print(f"  Event coords: ({event.x}, {event.y})")
        print(f"  Canvas coords: ({canvas_x:.1f}, {canvas_y:.1f})")
        print(f"  Original coords: ({x}, {y})")
        print(f"  Zoom: {self.zoom_factor:.2f}x")
        print(f"  Image size: {image_width}x{image_height}")
        print(f"  Canvas bbox: {bbox}")
        print(f"  Scroll region: {scroll_region}")
        
        points_needed = 2 - len(self.points)
        if points_needed > 0:
            self.status_bar.config(text=f"✓ Point {len(self.points)} marked at ({x}, {y}) - Click {points_needed} more point(s)")
        
        # If we have 2 points, create the wire
        if len(self.points) == 2:
            try:
                print(f"\n=== WIRE CREATION DEBUG ===")
                print(f"Wire style selected: {self.wire_style.get()}")
                print(f"Smart routing available: {SMART_ROUTING_AVAILABLE}")
                print(f"Obstacle map exists: {self.smart_router and self.smart_router.current_obstacle_map is not None}")
                
                # ALWAYS use smart routing if available (ignore wire style selection)
                if SMART_ROUTING_AVAILABLE and self.smart_router and self.smart_router.current_obstacle_map is not None:
                    print(f"Using AUTOMATIC smart routing from {self.points[0]} to {self.points[1]}")
                    # Use smart routing
                    wire_path = self.create_smart_wire(self.points[0], self.points[1])
                    if not wire_path:
                        self.status_bar.config(text="⚠ Smart routing failed - no path found. Try different points.")
                        self.points = []
                        self.canvas.delete("marker")
                        return
                    print(f"Smart routing SUCCESS: {len(wire_path)} waypoints")
                    actual_style = "smart"  # Override style
                else:
                    # Fallback to traditional routing
                    wire_path = [self.points[0], self.points[1]]
                    actual_style = self.wire_style.get()
                    print(f"Using traditional routing: {actual_style}")
                
                print(f"Final wire path: {len(wire_path)} points")
                print(f"=== END DEBUG ===\n")
                
                # Store wire data (coordinates are in original image space)
                wire_data = {
                    'pt1': self.points[0],
                    'pt2': self.points[1],
                    'style': actual_style,  # Use actual routing style
                    'color': self.wire_color,
                    'thickness': self.wire_thickness,
                    'path': wire_path  # Store the full path for smart routing
                }
                self.wires.append(wire_data)
                
                # Redraw everything to show the new wire
                print(f"About to redraw with wire: {wire_data['pt1']} -> {wire_data['pt2']}")
                self.redraw_all()
                
                # Clear points and markers
                self.points = []
                self.canvas.delete("marker")
                
                self.status_bar.config(text=f"✓ Wire drawn! Total wires: {len(self.wires)} - Click 2 points for next wire")
                print(f"WIRE COMPLETE: Points {wire_data['pt1']} -> {wire_data['pt2']}, Total wires: {len(self.wires)}")
            except Exception as e:
                print(f"Error drawing wire: {e}")
                self.status_bar.config(text=f"Error drawing wire: {str(e)}")
    

    
    def draw_wire_on_image(self, pt1, pt2, style, color, thickness, wire_path=None):
        """Draw a single wire on the current image with proper zoom scaling"""
        if not self.image or not pt1 or not pt2:
            return
            
        try:
            draw = ImageDraw.Draw(self.image)
            scaled_thickness = max(1, int(thickness * self.zoom_factor))
            img_width, img_height = self.image.size
            
            if style == "smart" and wire_path:
                # Draw smart routed path
                for i in range(len(wire_path) - 1):
                    # Scale coordinates
                    scaled_pt1 = (int(wire_path[i][0] * self.zoom_factor), int(wire_path[i][1] * self.zoom_factor))
                    scaled_pt2 = (int(wire_path[i+1][0] * self.zoom_factor), int(wire_path[i+1][1] * self.zoom_factor))
                    
                    # Ensure coordinates are within bounds
                    scaled_pt1 = (max(0, min(scaled_pt1[0], img_width-1)), max(0, min(scaled_pt1[1], img_height-1)))
                    scaled_pt2 = (max(0, min(scaled_pt2[0], img_width-1)), max(0, min(scaled_pt2[1], img_height-1)))
                    
                    draw.line([scaled_pt1, scaled_pt2], fill=color, width=scaled_thickness)
            else:
                # Traditional routing
                scaled_pt1 = (int(pt1[0] * self.zoom_factor), int(pt1[1] * self.zoom_factor))
                scaled_pt2 = (int(pt2[0] * self.zoom_factor), int(pt2[1] * self.zoom_factor))
                
                # Ensure coordinates are within image bounds
                scaled_pt1 = (max(0, min(scaled_pt1[0], img_width-1)), max(0, min(scaled_pt1[1], img_height-1)))
                scaled_pt2 = (max(0, min(scaled_pt2[0], img_width-1)), max(0, min(scaled_pt2[1], img_height-1)))
                
                if style == "straight":
                    draw.line([scaled_pt1, scaled_pt2], fill=color, width=scaled_thickness)
                elif style == "orthogonal":
                    # Create L-shaped wire (horizontal then vertical)
                    mid_point = (scaled_pt2[0], scaled_pt1[1])
                    mid_point = (max(0, min(mid_point[0], img_width-1)), max(0, min(mid_point[1], img_height-1)))
                    draw.line([scaled_pt1, mid_point], fill=color, width=scaled_thickness)
                    draw.line([mid_point, scaled_pt2], fill=color, width=scaled_thickness)
        except Exception as e:
            print(f"Error drawing wire: {e}")
    
    def redraw_all(self):
        if self.original_image:
            # Check if we should show obstacles
            if hasattr(self, 'show_obstacles') and self.show_obstacles.get():
                self.redraw_with_obstacles()
                return
            
            # Always start with original image and scale up
            width = int(self.original_image.width * self.zoom_factor)
            height = int(self.original_image.height * self.zoom_factor)
            
            # Create zoomed base image
            if abs(self.zoom_factor - 1.0) > 0.001:  # Use small epsilon for float comparison
                self.image = self.original_image.resize((width, height), Image.Resampling.LANCZOS)
            else:
                self.image = self.original_image.copy()
            
            # Apply all wires to the zoomed image
            for wire in self.wires:
                wire_path = wire.get('path', None)
                self.draw_wire_on_image(wire['pt1'], wire['pt2'], wire['style'], wire['color'], wire['thickness'], wire_path)
            
            # Display and ensure proper canvas setup
            self.display_image()
            
            # Force canvas update to ensure coordinate system is ready
            self.canvas.update_idletasks()
            
            print(f"Redraw complete: {len(self.wires)} wires, zoom={self.zoom_factor:.2f}")
        else:
            # Clear canvas if no image
            self.canvas.delete("all")
            self.zoom_label.config(text="100%")
    
    def clear_markers(self):
        """Clear all point markers from canvas"""
        self.canvas.delete("marker")
        self.points = []
    
    def undo_wire(self):
        if self.wires:
            self.wires.pop()
            self.clear_markers()
            self.redraw_all()
            self.status_bar.config(text=f"Wire removed! Total wires: {len(self.wires)}")
            print(f"Wire undone. Remaining wires: {len(self.wires)}")
        else:
            # If no wires but there are markers, clear them
            if self.points:
                self.clear_markers()
                self.status_bar.config(text="Markers cleared")
            else:
                messagebox.showinfo("Info", "No wires to undo!")
    
    def clear_all(self):
        if self.wires or self.points:
            confirm = messagebox.askyesno("Confirm", "Clear all wires and markers?")
            if confirm:
                self.wires = []
                self.clear_markers()
                self.redraw_all()
                self.status_bar.config(text="All wires and markers cleared!")
                print("All wires and markers cleared")
    
    def on_wire_style_change(self):
        """Handle wire style change"""
        style = self.wire_style.get()
        if style == "smart" and not SMART_ROUTING_AVAILABLE:
            self.status_bar.config(text="⚠ Smart routing unavailable - requires NumPy and OpenCV. Using orthogonal.")
            self.wire_style.set("orthogonal")
        else:
            self.status_bar.config(text=f"Wire style: {style}")
    
    def change_color(self):
        self.wire_color = self.color_combo.get()
        self.status_bar.config(text=f"Wire color changed to: {self.wire_color}")
    
    def change_thickness(self):
        try:
            self.wire_thickness = int(self.thickness_spinbox.get())
            self.status_bar.config(text=f"Wire thickness changed to: {self.wire_thickness}")
        except:
            pass
    
    def canvas_drag(self, event):
        if self.panning and self.space_pressed:
            # Calculate movement delta
            dx = event.x - self.last_x
            dy = event.y - self.last_y
            
            # Get current scroll region
            x1, y1, x2, y2 = self.canvas.bbox("all")
            if x1 is not None:  # Make sure there's content
                # Move the canvas content smoothly
                self.canvas.move("all", dx, dy)
                
                # Update scroll region
                self.canvas.config(scrollregion=self.canvas.bbox("all"))
            
            # Update last position for next drag event
            self.last_x = event.x
            self.last_y = event.y
    
    def canvas_release(self, event):
        if self.panning:
            self.panning = False
            # Small delay to prevent accidental clicks after panning
            self.root.after(50, lambda: None)
    
    def space_press(self, event):
        if not self.space_pressed:
            self.space_pressed = True
            self.canvas.config(cursor="fleur")  # Hand/move cursor
            self.status_bar.config(text="Pan mode: Click and drag to move image - Release Space to return to wire mode")
    
    def space_release(self, event):
        if self.space_pressed:
            self.space_pressed = False
            self.panning = False
            self.canvas.config(cursor="cross")
            if len(self.wires) == 0:
                self.status_bar.config(text="Click 2 points to draw wire")
            else:
                self.status_bar.config(text=f"Total wires: {len(self.wires)} - Click 2 points for next wire")
    
    def mouse_wheel_zoom(self, event):
        if self.image:
            # Use event coordinates directly (these are canvas viewport coordinates)
            mouse_x = event.x
            mouse_y = event.y
            
            # Zoom in or out at mouse position
            if event.delta > 0:
                self.zoom_at_point(mouse_x, mouse_y, 1.2)
            else:
                self.zoom_at_point(mouse_x, mouse_y, 1/1.2)
    
    def zoom_in(self):
        if self.original_image and self.zoom_factor < self.max_zoom:
            # Get center of visible canvas area
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            center_x = canvas_width / 2
            center_y = canvas_height / 2
            
            self.zoom_at_point(center_x, center_y, 1.2)
    
    def zoom_out(self):
        if self.original_image and self.zoom_factor > self.min_zoom:
            # Get center of visible canvas area
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            center_x = canvas_width / 2
            center_y = canvas_height / 2
            
            self.zoom_at_point(center_x, center_y, 1/1.2)
    
    def zoom_at_point(self, canvas_x, canvas_y, zoom_multiplier):
        """Zoom in/out at specific point like CAD applications"""
        if not self.original_image:
            return
        
        # Calculate new zoom factor
        old_zoom = self.zoom_factor
        new_zoom = old_zoom * zoom_multiplier
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
        
        if abs(old_zoom - new_zoom) < 0.001:
            return  # No significant change
        
        # Get current scroll positions
        h_scroll_top = self.canvas.canvasx(0)
        v_scroll_top = self.canvas.canvasy(0)
        
        # Calculate the position in the original image that's under the mouse
        original_x = (canvas_x + h_scroll_top) / old_zoom
        original_y = (canvas_y + v_scroll_top) / old_zoom
        
        # Update zoom and redraw
        self.zoom_factor = new_zoom
        self.redraw_all()
        
        # Calculate where that point should be in the new zoomed image
        new_canvas_x = original_x * new_zoom
        new_canvas_y = original_y * new_zoom
        
        # Calculate the new scroll position to keep the mouse point in place
        new_h_scroll = new_canvas_x - canvas_x
        new_v_scroll = new_canvas_y - canvas_y
        
        # Update canvas position
        bbox = self.canvas.bbox("all")
        if bbox:
            # Move the image so the zoom point stays under the mouse
            current_x = bbox[0]
            current_y = bbox[1]
            
            target_x = -new_h_scroll
            target_y = -new_v_scroll
            
            offset_x = target_x - current_x
            offset_y = target_y - current_y
            
            if abs(offset_x) > 0.5 or abs(offset_y) > 0.5:
                self.canvas.move("all", offset_x, offset_y)
                self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
        # Update status
        self.status_bar.config(text=f"Zoom: {int(self.zoom_factor * 100)}%")
        print(f"Zoom changed from {old_zoom:.2f} to {new_zoom:.2f} at point ({canvas_x:.1f}, {canvas_y:.1f})")
    
    def reset_zoom(self):
        if self.original_image:
            self.zoom_factor = 1.0
            self.redraw_all()
            # Center the image after reset
            self.root.after(10, self.center_image)
            self.status_bar.config(text="Zoom reset to 100%")
    
    def center_image(self):
        """Center the image in the canvas view"""
        bbox = self.canvas.bbox("all")
        if bbox:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            img_width = bbox[2] - bbox[0]
            img_height = bbox[3] - bbox[1]
            
            # Calculate centering offset
            center_x = (canvas_width - img_width) / 2
            center_y = (canvas_height - img_height) / 2
            
            # Move image to center
            current_x = bbox[0]
            current_y = bbox[1]
            
            offset_x = center_x - current_x
            offset_y = center_y - current_y
            
            self.canvas.move("all", offset_x, offset_y)
            self.canvas.config(scrollregion=self.canvas.bbox("all"))
    
    def analyze_circuit(self):
        """Analyze the current circuit image for smart routing"""
        if not SMART_ROUTING_AVAILABLE:
            messagebox.showwarning("Feature Unavailable", 
                                 "Smart routing requires NumPy and OpenCV.\n\n"
                                 "Install with: pip install numpy opencv-python")
            return
        
        if not self.original_image:
            messagebox.showwarning("No Image", "Please load an image first!")
            return
        
        try:
            # Convert PIL image to OpenCV format
            cv_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
            
            # Analyze circuit with GENERIC OBSTACLE DETECTION
            self.status_bar.config(text="🔍 Analyzing circuit using generic obstacle detection...")
            self.root.update()
            
            analysis = self.smart_router.analyze_circuit(
                cv_image,
                dilation_size=5,    # Optimal: detects components without over-blocking
                safety_padding=3     # Optimal: adds clearance without excessive blocking
            )
            
            if analysis['method'] == 'generic':
                coverage = analysis['coverage']
                self.status_bar.config(
                    text=f"✓ Circuit analyzed: {coverage}% obstacle coverage (Generic detection - works with hand-drawn!)"
                )
            else:
                symbols_count = analysis.get('symbols_count', 0)
                self.status_bar.config(text=f"✓ Circuit analyzed: {symbols_count} components detected")
            
            print(f"Circuit analysis complete:")
            print(f"  Method: {analysis['method']}")
            if 'obstacle_pixels' in analysis:
                print(f"  Obstacle pixels: {analysis['obstacle_pixels']}")
                print(f"  Coverage: {analysis['coverage']}%")
            
            # Show obstacle overlay
            self.show_obstacle_overlay()
            
        except Exception as e:
            print(f"Error analyzing circuit: {e}")
            import traceback
            traceback.print_exc()
            self.status_bar.config(text=f"Error analyzing circuit: {str(e)}")
    
    def auto_analyze_circuit(self):
        """Auto-analyze circuit when image loads"""
        self.analyze_circuit()
    
    def toggle_obstacle_view(self):
        """Toggle showing/hiding obstacles"""
        if self.show_obstacles.get():
            self.show_obstacle_overlay_permanent()
        else:
            self.hide_obstacle_overlay()
    
    def show_obstacle_overlay_permanent(self):
        """Show obstacle overlay permanently until toggled off"""
        if not SMART_ROUTING_AVAILABLE or self.smart_router.current_obstacle_map is None:
            return
        
        try:
            # Redraw with obstacle overlay
            self.redraw_with_obstacles()
            self.status_bar.config(text="🔴 Red tint = Components (wires avoid these areas)")
            
        except Exception as e:
            print(f"Error showing obstacle overlay: {e}")
    
    def redraw_with_obstacles(self):
        """Redraw image with obstacle overlay"""
        if self.original_image:
            # Start with zoomed base image
            width = int(self.original_image.width * self.zoom_factor)
            height = int(self.original_image.height * self.zoom_factor)
            
            if abs(self.zoom_factor - 1.0) > 0.001:
                self.image = self.original_image.resize((width, height), Image.Resampling.LANCZOS)
            else:
                self.image = self.original_image.copy()
            
            # Add obstacle overlay if enabled
            if self.show_obstacles.get() and self.smart_router.current_obstacle_map is not None:
                # Get obstacle map scaled to current zoom
                from PIL import Image as PILImage
                obstacle_map = self.smart_router.current_obstacle_map
                obstacle_pil = PILImage.fromarray(obstacle_map)
                obstacle_scaled = obstacle_pil.resize((width, height), PILImage.Resampling.NEAREST)
                
                # Convert to numpy for processing
                obstacle_array = np.array(obstacle_scaled)
                overlay_array = np.array(self.image.convert('RGB'))
                
                # Apply red tint to obstacle areas (semi-transparent)
                mask = obstacle_array > 0
                overlay_array[mask, 0] = np.clip(overlay_array[mask, 0] + 80, 0, 255)  # Add red
                
                # Convert back to PIL
                self.image = PILImage.fromarray(overlay_array)
            
            # Apply all wires
            for wire in self.wires:
                wire_path = wire.get('path', None)
                self.draw_wire_on_image(wire['pt1'], wire['pt2'], wire['style'], wire['color'], wire['thickness'], wire_path)
            
            # Display
            self.display_image()
            self.canvas.update_idletasks()
    
    def show_obstacle_overlay(self):
        """Show visual overlay of detected obstacles"""
        if not SMART_ROUTING_AVAILABLE or self.smart_router.current_obstacle_map is None:
            return
        
        try:
            # Automatically turn on the toggle
            self.show_obstacles.set(True)
            self.show_obstacle_overlay_permanent()
            
        except Exception as e:
            print(f"Error showing obstacle overlay: {e}")
    
    def hide_obstacle_overlay(self):
        """Hide obstacle overlay and return to normal view"""
        self.redraw_all()
        self.status_bar.config(text="Ready to draw smart wires - they will avoid components")
    
    def create_smart_wire(self, start_point, end_point):
        """Create a smart routed wire path"""
        if not SMART_ROUTING_AVAILABLE:
            return None
            
        try:
            # Convert PIL image to OpenCV format if needed
            if not hasattr(self.smart_router, 'current_obstacle_map') or self.smart_router.current_obstacle_map is None:
                cv_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
                self.smart_router.analyze_circuit(cv_image)
            
            # Route the wire
            path = self.smart_router.route_wire(start_point, end_point, self.routing_style.get())
            
            if path:
                # Get routing statistics
                stats = self.smart_router.get_routing_statistics(path)
                print(f"Smart routing stats: {stats}")
                
                return path
            else:
                print("Smart routing failed - no path found")
                return None
                
        except Exception as e:
            print(f"Error in smart routing: {e}")
            return None
    
    def initialize_canvas_position(self):
        """Initialize canvas position and scroll region properly after image load"""
        if self.image:
            # Make sure the image is positioned at origin initially
            bbox = self.canvas.bbox("all")
            if bbox:
                # Reset canvas scroll to show top-left of image
                self.canvas.xview_moveto(0)
                self.canvas.yview_moveto(0)
                
                # Ensure scroll region is set correctly
                self.canvas.config(scrollregion=bbox)
                
                print(f"Canvas initialized: bbox={bbox}, scrollregion set")
    
    def save_image(self):
        if self.original_image:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                try:
                    # Create a copy of original image and draw wires at original resolution
                    save_image = self.original_image.copy()
                    for wire in self.wires:
                        wire_path = wire.get('path', None)
                        self.draw_wire_on_original(save_image, wire['pt1'], wire['pt2'], wire['style'], wire['color'], wire['thickness'], wire_path)
                    
                    save_image.save(file_path)
                    messagebox.showinfo("Success", f"Image saved to:\n{file_path}")
                    self.status_bar.config(text=f"Image saved: {os.path.basename(file_path)}")
                except Exception as e:
                    messagebox.showerror("Error", f"Could not save image:\n{str(e)}")
        else:
            messagebox.showwarning("No Image", "No image to save!")
    
    def draw_wire_on_original(self, image, pt1, pt2, style, color, thickness, wire_path=None):
        """Draw wire on original resolution image"""
        if not image or not pt1 or not pt2:
            return
            
        try:
            draw = ImageDraw.Draw(image)
            img_width, img_height = image.size
            
            if style == "smart" and wire_path:
                # Draw smart routed path
                for i in range(len(wire_path) - 1):
                    p1 = (max(0, min(wire_path[i][0], img_width-1)), max(0, min(wire_path[i][1], img_height-1)))
                    p2 = (max(0, min(wire_path[i+1][0], img_width-1)), max(0, min(wire_path[i+1][1], img_height-1)))
                    draw.line([p1, p2], fill=color, width=thickness)
            else:
                # Traditional routing
                pt1 = (max(0, min(pt1[0], img_width-1)), max(0, min(pt1[1], img_height-1)))
                pt2 = (max(0, min(pt2[0], img_width-1)), max(0, min(pt2[1], img_height-1)))
                
                if style == "straight":
                    draw.line([pt1, pt2], fill=color, width=thickness)
                elif style == "orthogonal":
                    mid_point = (pt2[0], pt1[1])
                    mid_point = (max(0, min(mid_point[0], img_width-1)), max(0, min(mid_point[1], img_height-1)))
                    draw.line([pt1, mid_point], fill=color, width=thickness)
                    draw.line([mid_point, pt2], fill=color, width=thickness)
        except Exception as e:
            print(f"Error drawing wire on original: {e}")


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = CircuitWireDrawerGUI(root)
    root.mainloop()