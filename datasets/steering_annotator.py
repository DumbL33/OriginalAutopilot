import cv2
import numpy as np
import os
import json
from pathlib import Path
import time
from datetime import datetime

from Rosmaster_Lib import Rosmaster

car = Rosmaster()
car.set_beep(100)

def debug_print(message):
    print(f"🔍 {message}")

# Test camera functionality
def test_camera_functionality():
    debug_print("Testing OpenCV camera functionality...")
    
    try:
        # Try to create OpenCV camera with /dev/video2
        cap = cv2.VideoCapture('/dev/video2')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        debug_print(f"OpenCV camera created successfully - Resolution: 1920x1080")
        
        # Try to read a frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print("❌ Could not read frame from OpenCV camera!")
            cap.release()
            return False
        
        debug_print(f"Frame read successfully - Shape: {frame.shape}")
        
        # Test window creation
        try:
            cv2.namedWindow('OpenCV Test', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('OpenCV Test', 800, 600)
            cv2.imshow('OpenCV Test', frame)
            cv2.waitKey(1000)  # Show for 1 second
            debug_print("OpenCV display test successful")
        except Exception as e:
            print(f"❌ OpenCV display test failed: {e}")
            cap.release()
            return False
        
        cap.release()
        cv2.destroyAllWindows()
        return True
        
    except Exception as e:
        print(f"❌ OpenCV initialization failed: {e}")
        return False

def create_relative_coordinates(click_x, click_y, display_width, display_height, orig_width, orig_height):
    """
    Convert click coordinates to relative coordinates (-1 to 1)
    - Center of image = (0, 0)
    - Left edge = x = -1, Right edge = x = 1
    - Top edge = y = -1, Bottom edge = y = 1
    """
    # Convert click coordinates from display resolution to original resolution
    scale_x = orig_width / display_width
    scale_y = orig_height / display_height
    
    orig_click_x = click_x * scale_x
    orig_click_y = click_y * scale_y
    
    # Convert to relative coordinates (-1 to 1)
    # Center of image should be (0, 0)
    relative_x = (orig_click_x / (orig_width / 2)) - 1.0
    relative_y = (orig_click_y / (orig_height / 2)) - 1.0
    
    # Clamp to [-1, 1] range to handle edge cases
    relative_x = max(-1.0, min(1.0, relative_x))
    relative_y = max(-1.0, min(1.0, relative_y))
    
    return relative_x, relative_y

class LiveCameraAnnotator:
    def __init__(self, camera_device='/dev/video2', base_output_dir="dataset1", display_width=1280, display_height=720):
        debug_print("Initializing LiveCameraAnnotator with OpenCV...")
        self.camera_device = camera_device
        self.base_output_dir = base_output_dir
        self.display_width = display_width
        self.display_height = display_height
        
        # Full HD capture resolution
        self.capture_width = 1920
        self.capture_height = 1080
        
        self.current_frame_display = None  # Frame for display (resized)
        self.current_frame_original = None  # Original 1920x1080 frame
        
        # OpenCV camera object
        self.cap = None
        
        # Keep track of image counters for each command
        self.command_counters = {}
        
        # Updated command system - now 5 commands including go_left
        self.current_command = 0
        self.current_go_stop = 1  # -1 = backward, 0 = stop, 1 = forward
        
        # Five commands now including go_left
        self.commands = {
            0: "go_straight",
            1: "go_right",
            2: "go_left",      # NEW COMMAND
            3: "park",
            4: "go_park"
        }
        
        # Five colors for the commands
        self.colors = {
            0: (0, 255, 0),      # green for go_straight
            1: (0, 165, 255),    # orange for go_right
            2: (255, 255, 0),    # cyan for go_left (NEW)
            3: (255, 0, 0),      # blue for park
            4: (255, 0, 255)     # magenta for go_park
        }
        
        # Go/stop labels and colors
        self.go_stop_labels = {-1: "BACKWARD", 0: "STOP", 1: "FORWARD"}
        self.go_stop_colors = {-1: (255, 0, 255), 0: (0, 0, 255), 1: (0, 255, 0)}
        
        # Annotation state
        self.pending_annotation = None  # Store click coordinates until saved
        self.last_save_time = 0
        self.save_count = 0
        
        # Track last saved annotation for deletion
        self.last_saved_files = None  # Will store (image_path, annotation_path, command_id)
        self.last_delete_time = 0
        
        # Create folder structure
        self.setup_command_folders()
        
        debug_print(f"Display resolution: {display_width}x{display_height}")
        debug_print(f"Capture resolution: {self.capture_width}x{self.capture_height} (preserves 109° FOV)")
        debug_print(f"Save resolution: {self.capture_width}x{self.capture_height} (Full HD)")
        debug_print("Using relative coordinates from -1 to 1")
        debug_print("LiveCameraAnnotator with OpenCV initialized successfully")
    
    def setup_command_folders(self):
        """Create folder structure: base_output_dir/command_name/images/ and annotations/"""
        debug_print("Setting up command folders...")
        
        Path(self.base_output_dir).mkdir(parents=True, exist_ok=True)
        
        for command_id, command_name in self.commands.items():
            command_dir = os.path.join(self.base_output_dir, command_name)
            images_dir = os.path.join(command_dir, "images")
            annotations_dir = os.path.join(command_dir, "annotations")
            
            Path(command_dir).mkdir(parents=True, exist_ok=True)
            Path(images_dir).mkdir(parents=True, exist_ok=True)
            Path(annotations_dir).mkdir(parents=True, exist_ok=True)
            
            # Initialize counter for each command
            self.command_counters[command_id] = self.get_next_image_number(command_id)
            
        debug_print(f"Created folder structure in: {self.base_output_dir}/")
        for command_name in self.commands.values():
            debug_print(f"   └── {command_name}/ (images/ + annotations/)")

    def get_command_paths(self, command_id):
        """Get the command folder, images folder and annotations paths"""
        command_name = self.commands[command_id]
        command_dir = os.path.join(self.base_output_dir, command_name)
        images_dir = os.path.join(command_dir, "images")
        annotations_dir = os.path.join(command_dir, "annotations")
        return command_dir, images_dir, annotations_dir

    def get_next_image_number(self, command_id):
        """Get the next image number for a specific command"""
        _, images_dir, _ = self.get_command_paths(command_id)
        if not os.path.exists(images_dir):
            return 1
        
        # Find existing image files and get the highest number
        existing_files = [f for f in os.listdir(images_dir) if f.startswith('image') and f.endswith('.jpg')]
        if not existing_files:
            return 1
        
        # Extract numbers from filenames like image1.jpg, image2.jpg
        numbers = []
        for filename in existing_files:
            try:
                # Extract number between 'image' and '.jpg'
                num_str = filename[5:-4]  # Remove 'image' and '.jpg'
                num_str = num_str.split('_')[0]  # Get first part before timestamp
                numbers.append(int(num_str))
            except ValueError:
                continue
        
        return max(numbers) + 1 if numbers else 1

    def get_new_image_filename(self, command_id):
        """Generate new image filename for the command with timestamp"""
        image_num = self.command_counters[command_id]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"image{image_num}_{timestamp}.jpg"

    def get_annotation_filename(self, image_filename):
        """Get corresponding annotation filename for an image"""
        name_without_ext = os.path.splitext(image_filename)[0]
        return f"{name_without_ext}.json"

    def delete_last_annotation(self):
        """Delete the most recently saved annotation and its image"""
        if not self.last_saved_files:
            print("❌ No recent annotation to delete")
            return False
        
        image_path, annotation_path, command_id = self.last_saved_files
        
        try:
            # Check if files exist before attempting deletion
            files_deleted = []
            
            if os.path.exists(image_path):
                os.remove(image_path)
                files_deleted.append("image")
                debug_print(f"Deleted image: {image_path}")
            
            if os.path.exists(annotation_path):
                os.remove(annotation_path)
                files_deleted.append("annotation")
                debug_print(f"Deleted annotation: {annotation_path}")
            
            if files_deleted:
                # Decrement the counter for this command
                if self.command_counters[command_id] > 1:
                    self.command_counters[command_id] -= 1
                
                # Decrement total save count
                if self.save_count > 0:
                    self.save_count -= 1
                
                # Clear the last saved files info
                self.last_saved_files = None
                self.last_delete_time = time.time()
                
                print(f"🗑️ Deleted last annotation: {' and '.join(files_deleted)} files removed")
                print(f"📊 Total saves now: {self.save_count}")
                return True
            else:
                print("❌ No files found to delete")
                return False
                
        except Exception as e:
            print(f"❌ Error deleting files: {e}")
            return False

    def initialize_camera(self):
        """Initialize OpenCV camera"""
        debug_print(f"Initializing OpenCV camera {self.camera_device}...")
        
        try:
            # Create OpenCV VideoCapture with 1920x1080 for 109° FOV
            self.cap = cv2.VideoCapture(self.camera_device)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            debug_print(f"OpenCV camera capture resolution: {self.capture_width}x{self.capture_height}")
            debug_print("OpenCV camera initialized successfully")
            
            # Test reading a frame
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                print(f"❌ Could not read frame from OpenCV camera {self.camera_device}")
                return False
            
            debug_print(f"Test frame shape: {test_frame.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Could not initialize OpenCV camera {self.camera_device}: {e}")
            return False

    def save_annotation_and_frame(self, display_x, display_y, command_id):
        """Save annotation and current frame to command-specific folders"""
        debug_print(f"Saving annotation and frame for command: {self.commands[command_id]}")
        
        # Convert click coordinates to relative coordinates (-1 to 1)
        relative_x, relative_y = create_relative_coordinates(
            display_x, display_y, 
            self.display_width, self.display_height,
            self.capture_width, self.capture_height
        )
        
        # Generate new image filename
        new_image_filename = self.get_new_image_filename(command_id)
        
        # Calculate steering angle from relative_x
        STEERING_GAIN = 45  # Gain factor to scale relative_x to steering angle
        STEERING_BIAS = 0   # Bias to adjust straight steering angle if needed
        steering_angle = relative_x * STEERING_GAIN + STEERING_BIAS
        car.set_akm_steering_angle(steering_angle)

        print("Relative x: ", relative_x)
        print("Steering angle", steering_angle)

        # Create annotation with relative coordinates
        annotation = {
            "image_filename": new_image_filename,
            "relative_x": relative_x,
            "relative_y": relative_y,
            "command": command_id,
            "command_name": self.commands[command_id],
            "go_stop": self.current_go_stop,
            "go_stop_label": self.go_stop_labels[self.current_go_stop],
            "image_resolution": {
                "width": self.capture_width,
                "height": self.capture_height
            },
            "coordinate_system": "relative",
            "coordinate_range": "[-1, 1]",
            "center_point": "(0, 0)"
        }
        
        # Get paths
        _, images_dir, annotations_dir = self.get_command_paths(command_id)
        
        # Save annotation JSON
        annotation_filename = self.get_annotation_filename(new_image_filename)
        annotation_path = os.path.join(annotations_dir, annotation_filename)
        
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        # Save original Full HD frame to images folder
        dest_image_path = os.path.join(images_dir, new_image_filename)
        cv2.imwrite(dest_image_path, self.current_frame_original)
        
        # Store info for potential deletion
        self.last_saved_files = (dest_image_path, annotation_path, command_id)
        
        # Increment counter for this command
        self.command_counters[command_id] += 1
        self.save_count += 1
        
        debug_print(f"Saved: {annotation_path}")
        debug_print(f"Saved: {dest_image_path} ({self.capture_width}x{self.capture_height} Full HD)")
        debug_print(f"Relative coordinates: ({relative_x:.4f}, {relative_y:.4f}) in range [-1, 1]")
        debug_print(f"Total saves: {self.save_count}")
        
        return annotation_path, dest_image_path, new_image_filename, annotation

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            debug_print(f"Mouse clicked at display coordinates ({x}, {y})")
            
            # Store pending annotation
            self.pending_annotation = {
                "x": x,
                "y": y,
                "command": self.current_command,
                "go_stop": self.current_go_stop
            }
            
            # Immediately save the annotation and current frame
            annotation_path, image_path, filename, annotation = self.save_annotation_and_frame(
                x, y, self.current_command
            )
            
            print(f"✅ Saved frame #{self.save_count}: {self.commands[self.current_command]} - {self.go_stop_labels[self.current_go_stop]}")
            print(f"   📍 Display click: ({x}, {y}) → Relative coords: ({annotation['relative_x']:.4f}, {annotation['relative_y']:.4f})")
            print(f"   📁 Image: {filename} ({self.capture_width}x{self.capture_height})")
            
            # Clear pending annotation after saving
            self.pending_annotation = None
            
            self.last_save_time = time.time()

    def draw_frame(self):
        if self.current_frame_display is None:
            return
        
        img_copy = self.current_frame_display.copy()
        h, w = img_copy.shape[:2]
        
        # Draw current command and direction info
        command_text = f"Command: {self.commands[self.current_command]} ({self.current_command})"
        direction_text = f"Direction: {self.go_stop_labels[self.current_go_stop]} ({self.current_go_stop})"
        
        command_color = self.colors[self.current_command]
        direction_color = self.go_stop_colors[self.current_go_stop]
        
        # Background for info (scaled for display resolution)
        info_height = 120  # Increased height for delete info
        cv2.rectangle(img_copy, (10, 10), (450, info_height), (0, 0, 0), -1)
        cv2.rectangle(img_copy, (10, 10), (450, info_height), (255, 255, 255), 2)
        
        # Draw command and direction
        cv2.putText(img_copy, command_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, command_color, 2)
        cv2.putText(img_copy, direction_text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, direction_color, 2)
        
        # Draw controls - updated to include go_left command
        controls_text = "0:straight 1:right 2:left 3:park 4:go_park f/s/b d:delete click=save q=quit"
        cv2.putText(img_copy, controls_text, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw delete status
        delete_status = "Last can be deleted" if self.last_saved_files else "Nothing to delete"
        delete_color = (0, 255, 255) if self.last_saved_files else (128, 128, 128)
        cv2.putText(img_copy, delete_status, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, delete_color, 1)
        
        # Draw resolution info (top-right)
        resolution_info = [
            f"Capture: {self.capture_width}x{self.capture_height} (109° FOV)",
            f"Display: {self.display_width}x{self.display_height}",
            f"Save: {self.capture_width}x{self.capture_height} (Full HD)",
            "Coords: Relative [-1, 1]"
        ]
        
        for i, text in enumerate(resolution_info):
            y_pos = 25 + i * 20
            cv2.putText(img_copy, text, (w - 320, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw save count
        save_text = f"Saved: {self.save_count}"
        cv2.putText(img_copy, save_text, (w - 100, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw live indicator
        live_color = (0, 255, 0)
        cv2.circle(img_copy, (w - 30, 120), 8, live_color, -1)  # Green dot
        cv2.putText(img_copy, "LIVE", (w - 60, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, live_color, 2)
        
        # Draw crosshair - full screen
        center_x, center_y = w // 2, h // 2
        
        # Horizontal line - from left edge to right edge
        cv2.line(img_copy, (0, center_y), (w, center_y), (255, 255, 255), 2)
        
        # Vertical line - from top edge to bottom edge  
        cv2.line(img_copy, (center_x, 0), (center_x, h), (255, 255, 255), 2)
        
        # Center circle for better visibility (represents (0,0) in relative coordinates)
        cv2.circle(img_copy, (center_x, center_y), 5, (255, 255, 255), -1)
        
        # Draw coordinate system info at center
        cv2.putText(img_copy, "(0,0)", (center_x + 10, center_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw pending annotation if exists
        if self.pending_annotation:
            x, y = self.pending_annotation["x"], self.pending_annotation["y"]
            command = self.pending_annotation["command"]
            go_stop = self.pending_annotation["go_stop"]
            
            # Draw main circle
            color = self.colors[command]
            cv2.circle(img_copy, (x, y), 15, color, -1)
            cv2.circle(img_copy, (x, y), 18, (255, 255, 255), 3)
            
            # Draw command text
            cv2.putText(img_copy, self.commands[command], (x + 25, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw direction text
            direction_color = self.go_stop_colors[go_stop]
            direction_text = self.go_stop_labels[go_stop]
            cv2.putText(img_copy, direction_text, (x + 25, y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, direction_color, 2)
        
        # Show recent save feedback
        if time.time() - self.last_save_time < 2:  # Show for 2 seconds
            center_x, center_y = w // 2, h // 2
            cv2.putText(img_copy, "SAVED!", (center_x - 50, center_y - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Show recent delete feedback
        if time.time() - self.last_delete_time < 2:  # Show for 2 seconds
            center_x, center_y = w // 2, h // 2
            cv2.putText(img_copy, "DELETED!", (center_x - 70, center_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        cv2.imshow('Live Camera Annotator - Relative Coords', img_copy)

    def run(self):
        debug_print("Starting LiveCameraAnnotator.run() with OpenCV")
        
        # Initialize camera
        if not self.initialize_camera():
            return
        
        # Create window
        cv2.namedWindow('Live Camera Annotator - Relative Coords', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Live Camera Annotator - Relative Coords', self.display_width, self.display_height)
        cv2.setMouseCallback('Live Camera Annotator - Relative Coords', self.mouse_callback)
        
        debug_print("Starting OpenCV camera feed... Press 'q' to quit")
        print(f"🖼️ Capturing at {self.capture_width}x{self.capture_height} (preserves full 109° FOV)")
        print(f"🖥️ Displaying at {self.display_width}x{self.display_height} (for comfortable labeling)")
        print(f"💾 Saving at {self.capture_width}x{self.capture_height} (Full HD)")
        print(f"📍 Using relative coordinates from -1 to 1 (center = 0,0)")
        
        try:
            while True:
                # Read frame from OpenCV camera (1920x1080)
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    print("❌ Could not read frame from OpenCV camera")
                    break
                
                # Store original frame (1920x1080)
                self.current_frame_original = frame.copy()
                
                # Create display frame (resized for comfortable viewing)
                self.current_frame_display = cv2.resize(
                    frame, 
                    (self.display_width, self.display_height), 
                    interpolation=cv2.INTER_AREA
                )
                
                # Update window title with current status
                title = f"Live Camera Annotator - {self.commands[self.current_command]} - {self.go_stop_labels[self.current_go_stop]} - Saved: {self.save_count}"
                cv2.setWindowTitle('Live Camera Annotator - Relative Coords', title)
                
                # Draw and display frame
                self.draw_frame()
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print(f"💾 Session complete! {self.save_count} frames saved")
                    break
                
                elif key == ord('0'):
                    # Go straight
                    self.current_command = 0
                    debug_print(f"Command changed to: {self.commands[self.current_command]}")
                
                elif key == ord('1'):
                    # Go right
                    self.current_command = 1
                    debug_print(f"Command changed to: {self.commands[self.current_command]}")
                
                elif key == ord('2'):
                    # Go left (NEW COMMAND)
                    self.current_command = 2
                    debug_print(f"Command changed to: {self.commands[self.current_command]}")
                
                elif key == ord('3'):
                    # Park
                    self.current_command = 3
                    debug_print(f"Command changed to: {self.commands[self.current_command]}")
                
                elif key == ord('4'):
                    # Go park
                    self.current_command = 4
                    debug_print(f"Command changed to: {self.commands[self.current_command]}")
                
                elif key == ord('f'):
                    # Forward
                    self.current_go_stop = 1
                    debug_print(f"Direction changed to: {self.go_stop_labels[self.current_go_stop]}")
                
                elif key == ord('s'):
                    # Stop
                    self.current_go_stop = 0
                    debug_print(f"Direction changed to: {self.go_stop_labels[self.current_go_stop]}")
                
                elif key == ord('b'):
                    # Backward
                    self.current_go_stop = -1
                    debug_print(f"Direction changed to: {self.go_stop_labels[self.current_go_stop]}")
                
                elif key == ord('d'):
                    # Delete last annotation
                    self.delete_last_annotation()
                
                elif key == ord('h'):
                    # Help
                    self.show_help()
        
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
    
    def show_help(self):        
        help_text = f"""
🎯 LIVE CAMERA STEERING ANNOTATOR - RELATIVE COORDINATES
Frames saved: {self.save_count}

RESOLUTION STRATEGY:
📹 Capture: {self.capture_width}x{self.capture_height} (preserves full 109° FOV)
🖥️ Display: {self.display_width}x{self.display_height} (comfortable labeling)
💾 Save: {self.capture_width}x{self.capture_height} (Full HD)
📊 Coordinates: Relative [-1, 1] (center = 0,0)

CONTROLS:
- Click on camera feed: Save Full HD frame with relative coordinates
- d: Delete last saved annotation and image

COMMANDS (0-4):
- 0: Go Straight (Green)
- 1: Go Right (Orange)
- 2: Go Left (Cyan) ⭐ NEW
- 3: Park (Blue)
- 4: Go Park (Magenta)

DIRECTION CONTROLS:
- f: FORWARD (Green)
- s: STOP (Red)  
- b: BACKWARD (Magenta)

OTHER:
- h: Show this help
- q: Quit

💾 Each click saves full {self.capture_width}x{self.capture_height} frame with relative coordinates
📊 Relative coordinates: Center (0,0), Left edge (-1,y), Right edge (1,y)
🗑️ Press 'd' to delete the most recently saved annotation and image
📁 Files saved in: {self.base_output_dir}/[command]/images/ and annotations/
🚀 Using OpenCV with {self.camera_device} - preserving full 109° FOV
🎯 Generalized coordinates perfect for ML training!
        """
        print(help_text)

def main():
    print("🚀 LIVE CAMERA ANNOTATOR - RELATIVE COORDINATES")
    print("=" * 60)
    
    # Test camera first
    if not test_camera_functionality():
        print("❌ Camera test failed. Please check your camera connection.")
        return
    
    # Run live camera annotator
    print("Starting live camera annotator with relative coordinates...")
    
    try:
        # You can adjust display_width and display_height for comfortable labeling
        annotator = LiveCameraAnnotator(
            camera_device='/dev/video2',
            display_width=1280,  # Comfortable display size
            display_height=720   # 16:9 aspect ratio
        )
        print("✅ LiveCameraAnnotator created successfully")
        
        print("\n📋 CONTROLS:")
        print("- Click on camera feed: Save full resolution frame with relative coordinates")
        print("- Commands: 0 (go_straight), 1 (go_right), 2 (go_left), 3 (park), 4 (go_park)")
        print("- Direction: f (forward), s (stop), b (backward)")
        print("- d: Delete last saved annotation")
        print("- h: Show help")
        print("- q: Quit")
        
        print(f"\n📁 Data will be saved to: {annotator.base_output_dir}/")
        print("📊 Each command gets its own folder with images/ and annotations/ subfolders")
        print("🎯 Coordinates are normalized to [-1, 1] range for ML training")
        
        # Show initial help
        annotator.show_help()
        
        # Start the annotation process
        annotator.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🏁 Live Camera Annotator session ended")

if __name__ == "__main__":
    main()