"""
Real-time Parking Data Annotator - Simplified Version
Click anywhere on the live feed to capture and annotate instantly

Usage:
    python annotator.py --source /dev/video2 --command park

Controls:
    CLICK: Capture frame with that steering target
    W/S/X: Change action mode (Forward/Backward/Stop)
    N: Start new sequence
    Q: Quit
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
import time

from Rosmaster_Lib import Rosmaster

car = Rosmaster()

class ParkingAnnotator:
    """
    Simplified interactive annotator - click to instantly save
    """
    
    def __init__(self, output_dir='parking_dataset', command='park', video_source=0):
        self.output_dir = Path(output_dir)
        self.command = command
        self.video_source = video_source
        
        # Display and save resolution
        self.display_width = 1280
        self.display_height = 720
        
        # Create directory structure
        self.command_dir = self.output_dir / command
        self.images_dir = self.command_dir / 'images'
        self.annotations_dir = self.command_dir / 'annotations'
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.frame_count = 0
        self.annotation_count = 0
        self.current_frame = None
        
        # Action mode (changes with W/S/X keys)
        self.action = 1.0  # Forward by default
        
        # For sequence tracking (RNN)
        self.sequence_id = int(time.time())
        self.frames_in_sequence = 0
        
        # Video capture
        self.cap = None
        
        # UI colors
        self.COLOR_TARGET = (0, 255, 0)  # Green
        self.COLOR_CENTER = (255, 255, 0)  # Cyan
        self.COLOR_TEXT = (255, 255, 255)  # White
        
        print("="*60)
        print("Simplified Parking Data Annotator")
        print("="*60)
        print(f"Output directory: {self.output_dir}")
        print(f"Command: {command}")
        print(f"Video source: {video_source}")
        print("="*60)
    
    def start(self):
        """Start the annotation session"""
        self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            print(f"ERROR: Could not open video source: {self.video_source}")
            return
        
        # Set camera properties for 1080p capture
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # Get actual video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"\nVideo opened:")
        print(f"  Capture: {width}x{height} @ {fps}fps")
        print(f"  Display/Save: {self.display_width}x{self.display_height}")
        print("\nControls:")
        print("  CLICK: Capture frame at clicked position")
        print("  W: Set action to FORWARD")
        print("  S: Set action to BACKWARD")
        print("  X: Set action to STOP")
        print("  N: Start new sequence")
        print("  Q: Quit")
        print("\nWorkflow:")
        print("  1. Press N at start of parking maneuver")
        print("  2. Set action mode (W/S/X)")
        print("  3. Click on target position to save frame")
        print("  4. Repeat throughout maneuver")
        print("="*60 + "\n")
        
        # Create window
        cv2.namedWindow('Parking Annotator', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Parking Annotator', self.display_width, self.display_height + 150)
        cv2.setMouseCallback('Parking Annotator', self.mouse_callback)
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("End of video or camera disconnected")
                break
            
            # Resize to 720p for display
            frame_display = cv2.resize(frame, (self.display_width, self.display_height))
            
            self.current_frame = frame_display.copy()
            self.frame_count += 1
            
            # Display frame with overlay
            display_frame = self.draw_overlay(frame_display)
            
            cv2.imshow('Parking Annotator', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('n'):
                self.start_new_sequence()
            elif key == ord('w'):
                self.action = 1.0
                print(f"Action mode: FORWARD ({self.action})")
            elif key == ord('s'):
                self.action = -1.0
                print(f"Action mode: BACKWARD ({self.action})")
            elif key == ord('x'):
                self.action = 0.0
                print(f"Action mode: STOP ({self.action})")
        
        self.cleanup()
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks - save annotation immediately"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_frame is None:
                return
            
            # Calculate relative_x from click position
            width = self.display_width
            relative_x = (x - width/2) / (width/2)
            relative_x = np.clip(relative_x, -1.0, 1.0)
            
            # Set car steering for preview
            car.set_akm_steering_angle(relative_x * 45)
            
            # Save immediately
            self.save_annotation(x, y, relative_x)
    
    def draw_overlay(self, frame):
        """Draw UI overlay on frame"""
        display = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw crosshair
        cv2.line(display, (width//2, 0), (width//2, height), self.COLOR_CENTER, 2)
        cv2.line(display, (0, height//2), (width, height//2), self.COLOR_CENTER, 2)
        
        # Info panel
        panel_height = 150
        info_panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        
        # Top row
        cv2.putText(info_panel, f"Annotations: {self.annotation_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 2)
        cv2.putText(info_panel, f"Sequence: {self.sequence_id}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 2)
        cv2.putText(info_panel, f"Frame in seq: {self.frames_in_sequence}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 2)
        
        # Current action mode (big and visible)
        action_text = "FORWARD" if self.action > 0.3 else "BACKWARD" if self.action < -0.3 else "STOP"
        action_color = (0, 255, 0) if self.action > 0.3 else (0, 0, 255) if self.action < -0.3 else (0, 255, 255)
        cv2.putText(info_panel, f"Action: {action_text}", (width//2, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, action_color, 3)
        
        cv2.putText(info_panel, "CLICK anywhere to save", (width//2, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(info_panel, "W:Forward | S:Backward | X:Stop | N:New Seq", (width//2, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 2)
        
        # Combine
        display = np.vstack([display, info_panel])
        
        return display
    
    def start_new_sequence(self):
        """Start a new sequence (for RNN training)"""
        self.sequence_id = int(time.time())
        self.frames_in_sequence = 0
        print(f"\n✓ Started new sequence: {self.sequence_id}")
    
    def save_annotation(self, click_x, click_y, relative_x):
        """Save frame and annotation immediately"""
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename_base = f"{self.command}_{self.sequence_id}_{self.frames_in_sequence:04d}_{timestamp}"
        
        image_filename = f"{filename_base}.png"
        annotation_filename = f"{filename_base}.json"
        
        # Save image at 720p
        image_path = self.images_dir / image_filename
        cv2.imwrite(str(image_path), self.current_frame)
        
        # Create annotation
        annotation = {
            # Basic info
            "image_filename": image_filename,
            "timestamp": timestamp,
            "frame_number": self.frame_count,
            
            # Targets (for CNN/RNN training)
            "relative_x": float(relative_x),
            "go_stop": float(self.action),
            
            # Sequence info (for RNN)
            "sequence_id": self.sequence_id,
            "frame_in_sequence": self.frames_in_sequence,
            
            # Target pixel coordinates
            "target_pixel_x": click_x,
            "target_pixel_y": click_y,
            
            # Image dimensions
            "image_width": self.display_width,
            "image_height": self.display_height,
            
            # Command type
            "command": self.command,
            
            # Labels
            "steering_label": "LEFT" if relative_x < -0.3 else "RIGHT" if relative_x > 0.3 else "STRAIGHT",
            "action_label": "FORWARD" if self.action > 0.3 else "BACKWARD" if self.action < -0.3 else "STOP"
        }
        
        # Save annotation
        annotation_path = self.annotations_dir / annotation_filename
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        self.annotation_count += 1
        self.frames_in_sequence += 1
        
        # Quick feedback
        print(f"✓ Saved #{self.annotation_count}: X={relative_x:.2f} ({annotation['steering_label']}), Action={annotation['action_label']}")
    
    def cleanup(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("Session Summary")
        print("="*60)
        print(f"Total annotations: {self.annotation_count}")
        print(f"Total frames in sequence: {self.frames_in_sequence}")
        print(f"Output: {self.output_dir / self.command}")
        print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simplified Parking Annotator')
    parser.add_argument('--source', type=str, default='/dev/video2', 
                       help='Video source (e.g., /dev/video2)')
    parser.add_argument('--output', type=str, default='parking_dataset',
                       help='Output directory for dataset')
    parser.add_argument('--command', type=str, default='park',
                       help='Command type (park, go_straight, etc.)')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a number (webcam index)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # Create and start annotator
    annotator = ParkingAnnotator(
        output_dir=args.output,
        command=args.command,
        video_source=source
    )
    
    annotator.start()