import cv2
import json
import os
from pathlib import Path
import numpy as np

class CrossroadAnnotator:
    """
    Interactive annotator for crossroad detection from live camera feed
    Click on live video to capture and label frames
    """
    def __init__(self, dataset_root="crossroad_dataset"):
        self.dataset_root = Path(dataset_root)
        self.images_dir = self.dataset_root / "images"
        self.annotations_dir = self.dataset_root / "annotations"
        
        # Create directories if they don't exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_counter = 0
        
        # Load existing annotations count to continue numbering
        existing_annotations = list(self.annotations_dir.glob("*.json"))
        if existing_annotations:
            numbers = []
            for f in existing_annotations:
                try:
                    num = int(f.stem.split('_')[1])
                    numbers.append(num)
                except:
                    pass
            if numbers:
                self.image_counter = max(numbers) + 1
        
        # Camera settings
        self.camera_device = '/dev/video2'  # Camera device path
        self.capture_width = 1920  # Capture at full ultrawide resolution
        self.capture_height = 1080
        self.save_width = 1280  # Downscale to 720p for saving
        self.save_height = 720
        self.fps = 30
        
        print("=" * 60)
        print("Crossroad Camera Annotator")
        print("=" * 60)
        print(f"Camera: {self.camera_device}")
        print(f"Capture Resolution: {self.capture_width}x{self.capture_height}")
        print(f"Save Resolution: {self.save_width}x{self.save_height}")
        print(f"FPS: {self.fps}")
        print("=" * 60)
        print("Controls:")
        print("  'c' key     = Capture & label as CROSSROAD (1)")
        print("  'n' key     = Capture & label as NO CROSSROAD (0)")
        print("  's' key     = Show current stats")
        print("  'q' key     = Quit annotator")
        print("=" * 60)
        print("Controls:")
        print("  LEFT CLICK  = Capture & label as CROSSROAD (1)")
        print("  RIGHT CLICK = Capture & label as NO CROSSROAD (0)")
        print("  SPACE       = Show current stats")
        print("  'q' key     = Quit annotator")
        print("=" * 60)
        
        # Stats
        self.crossroad_count = 0
        self.no_crossroad_count = 0
        
        # Count existing annotations
        for ann_file in existing_annotations:
            try:
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                    if data['is_crossroad'] == 1:
                        self.crossroad_count += 1
                    else:
                        self.no_crossroad_count += 1
            except:
                pass
    
    def save_annotation(self, frame, is_crossroad):
        """Save the captured frame and its annotation"""
        # Generate filename
        filename = f"image_{self.image_counter:06d}"
        
        # Downscale frame to 720p before saving
        frame_resized = cv2.resize(frame, (self.save_width, self.save_height), 
                                   interpolation=cv2.INTER_AREA)
        
        # Save image
        image_path = self.images_dir / f"{filename}.png"
        cv2.imwrite(str(image_path), frame_resized)
        
        # Create annotation
        annotation = {
            "image_filename": f"{filename}.png",
            "is_crossroad": is_crossroad,
            "image_path": str(image_path.relative_to(self.dataset_root)),
            "original_resolution": f"{self.capture_width}x{self.capture_height}",
            "saved_resolution": f"{self.save_width}x{self.save_height}"
        }
        
        # Save annotation
        annotation_path = self.annotations_dir / f"{filename}.json"
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        # Update counters
        self.image_counter += 1
        if is_crossroad == 1:
            self.crossroad_count += 1
            label_text = "CROSSROAD"
        else:
            self.no_crossroad_count += 1
            label_text = "NO CROSSROAD"
        
        print(f"✓ Saved {filename} as {label_text} | Total: {self.image_counter} "
              f"(Crossroad: {self.crossroad_count}, No: {self.no_crossroad_count})")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for annotation"""
        frame = param['frame']
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click = Crossroad
            self.save_annotation(frame.copy(), is_crossroad=1)
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click = No crossroad
            self.save_annotation(frame.copy(), is_crossroad=0)
    
    def show_stats(self):
        """Print current dataset statistics"""
        total = self.crossroad_count + self.no_crossroad_count
        if total == 0:
            print("\nNo annotations yet!")
            return
        
        print("\n" + "=" * 60)
        print("Current Dataset Statistics")
        print("=" * 60)
        print(f"Total images: {total}")
        print(f"Crossroad (1): {self.crossroad_count} ({100*self.crossroad_count/total:.1f}%)")
        print(f"No Crossroad (0): {self.no_crossroad_count} ({100*self.no_crossroad_count/total:.1f}%)")
        print("=" * 60 + "\n")
    
    def run(self):
        """Run the camera annotator"""
        # Open camera with V4L2 backend explicitly
        cap = cv2.VideoCapture(self.camera_device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_device}")
            print("\nTroubleshooting:")
            print("1. Check if camera exists: ls -l /dev/video*")
            print("2. Check permissions: sudo chmod 666 /dev/video2")
            print("3. Try different camera: modify camera_device in the script")
            return
        
        # Verify actual resolution
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\n✓ Camera opened successfully")
        print(f"Actual capture resolution: {actual_width}x{actual_height}")
        print("Ready to annotate! Press 'c' for crossroad or 'n' for no crossroad.\n")
        
        window_name = "Crossroad Annotator - Live Feed"
        cv2.namedWindow(window_name)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Downscale frame for display (720p)
            frame_display = cv2.resize(frame, (self.save_width, self.save_height), 
                                      interpolation=cv2.INTER_AREA)
            
            # Create display with overlay
            display = frame_display.copy()
            h, w = display.shape[:2]
            
            # Semi-transparent overlay bar
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
            
            # Instructions
            cv2.putText(display, "C = Crossroad | N = No Crossroad | S = Stats | Q = Quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Stats on screen
            total = self.crossroad_count + self.no_crossroad_count
            stats_text = f"Total: {total} | Crossroad: {self.crossroad_count} | No: {self.no_crossroad_count}"
            cv2.putText(display, stats_text, 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting annotator...")
                break
            elif key == ord('c'):
                # Capture and label as crossroad
                self.save_annotation(frame.copy(), is_crossroad=1)
            elif key == ord('n'):
                # Capture and label as no crossroad
                self.save_annotation(frame.copy(), is_crossroad=0)
            elif key == ord('s'):
                self.show_stats()
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final stats
        print("\n" + "=" * 60)
        print("Annotation Session Complete!")
        print("=" * 60)
        self.show_stats()
        print(f"Dataset saved to: {self.dataset_root}")
        print("=" * 60)


def main():
    """Main function"""
    dataset_root = "crossroad_dataset"
    
    annotator = CrossroadAnnotator(dataset_root=dataset_root)
    annotator.run()


if __name__ == "__main__":
    main()