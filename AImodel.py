class CNN_Model:
    def __init__(self):
        from ultralytics import YOLO
        import subprocess
        import cv2
        import os
        import time
        import torch
        import numpy as np
        import math
        self.math = math
        self.np = np
        self.time = time
        self.torch = torch
        # self.model = YOLO("yolo11n.pt")
        self.model = YOLO("miceDetectorModel.pt")
        if (self.torch.cuda.is_available()):
            self.model.to("cuda")
        self.subprocess = subprocess
        self.os = os
        self.cv2 = cv2
        self.progress = {}
    
    def process_video(self, vidPath):
        cap = self.cv2.VideoCapture(vidPath)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {vidPath}")
        else:
            total_frames = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))
            
        output_folder = self.os.path.join("static", "outputs")
        self.os.makedirs(output_folder, exist_ok=True)

        base_filename = self.os.path.basename(vidPath)
        filename = self.os.path.splitext(base_filename)[0]
        
        raw_output_path = self.os.path.join(output_folder, f"raw_{filename}.avi")
        final_output_path = self.os.path.join(output_folder, f"processed_{base_filename}")
        trajectory_path = self.os.path.join(output_folder, f"trajectory_{filename}.jpg")
        
        fourcc = self.cv2.VideoWriter_fourcc(*'mp4v')
        out = self.cv2.VideoWriter(
            raw_output_path,
            fourcc,
            cap.get(self.cv2.CAP_PROP_FPS),
            (int(cap.get(self.cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT)))
        )

        if filename not in self.progress:
            self.progress.update({filename: 0})
            
        counter = 1
        trail_points = []
        brightness_offset = 50  # brightness boost

        # Get dimensions for trajectory image
        width = int(cap.get(self.cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT))
        trajectory_img = self.np.zeros((height, width, 3), dtype=self.np.uint8)


        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Brighten frame
            frame = self.cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness_offset)
            
            # Run detection
            results = self.model(frame)[0]
            
            if len(results.boxes) > 0:
                best_box_idx = results.boxes.conf.argmax()
                box = results.boxes[best_box_idx]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                # Draw bounding box
                self.cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Save center point
                trail_points.append((cx, cy))

            # Draw trail on video frame
            # ==========================================================================
            # MEASURING DISTANCE (IN PROGRESS)
            last_point = None
            distance = 0
            for point in trail_points:
                self.cv2.circle(frame, point, radius=3, color=(0, 255, 0), thickness=-1)
                if last_point == None:
                    last_point = point
                else:
                    distance += self.pixel_distance(last_point,point)
                    last_point = point
                print(f"{self.convertPxToCM(distance,120,600)} CM <----")
            # ==========================================================================

            out.write(frame)

            # Update progress
            self.progress[filename] = (counter / total_frames) * 100
            if self.progress[filename] < 99:
                counter += 1

        cap.release()
        out.release()

        # Save the trajectory image
        for point in trail_points:
            self.cv2.circle(trajectory_img, point, radius=3, color=(0, 255, 0), thickness=-1)
        self.cv2.imwrite(trajectory_path, trajectory_img)
        print(f"Trajectory image saved to: {trajectory_path}")

        self.time.sleep(1)

        # Re-encode video for browser compatibility
        self.progress[filename] = 303  # encoding
        self.subprocess.run([
            "ffmpeg",
            "-y",
            "-i", raw_output_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            final_output_path
        ], stdout=self.subprocess.PIPE, stderr=self.subprocess.PIPE)

        print(f"Final re-encoded video saved to: {final_output_path}")
        self.progress[filename] = 100
        print(self.progress[filename])

        # return f"/static/outputs/processed_{base_filename}"

    # ==========================================================================
    # MEASURING DISTANCE (IN PROGRESS)
    def pixel_distance(self,p1, p2):
        return self.math.dist(p1, p2)
    
    def convertPxToCM(self, pixelDistance,realDiameterInCM,diameterInPixel):
        scale_cm_per_pixel = realDiameterInCM / diameterInPixel
        real_distance_cm = pixelDistance * scale_cm_per_pixel
        return real_distance_cm
    # ==========================================================================


if __name__ == "__main__":
    ai = CNN_Model()
    # print(ai.model.device)
    
    # import torch
    # print("CUDA available:", torch.cuda.is_available())
    # print("Current device:", torch.cuda.current_device())
    # print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    
    l = [1,2,3,4,5,6,7,8]
    
    lastNum = None
    for i in l:
        if lastNum == None:
            lastNum = i
        else:
            print(lastNum,i)
            lastNum = i