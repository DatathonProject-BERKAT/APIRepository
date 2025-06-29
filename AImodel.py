class CNN_Model:
    def __init__(self):
        from ultralytics import YOLO
        import subprocess
        import cv2
        import os
        import torch
        self.torch = torch
        self.model = YOLO("yolo11n.pt")
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
        filename =  self.os.path.splitext(base_filename)[0]
        
        raw_output_path = self.os.path.join(output_folder, f"raw_{base_filename}")
        final_output_path = self.os.path.join(output_folder, f"processed_{base_filename}")

        

        fourcc = self.cv2.VideoWriter_fourcc(*'mp4v')
        out = self.cv2.VideoWriter(
            raw_output_path,
            fourcc,
            cap.get(self.cv2.CAP_PROP_FPS),
            (int(cap.get(self.cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT)))
        )

        
        if (filename not in self.progress):
            self.progress.update({filename : 0})
            
        counter = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            
            # 1 frame is processed
            self.progress[filename] = (counter/total_frames)*100
            if (self.progress[filename] < 98):
                counter += 1
                
        cap.release()
        out.release()

        # Re-encode for browser compatibility
        self.subprocess.run([
            "ffmpeg",
            "-y",                      # Overwrite if exists
            "-i", raw_output_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",     # Ensures compatibility with browsers
            final_output_path
        ])
        print(f"Final re-encoded video saved to: {final_output_path}")
        self.progress[filename] = 100
        # return f"/static/outputs/processed_{base_filename}"




if __name__ == "__main__":
    ai = CNN_Model()
    print(ai.model.device)
    
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
