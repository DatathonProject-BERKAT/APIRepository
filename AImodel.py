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
        self.progress = 0
    
    def predict(self,imgPath):
        img = self.cv2.imread(imgPath)

        result = self.model(img)

        annotated_img = result[0].plot()

        self.os.makedirs("imageFolder", exist_ok=True)

        # Extract filename from imgPath
        filename = self.os.path.basename(imgPath)
    
        save_path = self.os.path.join("imageFolder", filename)

        # Save the image
        self.cv2.imwrite(save_path, annotated_img)
        return {"result":result,"outputPath":save_path}
    
    def process_video(self, vidPath):

        cap = self.cv2.VideoCapture(vidPath)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {vidPath}")
        else:
            total_frames = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))
            
        output_folder = self.os.path.join("static", "outputs")
        self.os.makedirs(output_folder, exist_ok=True)

        base_filename = self.os.path.basename(vidPath)
        raw_output_path = self.os.path.join(output_folder, f"raw_{base_filename}")
        final_output_path = self.os.path.join(output_folder, f"processed_{base_filename}")

        fourcc = self.cv2.VideoWriter_fourcc(*'mp4v')
        out = self.cv2.VideoWriter(
            raw_output_path,
            fourcc,
            cap.get(self.cv2.CAP_PROP_FPS),
            (int(cap.get(self.cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT)))
        )

        self.progress = 0
        counter = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            
            # 1 frame is processed
            self.progress = (counter/total_frames)*100
            counter += 1
            
        cap.release()
        out.release()

        # Re-encode for browser compatibility
        self.subprocess.run([
            "ffmpeg",
            "-y",
            "-i", raw_output_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-c:a", "aac",
            final_output_path
        ])


        print(f"Final re-encoded video saved to: {final_output_path}")
        return final_output_path




if __name__ == "__main__":
    ai = CNN_Model()
    print(ai.model.device)
    
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
