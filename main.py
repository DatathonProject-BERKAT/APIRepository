from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from AImodel import CNN_Model
from typing import List,Dict
import shutil
import threading
import os
from datetime import datetime

app = FastAPI()
model = CNN_Model()

# Mount static files at /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory "database"
items = []

class Item(BaseModel):
    id: int
    video_path: str
    output_path: str

@app.get("/", response_class=FileResponse)
def serve_index():
    return FileResponse("static/index.html")

@app.post("/upload", response_model=Item)
async def upload_file(
    video: UploadFile = File(...)
):
    timestamp = datetime.now().strftime("%d%m%y%H%M%S")
    _, extension = os.path.splitext(video.filename)
    filename = f"{timestamp}{extension}"

    upload_path = os.path.join(UPLOAD_DIR, filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Process the video and get the path to processed video
    # output_url = model.process_video(upload_path)  # returns URL like /static/outputs/processed_260625123456.mp4

    threading.Thread(target=model.process_video, args=(upload_path,), daemon=True).start()
    
    item = Item(
        id=int(timestamp),
        video_path=f"/static/{filename}",     # optional, or remove if not needed
        output_path = f"/static/outputs/processed_{filename}"             # this is what will show in <video src>
    )
    items.append(item)

    return RedirectResponse(url="/", status_code=303)

@app.get("/api/items/", response_model=List[Item])
def get_all_items():
    return items

@app.get("/api/progress/",response_model=float)
def get_all_progress():
    return model.progress

@app.delete("/api/items/{item_id}")
def delete_item(item_id: int):
    global items
    for i, item in enumerate(items):
        if item.id == item_id:
            # Delete the image file
            if os.path.exists(item.video_path.strip("/")):
                os.remove(item.video_path.strip("/"))
            # Remove item
            del items[i]
            return {"message": "Item deleted"}
    raise HTTPException(status_code=404, detail="Item not found")
