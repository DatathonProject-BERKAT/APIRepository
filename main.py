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
    # Save the uploaded video
    upload_path = os.path.join(UPLOAD_DIR, video.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Start background processing using threading
    threading.Thread(
        target=model.process_video,
        args=(upload_path,),
        daemon=True
    ).start()

    # Prepare and return item info
    item = Item(
        id=int(video.filename.split('.')[0]),  # assumes filename is timestamp.ext
        video_path=f"/static/{video.filename}",
        output_path=f"/static/outputs/processed_{video.filename}"
    )
    items.append(item)
    return item

@app.get("/api/isFileExist", response_model=bool)
def is_file_exist(file_name: str):
    folder_path = os.path.join(os.path.dirname(__file__), "static", "outputs")
    file_path = os.path.join(folder_path, file_name)
    print(file_path)
    return os.path.isfile(file_path)

@app.get("/api/items/", response_model=List[Item])
def get_all_items():
    return items

@app.get("/api/progress/", response_model=Dict[str, int])
def get_all_progress():
    return {k: int(v) for k, v in model.progress.items()}

@app.delete("/api/items/{item_id}")
def delete_item(item_id: int):
    global items
    for i, item in enumerate(items):
        if item.id == item_id:
            # Delete the image file
            if os.path.exists(item.output_path.strip("/")):
                os.remove(item.output_path.strip("/"))
                os.remove(f"/static/outputs/raw_{item.id}.mp4".strip("/"))
                os.remove(f"/uploads/{item.id}.mp4".strip("/"))
            # Remove item
            del items[i]
            return {"message": "Item deleted"}
    raise HTTPException(status_code=404, detail="Item not found")
