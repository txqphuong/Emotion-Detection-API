import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os

# ---------- KIẾN TRÚC MÔ HÌNH ----------
class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------- CẤU HÌNH ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

model = EmotionCNN().to(device)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "app", "model", "best_model.pt")

model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()

mtcnn = MTCNN(keep_all=True, device=device)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---------- KHỞI TẠO FASTAPI ----------
app = FastAPI()

@app.post("/emotion")
async def analyze_emotion(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ JPG hoặc PNG.")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Không thể đọc file ảnh.")

    boxes, probs = mtcnn.detect(img)

    if boxes is None or len(boxes) == 0:
        return []

    results = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        width = x2 - x1
        height = y2 - y1

        face = img.crop((x1, y1, x2, y2))
        face_tensor = transform(face).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face_tensor)
            prob = F.softmax(output, dim=1)
            confidence, pred = torch.max(prob, 1)

        result = {
            "bbox": [x1, y1, width, height],
            "emotion": class_names[pred.item()],
            "confidence": round(confidence.item(), 2)
        }
        results.append(result)

    return JSONResponse(content=results)
