# 😄 Emotion Detection API (Face + Emotion CNN)

A lightweight API that detects all faces in an image and classifies each face’s emotion using a CNN model trained on FER2013.

---

## 💡 What this project does

1. Detects **all faces** in an image using [MTCNN](https://github.com/timesler/facenet-pytorch)
2. Classifies **emotion** per face using a custom **CNN** model
3. Returns results as JSON: bounding box, emotion, and confidence for each face

---

## ✅ How this was built

### 1. Model Training (on Google Colab with T4 GPU)
- Trained `EmotionCNN` model using FER2013 dataset.
- Face detection via **MTCNN** (facenet-pytorch).
- Combined detection + classification in one pipeline.
- Final export as: `best_model.pt`

📁 Relevant notebooks:
- `emotion_cnn.ipynb` — CNN training
- `face_cnn.ipynb` — Face detection (MTCNN)
- `face_emotion_detection.ipynb` — Combine both

---

### 2. Build API with FastAPI
- Created `main.py` as FastAPI app.
- Defined endpoint: `POST /emotion`
- Input: image file (`multipart/form-data`)
- Output: list of detected faces with:
  ```json
  {
    "bbox": [x, y, width, height],
    "emotion": "happy",
    "confidence": 0.92
  }

## ✅ How to use
### 1. Clone repo & navigate
git clone <repo_url>
cd emotion_detection_api
### 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
or
source venv/bin/activate   # macOS/Linux
### 3. Install dependencies
pip install -r requirements.txt
### 4. Run the API locally
uvicorn app.main:app --reload
Open docs in browser:
📄 http://127.0.0.1:8000/docs

## 📤 API Usage
POST /emotion
Content-Type: multipart/form-data
Body:
file: Upload .jpg or .png image

### Response
[
  {
    "bbox": [120, 100, 60, 60],
    "emotion": "happy",
    "confidence": 0.92
  },
  ...
]

### 🧪 Test with curl (Windows)
-X POST http://127.0.0.1:8000/emotion -H "accept: application/json" -F "file=@\"C:/Users/Xuan Quynh Phuong/OneDrive - Ho Chi Minh city University of Food Industry/Desktop/emotion_detection_api/test_img/test.jpg\""

## 📁 Project Structure
emotion_detection_api/
├── app/
│   ├── main.py              # FastAPI app
│   ├── model/
│   │   └── best_model.pt    # Trained CNN model
├── test_img/
│   └── test.jpg             # Sample test image
├── requirements.txt
├── README.md
