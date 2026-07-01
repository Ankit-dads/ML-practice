"""
Face Recognition Web App
---------------------------
A single Streamlit app combining:
  - Register New Face (capture via browser webcam)
  - Train Model (retrain CNN on current dataset)
  - Live Recognition (real-time recognition + attendance logging)
  - Attendance Dashboard (view logged data)

Usage:
    streamlit run app.py

Requires:
    pip install streamlit streamlit-webrtc opencv-contrib-python torch torchvision pandas av
"""

import os
import json
import csv
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ---------- CONFIG ----------
DATASET_DIR = "dataset"
MODEL_PATH = "face_cnn_model.pth"
LABELS_PATH = "class_labels.json"
ATTENDANCE_CSV = "attendance_log.csv"
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.70
LOG_COOLDOWN_SECONDS = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# ---------- MODEL ----------
class FaceCNN(nn.Module):
    def __init__(self, num_classes):
        super(FaceCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        return None, None
    with open(LABELS_PATH, "r") as f:
        class_names = json.load(f)
    model = FaceCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, class_names


def init_csv():
    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, "w", newline="") as f:
            csv.writer(f).writerow(["name", "date", "time", "confidence"])


def log_attendance(name, confidence):
    now = datetime.now()
    with open(ATTENDANCE_CSV, "a", newline="") as f:
        csv.writer(f).writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), f"{confidence*100:.1f}"])


INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# ---------- VIDEO PROCESSORS ----------
class RegisterProcessor(VideoProcessorBase):
    """Captures face crops into a target folder while capture_active is True."""
    def __init__(self):
        self.detector = cv2.CascadeClassifier(CASCADE_PATH)
        self.save_dir = None
        self.capture_active = False
        self.count = 0
        self.target_count = 100

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            if self.capture_active and self.save_dir and self.count < self.target_count:
                crop = cv2.resize(gray[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(os.path.join(self.save_dir, f"img_{self.count}.jpg"), crop)
                self.count += 1
            color = (0, 255, 0) if self.capture_active else (255, 165, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            if self.capture_active:
                cv2.putText(img, f"{self.count}/{self.target_count}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            break
        return av.VideoFrame.from_ndarray(img, format="bgr24")


class RecognizeProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = cv2.CascadeClassifier(CASCADE_PATH)
        self.model, self.class_names = load_model()
        self.last_logged = {}
        self.log_enabled = True

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.model is None:
            cv2.putText(img, "No trained model found", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            crop = gray[y:y+h, x:x+w]
            pil_img = Image.fromarray(crop)
            tensor = INFER_TRANSFORM(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, idx = torch.max(probs, 1)
                confidence, idx = confidence.item(), idx.item()

            if confidence >= CONFIDENCE_THRESHOLD:
                name = self.class_names[idx]
                color = (0, 255, 0)
                if self.log_enabled:
                    now_ts = time.time()
                    if name not in self.last_logged or (now_ts - self.last_logged[name]) > LOG_COOLDOWN_SECONDS:
                        log_attendance(name, confidence)
                        self.last_logged[name] = now_ts
            else:
                name, color = "Unknown", (0, 0, 255)

            label = f"{name} ({confidence*100:.1f}%)"
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------- TRAINING ----------
def train_model(epochs, progress_callback):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    full_dataset = datasets.ImageFolder(root=DATASET_DIR, transform=transform)
    class_names = full_dataset.classes
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = FaceCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = torch.argmax(model(images), 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = 100 * correct / total if total else 0
        progress_callback(epoch, epochs, val_acc)

    torch.save(model.state_dict(), MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump(class_names, f)
    return class_names


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Face Recognition System", layout="wide")
st.title("Face Recognition System")

tab1, tab2, tab3, tab4 = st.tabs(["Register New Face", "Train Model", "Live Recognition", "Attendance Dashboard"])

# --- TAB 1: REGISTER ---
with tab1:
    st.subheader("Capture face images for a new person")
    name_input = st.text_input("Person's name").strip().lower().replace(" ", "_")
    target_n = st.slider("Number of images to capture", 50, 300, 150)

    if name_input:
        save_path = os.path.join(DATASET_DIR, name_input)
        os.makedirs(save_path, exist_ok=True)

        ctx = webrtc_streamer(key="register", video_processor_factory=RegisterProcessor)

        if ctx.video_processor:
            ctx.video_processor.save_dir = save_path
            ctx.video_processor.target_count = target_n

            col1, col2 = st.columns(2)
            if col1.button("Start Capturing"):
                ctx.video_processor.capture_active = True
                ctx.video_processor.count = 0
            if col2.button("Stop Capturing"):
                ctx.video_processor.capture_active = False

            st.info(f"Captured so far: {ctx.video_processor.count}/{target_n}")
    else:
        st.info("Enter a name above to enable webcam capture.")

# --- TAB 2: TRAIN ---
with tab2:
    st.subheader("Train / Retrain the CNN")
    if os.path.exists(DATASET_DIR):
        people = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
        st.write(f"Found {len(people)} people in dataset: {people}")
    else:
        st.warning("No dataset folder found yet — register at least 2 people first.")
        people = []

    epochs = st.slider("Epochs", 5, 50, 25)
    if st.button("Start Training", disabled=len(people) < 2):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def callback(epoch, total_epochs, val_acc):
            progress_bar.progress(epoch / total_epochs)
            status_text.text(f"Epoch {epoch}/{total_epochs} — Val Accuracy: {val_acc:.2f}%")

        with st.spinner("Training in progress..."):
            classes = train_model(epochs, callback)
        st.success(f"Training complete. Classes: {classes}")

# --- TAB 3: LIVE RECOGNITION ---
with tab3:
    st.subheader("Live Recognition + Attendance Logging")
    init_csv()
    if not os.path.exists(MODEL_PATH):
        st.warning("No trained model found. Go to 'Train Model' tab first.")
    else:
        log_toggle = st.checkbox("Log attendance while recognizing", value=True)
        ctx2 = webrtc_streamer(key="recognize", video_processor_factory=RecognizeProcessor)
        if ctx2.video_processor:
            ctx2.video_processor.log_enabled = log_toggle

# --- TAB 4: DASHBOARD ---
with tab4:
    st.subheader("Attendance Dashboard")
    if not os.path.exists(ATTENDANCE_CSV):
        st.info("No attendance data yet.")
    else:
        df = pd.read_csv(ATTENDANCE_CSV)
        if df.empty:
            st.info("Attendance log is empty so far.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total check-ins", len(df))
            c2.metric("Unique people", df["name"].nunique())
            c3.metric("Days recorded", df["date"].nunique())

            people_filter = st.multiselect("Filter by person", sorted(df["name"].unique()),
                                            default=list(df["name"].unique()))
            fdf = df[df["name"].isin(people_filter)]

            st.dataframe(fdf.sort_values(by=["date", "time"], ascending=False), use_container_width=True)
            st.bar_chart(fdf["name"].value_counts())
            st.line_chart(fdf.groupby("date").size())

            st.download_button("Download filtered CSV", fdf.to_csv(index=False),
                                "attendance_filtered.csv", "text/csv")
            