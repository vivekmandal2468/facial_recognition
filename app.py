import streamlit as st
import cv2
import os
import numpy as np
from datetime import datetime
import pandas as pd

DATA_DIR = "data/known_faces"
ATTENDANCE_FILE = "data/attendance.csv"

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Train recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

def train_model():
    faces = []
    labels = []
    label_map = {}
    label_id = 0

    for person in os.listdir(DATA_DIR):
        person_path = os.path.join(DATA_DIR, person)
        if not os.path.isdir(person_path): continue

        label_map[label_id] = person
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            faces.append(img)
            labels.append(label_id)
        label_id += 1

    if faces:
        recognizer.train(faces, np.array(labels))
        return label_map
    return {}

def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    new_entry = {'Name': name, 'Date': date_str, 'Time': time_str}

    if not os.path.exists(ATTENDANCE_FILE):
        pd.DataFrame([new_entry]).to_csv(ATTENDANCE_FILE, index=False)
    else:
        df = pd.read_csv(ATTENDANCE_FILE)
        if not ((df['Name'] == name) & (df['Date'] == date_str)).any():
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            df.to_csv(ATTENDANCE_FILE, index=False)

def detect_and_recognize(image, label_map):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_rects:
        roi_gray = gray[y:y + h, x:x + w]
        try:
            roi_gray = cv2.resize(roi_gray, (200, 200))
            label, confidence = recognizer.predict(roi_gray)
            name = label_map.get(label, "Unknown")
            if confidence < 70:
                mark_attendance(name)
                label_text = f"{name} ({confidence:.1f})"
            else:
                label_text = "Unknown"
        except:
            label_text = "Error"

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return image

# Streamlit UI
st.title("📸 Attendance Recognition System")
label_map = train_model()

option = st.radio("Select input mode:", ["📷 Webcam", "📁 Upload Image"])

if option == "📁 Upload Image":
    uploaded = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        result = detect_and_recognize(img, label_map)
        st.image(result, channels="BGR", caption="Processed Image")

else:
    run = st.checkbox("Start Webcam")
    frame_placeholder = st.empty()

    cap = None
    if run:
        cap = cv2.VideoCapture(0)

    while run and cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        result = detect_and_recognize(frame, label_map)
        frame_placeholder.image(result, channels="BGR")

    if cap:
        cap.release()

# View attendance
if st.button("📄 View Attendance"):
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df)
    else:
        st.info("No attendance data found.")
