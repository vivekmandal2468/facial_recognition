import streamlit as st
import face_recognition
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Paths
KNOWN_FACES_DIR = 'data/known_faces'
ATTENDANCE_FILE = 'data/attendance.csv'

# Utility: Load known face encodings
def load_known_faces():
    known_encodings = []
    known_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(('.jpg', '.png')):
            image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{filename}")
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_encodings.append(encoding[0])
                known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names

# Utility: Mark attendance in CSV
def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    if not os.path.exists(ATTENDANCE_FILE):
        pd.DataFrame(columns=['Name', 'Date', 'Time']).to_csv(ATTENDANCE_FILE, index=False)

    df = pd.read_csv(ATTENDANCE_FILE)
    if not ((df['Name'] == name) & (df['Date'] == date_str)).any():
        new_entry = pd.DataFrame([[name, date_str, time_str]], columns=['Name', 'Date', 'Time'])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)

# Streamlit App
st.set_page_config(page_title="Attendance System", layout="wide")
st.title("📸 Attendance Recognition System")

tab1, tab2 = st.tabs(["📂 Upload Faces", "🟢 Start Recognition"])

with tab1:
    st.header("Upload Known Faces")
    uploaded_files = st.file_uploader("Upload Images (Name the files as the person's name)", type=['jpg', 'png'], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            with open(f"{KNOWN_FACES_DIR}/{file.name}", 'wb') as f:
                f.write(file.getbuffer())
        st.success("✅ Faces uploaded successfully!")

with tab2:
    st.header("Live Face Recognition")
    start = st.button("Start Webcam")

    if start:
        stframe = st.empty()
        video = cv2.VideoCapture(0)

        known_encodings, known_names = load_known_faces()

        while True:
            ret, frame = video.read()
            if not ret:
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    mark_attendance(name)

                    top, right, bottom, left = face_location
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels='RGB')

        video.release()

st.sidebar.header("📅 Attendance Log")
if os.path.exists(ATTENDANCE_FILE):
    df = pd.read_csv(ATTENDANCE_FILE)
    st.sidebar.dataframe(df)
else:
    st.sidebar.info("No attendance data yet.")
