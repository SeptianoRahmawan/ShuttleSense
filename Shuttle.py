import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st

# === Konstanta ===
PETUNJUK = [
    "1. Posisi tubuh serong (kaki kiri depan).",
    "2. Berat di kaki belakang (lutut ditekuk).",
    "3. Raket di depan pinggang (forehand grip).",
    "4. Shuttlecock di depan paha dan setinggi pinggang."
]
FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 0.7  # Diperkecil agar muat
THICKNESS = 1

# === Load model ===
loaded_model = joblib.load("lgbm_classifier.pkl")

# === Mediapipe setup ===
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# === Streamlit UI ===
st.title("ShuttleSense")

# Session state untuk kontrol kamera
if "run" not in st.session_state:
    st.session_state.run = False

col1, col2 = st.columns(2)
with col1:
    if st.button("Start Camera"):
        st.session_state.run = True
with col2:
    if st.button("Stop Camera"):
        st.session_state.run = False

FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.write("Kamera tidak terdeteksi.")
            break

        # Convert warna ke RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # Ambil koordinat pose
        koordinat_pose = []
        prediksi = -1  # Default value
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                koordinat_pose.append(lm.x)
                koordinat_pose.append(lm.y)
                koordinat_pose.append(lm.z)

            # Prediksi model
            prediksi = loaded_model.predict(np.array(koordinat_pose).reshape((1, -1)))[0]

            if prediksi == 0:
                text = "BENAR!"
                color = (0, 255, 0)
            elif prediksi == 1:
                text = "SALAH!"
                color = (0, 0, 255)
            else:
                text = "GATAU!"
                color = (255, 255, 0)
        else:
            text = "Tidak Terdeteksi"
            color = (0, 0, 255)

        # Visualisasi
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
            )

        # Tambah teks hasil
        cv2.putText(image, text, (10, 30), FONT, SCALE, color, THICKNESS, cv2.LINE_AA)

        # Tambah petunjuk hanya jika prediksi bukan 'BENAR' (prediksi != 0)
        if prediksi != 0:
            y_offset = 60
            for i, line in enumerate(PETUNJUK):
                cv2.putText(image, line, (10, y_offset + i * 25), FONT, SCALE, (0, 255, 0), THICKNESS, cv2.LINE_AA)

        # Tampilkan di Streamlit
        FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

cap.release()