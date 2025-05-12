import numpy as np
import cv2
import speech_recognition as sr
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from ytmusicapi import YTMusic
import sys

# Paths to platform icons
icons = {
    "youtube": "C:/Users/Lenovo/Documents/face_noice_search/aipython/data/youtube.png",
    "facebook": "C:/Users/Lenovo/Documents/face_noice_search/aipython/data/facebook.png",
    "google": "C:/Users/Lenovo/Documents/face_noice_search/aipython/data/google.png",
    "instagram": "C:/Users/Lenovo/Documents/face_noice_search/aipython/data/instagram.png",
    "youtubemusic": "C:/Users/Lenovo/Documents/face_noice_search/aipython/data/spotify.png"
}

# Icon display positions
icon_positions = {
    "youtube": (50, 50),
    "facebook": (162, 50),
    "google": (275, 50),
    "instagram": (387, 50),
    "youtubemusic": (500, 50)
}

# Load face detection model (Haar for platform selection, DNN for age/gender)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_model = cv2.dnn.readNetFromCaffe(
    'faces_data/detection/deploy.prototxt',
    'faces_data/detection/res10_300x300_ssd_iter_140000.caffemodel'
)
face_blob_height = 300
face_average_color = (104, 177, 123)
face_confidence_threshold = 0.995

# Load age classification model's
age_model = cv2.dnn.readNetFromCaffe(
    'faces_data/age_gender_classification/age_net_deploy.prototxt',
    'faces_data/age_gender_classification/age_net.caffemodel'
)
age_labels = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']

# Load gender classification model
gender_model = cv2.dnn.readNetFromCaffe(
    'faces_data/age_gender_classification/gender_net_deploy.prototxt',
    'faces_data/age_gender_classification/gender_net.caffemodel'
)
gender_labels = ['male', 'female']

# Load average face image for age/gender preprocessing
age_gender_blob_size = (256, 256)
age_gender_average_image = np.load('faces_data/age_gender_classification/average_face.npy')

# Initialize YouTube Music API
ytmusic = YTMusic()

def recognize_speech(prompt="🎤 Hãy nói..."):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(prompt)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text = recognizer.recognize_google(audio, language="vi-VN").lower()
            text = text.replace(' ', '')  # Remove spaces
            print("✅ Bạn đã nói:", text)
            return text
        except:
            print("⚠️ Không thể nhận diện giọng nói.")
            return None

def overlay_icons(frame):
    for name, path in icons.items():
        icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if icon is None:
            continue

        x, y = icon_positions[name]
        icon = cv2.resize(icon, (80, 80))

        if y + 80 > frame.shape[0] or x + 80 > frame.shape[1]:
            continue

        if icon.shape[2] == 4:  # Handle PNG with alpha channel
            alpha_s = icon[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                frame[y:y + 80, x:x + 80, c] = (
                    alpha_s * icon[:, :, c] + alpha_l * frame[y:y + 80, x:x + 80, c]
                ).astype(np.uint8)
        else:
            frame[y:y + 80, x:x + 80] = icon
    return frame

def handle_platform(platform):
    platform = platform.replace(' ', '').lower()
    if platform == "youtube":
        return "🎶 Bạn muốn tìm video gì trên YouTube?"
    elif platform == "facebook":
        return "👤 Bạn muốn tìm ai trên Facebook?"
    elif platform == "youtubemusic":
        return "😊 Bạn đang cảm thấy thế nào? (vui, buồn, phấn khích, thư giãn...)"
    elif platform == "google":
        return "🔍 Bạn muốn tìm gì trên Google?"
    elif platform == "instagram":
        return "📸 Bạn muốn tìm ai trên Instagram?"
    else:
        print("⚠️ Nền tảng không được hỗ trợ.")
        return None

def get_url(platform, query):
    if platform == "youtube":
        return f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}" if query else "https://www.youtube.com"
    elif platform == "facebook":
        return f"https://www.facebook.com/search/top/?q={query.replace(' ', '%20')}" if query else "https://www.facebook.com"
    elif platform == "youtubemusic":
        return f"https://music.youtube.com/search?q={query.replace(' ', '+')}"
    elif platform == "google":
        return f"https://www.google.com/search?q={query.replace(' ', '+')}" if query else "https://www.google.com"
    elif platform == "instagram":
        return f"https://www.instagram.com/{query.replace(' ', '')}/" if query else "https://www.instagram.com"
    return None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nhận diện & Điều khiển bằng giọng nói")
        self.setGeometry(100, 100, 1000, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Không mở được webcam.")
            return

        self.browser = QWebEngineView()
        self.browser.setMinimumWidth(500)
        self.layout.addWidget(self.browser)

        self.selected_platform = None
        self.search_prompt = None

        self.update_frame()

    def update_frame(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Overlay platform icons
            frame = overlay_icons(frame)

            # Face detection for platform selection (using Haar Cascade)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            face_detected = len(faces) > 0

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Age and gender detection (using DNN)
            h, w = frame.shape[:2]
            aspect_ratio = w / h
            face_blob_width = int(face_blob_height * aspect_ratio)
            face_blob_size = (face_blob_width, face_blob_height)

            face_blob = cv2.dnn.blobFromImage(frame, size=face_blob_size, mean=face_average_color)
            face_model.setInput(face_blob)
            face_results = face_model.forward()

            for face in face_results[0, 0]:
                face_confidence = face[2]
                if face_confidence > face_confidence_threshold:
                    x0, y0, x1, y1 = (face[3:7] * [w, h, w, h]).astype(int)

                    # Make square ROI for age/gender
                    y1_roi = y0 + int(1.2 * (y1 - y0))
                    x_margin = ((y1_roi - y0) - (x1 - x0)) // 2
                    x0_roi = x0 - x_margin
                    x1_roi = x1 + x_margin

                    if x0_roi < 0 or x1_roi > w or y0 < 0 or y1_roi > h:
                        continue

                    age_gender_roi = frame[y0:y1_roi, x0_roi:x1_roi]
                    scaled_roi = cv2.resize(age_gender_roi, age_gender_blob_size).astype(np.float32)
                    scaled_roi -= age_gender_average_image

                    age_gender_blob = cv2.dnn.blobFromImage(scaled_roi, size=age_gender_blob_size)

                    # Age prediction
                    age_model.setInput(age_gender_blob)
                    age_results = age_model.forward()
                    age_id = np.argmax(age_results)
                    age_label = age_labels[age_id]
                    age_conf = age_results[0, age_id]

                    # Gender prediction
                    gender_model.setInput(age_gender_blob)
                    gender_results = gender_model.forward()
                    gender_id = np.argmax(gender_results)
                    gender_label = gender_labels[gender_id]
                    gender_conf = gender_results[0, gender_id]

                    # Draw age/gender results
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                    cv2.rectangle(frame, (x0_roi, y0), (x1_roi, y1_roi), (0, 255, 255), 2)
                    label = f"{gender_label} ({gender_conf*100:.1f}%), {age_label} ({age_conf*100:.1f}%)"
                    cv2.putText(frame, label, (x0_roi, y0 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Webcam - Face, Age, Gender Detection", frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if face_detected:
                    self.selected_platform = recognize_speech(
                        "🎤 Hãy nói tên nền tảng bạn muốn chọn (youtube, facebook, google, instagram, youtubemusic)...")
                    if self.selected_platform:
                        if self.selected_platform in icons or self.selected_platform.replace(' ', '').lower() in icons:
                            print(f"🌐 Đã chọn {self.selected_platform}")
                            self.search_prompt = handle_platform(self.selected_platform)
                        else:
                            print("❌ Không tìm thấy nền tảng phù hợp. Hãy thử lại.")
                    else:
                        print("⚠️ Bạn chưa chọn nền tảng.")
                else:
                    print("⚠️ Vui lòng để khuôn mặt xuất hiện trước webcam để chọn nền tảng.")
            elif key == ord('f') and self.selected_platform:
                if face_detected:
                    if self.search_prompt:
                        query = recognize_speech(self.search_prompt)
                        url = get_url(self.selected_platform, query)
                        if url:
                            self.browser.setUrl(QUrl(url))
                            print(f"🌐 Đã truy cập: {url}")
                        else:
                            print("⚠️ Không tìm thấy nội dung phù hợp.")
                else:
                    print("⚠️ Vùi lòng để khuôn mặt xuất hiện trước webcam để tìm kiếm.")

        self.cap.release()
        cv2.destroyAllWindows()

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())