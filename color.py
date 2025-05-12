from random import random

import numpy as np
import cv2
import speech_recognition as sr
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from ytmusicapi import YTMusic
import threading
import sys


possible_emotions = ["hạnh phúc", "buồn", "giận dữ", "ngạc nhiên", "sợ hãi", "khinh thường", "vui"]
# Đường dẫn các logo
icons = {
    "youtube": "C:/Users/Lenovo/Documents/face_noice_search/aipython/data/youtube.png",
    "facebook": "C:/Users/Lenovo/Documents/face_noice_search/aipython/data/facebook.png",
    "google": "C:/Users/Lenovo/Documents/face_noice_search/aipython/data/google.png",
    "instagram": "C:/Users/Lenovo/Documents/face_noice_search/aipython/data/instagram.png",
    "youtubemusic": "C:/Users/Lenovo/Documents/face_noice_search/aipython/data/spotify.png"
}

# Tọa độ hiển thị các logo
icon_positions = {
    "youtube": (50, 50),
    "facebook": (162, 50),
    "google": (275, 50),
    "instagram": (387, 50),
    "youtubemusic": (500, 50)
}

# Tải mô hình nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

ytmusic = YTMusic()


def recognize_speech(prompt="🎤 Hãy nói..."):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(prompt)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text = recognizer.recognize_google(audio, language="vi-VN").lower()
            text = text.replace(' ', '')
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
        if icon.shape[2] == 4:
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
        return None  # Sử dụng cảm xúc
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


def detect_emotion_thread(result_holder):
    from fer import FER  # Import tại đây để tránh lỗi DLL của TensorFlow
    detector = FER()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = detector.detect_emotions(frame)
        if results:
            emotion, score = detector.top_emotion(frame)
            if emotion:
                result_holder.append(emotion)
                break
    cap.release()


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

            frame = overlay_icons(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            face_detected = len(faces) > 0

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Webcam", frame)

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
                    if self.selected_platform == "youtubemusic":
                        print("😊 Đang nhận diện cảm xúc...")
                        emotion_result = []
                        t = threading.Thread(target=detect_emotion_thread, args=(emotion_result,))
                        t.start()
                        t.join()
                        emotion = emotion_result[0] if emotion_result else random.choice(possible_emotions)
                        url = get_url(self.selected_platform, emotion)
                        if url:
                            self.browser.setUrl(QUrl(url))
                            print(f"🎵 Tìm nhạc phù hợp với cảm xúc: {emotion}")
                        else:
                            print("⚠️ Không tìm thấy nội dung phù hợp.")
                    elif self.search_prompt:
                        query = recognize_speech(self.search_prompt)
                        url = get_url(self.selected_platform, query)
                        if url:
                            self.browser.setUrl(QUrl(url))
                            print(f"🌐 Đã truy cập: {url}")
                        else:
                            print("⚠️ Không tìm thấy nội dung phù hợp.")
                else:
                    print("⚠️ Vui lòng để khuôn mặt xuất hiện trước webcam để tìm kiếm.")

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