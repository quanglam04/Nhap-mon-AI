# Face & Voice Search Application

## Giới thiệu

Ứng dụng **Face & Voice Search** là một dự án sáng tạo cho phép người dùng tương tác với các nền tảng trực tuyến (YouTube, Facebook, Google, Instagram, YouTube Music) thông qua nhận diện khuôn mặt và giọng nói. Ứng dụng sử dụng webcam để phát hiện khuôn mặt, nhận diện cảm xúc và điều khiển tìm kiếm bằng lệnh thoại, mang lại trải nghiệm người dùng độc đáo và hiện đại.

Dự án được phát triển bởi **Vương Đức Trọng**.

## Chức năng

Ứng dụng cung cấp các tính năng chính sau:

- **Nhận diện khuôn mặt**: Sử dụng OpenCV để phát hiện khuôn mặt trong thời gian thực qua webcam.
- **Nhận diện cảm xúc**: Tích hợp thư viện FER để phân tích cảm xúc (hạnh phúc, buồn, giận dữ, ngạc nhiên, sợ hãi, khinh thường, vui) và đề xuất nội dung phù hợp, đặc biệt trên YouTube Music.
- **Điều khiển bằng giọng nói**: Hỗ trợ nhận diện giọng nói tiếng Việt thông qua SpeechRecognition, cho phép người dùng chọn nền tảng và nhập truy vấn tìm kiếm.
- **Hiển thị biểu tượng nền tảng**: Các biểu tượng của YouTube, Facebook, Google, Instagram và YouTube Music được phủ lên khung hình webcam để người dùng dễ dàng chọn nền tảng.
- **Tìm kiếm trực tuyến**: Tự động mở trình duyệt với các truy vấn tìm kiếm trên nền tảng được chọn.
- **Giao diện thân thiện**: Sử dụng PyQt5 để tạo giao diện đồ họa mượt mà và tích hợp trình duyệt web.

## Công nghệ sử dụng

Dự án sử dụng các công nghệ và thư viện sau:

- **Python**: Ngôn ngữ lập trình chính.
- **OpenCV**: Xử lý hình ảnh và nhận diện khuôn mặt.
- **FER (Facial Expression Recognition)**: Nhận diện cảm xúc trên khuôn mặt.
- **SpeechRecognition**: Nhận diện và xử lý giọng nói.
- **PyQt5 & PyQtWebEngine**: Tạo giao diện đồ họa và tích hợp trình duyệt web.
- **YTMusicAPI**: Tích hợp tìm kiếm nhạc trên YouTube Music.
- **NumPy**: Xử lý dữ liệu số và ma trận.
- **TensorFlow**: Hỗ trợ mô hình học sâu cho nhận diện cảm xúc.
- **MoviePy**: Xử lý video (phiên bản 1.0.3).
- **PyAudio**: Hỗ trợ thu âm từ microphone.
- **Pillow**: Xử lý hình ảnh.

## Cài đặt và khởi chạy

### Yêu cầu hệ thống

- Python 3.8 hoặc cao hơn.
- Webcam hoạt động.
- Microphone để nhận diện giọng nói.
- Kết nối Internet để truy cập các nền tảng trực tuyến.

### Hướng dẫn cài đặt

1. **Clone repository về máy**:
   ```bash
   git clone <URL_repository>
   cd <tên_thư_mục>
   ```
2. Tạo môi trường ảo
   ```
   python -m venv .venv
   ```
3. Kích hoạt môi trường ảo
-  Trên Windows
   ```
   .venv\Scripts\activate.bat
   ```
-  Trên macOS/Linux
   ```
   source .venv/bin/activate
   ```
4. Cài đặt các thư viện cần thiết
   ```
    pip install numpy sounddevice speechrecognition
    pip install opencv-python
    pip install pyaudio
    pip install pillow
    pip install PyQt5 PyQtWebEngine
    pip install opencv-contrib-python ytmusicapi
    pip install fer moviepy==1.0.3 tensorflow
   ```
5. Chạy
   ```
   python main.py
   ```
