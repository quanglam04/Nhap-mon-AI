import cv2
from  fer import FER

# Khởi tạo detector cảm xúc
detector = FER()

# Mở camera laptop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Nhận diện cảm xúc
    result = detector.detect_emotions(frame)
    for face in result:
        (x, y, w, h) = face["box"]
        emotion, score = detector.top_emotion(frame)

        # Kiểm tra và hiển thị cảm xúc và độ chính xác
        if score is not None:
            cv2.putText(frame, f'{emotion} ({score:.2f})', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        else:
            cv2.putText(frame, f'{emotion}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Vẽ khung xung quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Hiển thị hình ảnh
    cv2.imshow("Emotion Detection", frame)

    # Bấm 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()