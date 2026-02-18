import cv2
import mediapipe as mp

mpose = mp.solutions.pose
mdraw = mp.solutions.drawing_utils

pose = mpose.Pose()
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hasil = pose.process(imgRGB)

    if hasil.pose_landmarks:
        mdraw.draw_landmarks(
            img,
            hasil.pose_landmarks,
            mpose.POSE_CONNECTIONS
        )

        # Ambil landmark
        landmarks = hasil.pose_landmarks.landmark

        # Koordinat tangan dan bahu
        right_wrist = landmarks[mpose.PoseLandmark.RIGHT_WRIST]
        right_shoulder = landmarks[mpose.PoseLandmark.RIGHT_SHOULDER]

        left_wrist = landmarks[mpose.PoseLandmark.LEFT_WRIST]
        left_shoulder = landmarks[mpose.PoseLandmark.LEFT_SHOULDER]

        h, w, c = img.shape

        # Konversi ke pixel
        rw_y = int(right_wrist.y * h)
        rs_y = int(right_shoulder.y * h)

        lw_y = int(left_wrist.y * h)
        ls_y = int(left_shoulder.y * h)

        # Deteksi tangan kanan
        if rw_y < rs_y:
            cv2.putText(img, "Tangan Kanan Terangkat", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Deteksi tangan kiri
        if lw_y < ls_y:
            cv2.putText(img, "Tangan Kiri Terangkat", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Pose Landmarks", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
