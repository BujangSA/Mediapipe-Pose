import cv2
import mediapipe as mp

mpose = mp.solutions.pose   # inisialisasi mediapipe pose
pose = mpose.Pose()

cap = cv2.VideoCapture(0)   # video dari webcam

while True:
    success, img = cap.read()   # pembacaan image
    if not success:
        break

    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # konversi warna BGR ke RGB
    hasil = pose.process(imgrgb)   # ekstraksi pose

    if hasil.pose_landmarks:
        print("terdeteksi")
    else:
        print("tidak terdeteksi")

    cv2.imshow("webcam", img)   # tampilkan video

    if cv2.waitKey(1) & 0xFF == ord('q'):   # tekan q untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
