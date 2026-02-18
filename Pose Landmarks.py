import cv2
import mediapipe as mp

mpose = mp.solutions.pose
mdraw = mp.solutions.drawing_utilsq

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

    cv2.imshow("Pose Landmarks", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
