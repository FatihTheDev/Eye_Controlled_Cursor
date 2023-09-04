import cv2
import mediapipe as mp
import pyautogui

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# make camera feed twice as large
screen_w *= 2
screen_h *= 2

cap = cv2.VideoCapture(0)

while 1:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb)

    landmark_points = result.multi_face_landmarks

    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))

            if id == 1:
                screen_x = int(landmark.x * frame_w)
                screen_y = int(landmark.y * frame_h)
                pyautogui.moveTo(screen_x, screen_y)

        right = [landmarks[145], landmarks[159]]
        for landmark in right:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 0, 255))
        if(right[0].y - right[1].y) < 0.009:
            pyautogui.click()

    # Resize the frame to the desired dimensions
    frame = cv2.resize(frame, (screen_w, screen_h))

    cv2.imshow("Touchless Mouse", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

cap.release()
cv2.destroyAllWindows()
