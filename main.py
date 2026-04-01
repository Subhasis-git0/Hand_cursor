import cv2
import mediapipe as mp
import pyautogui
import math
import time

pinch_start_time = 0
dragging = False
pinch_active = False

last_click_time = 0

screen_w, screen_h = pyautogui.size()

#disable pyautogui failsafe
pyautogui.FAILSAFE = False

model_path = "hand_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

landmarker = HandLandmarker.create_from_options(options)

prev_x, prev_y = 0, 0

cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        print("Hand detected")
        for hand_landmarks in result.hand_landmarks:
            
            thumb = hand_landmarks[4]
            index_finger = hand_landmarks[8]

            h, w, _ = frame.shape
            cx = int(index_finger.x * w)
            cy = int(index_finger.y * h)

            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
            index_x, index_y = int(index_finger.x * w), int(index_finger.y * h)

            distance = math.hypot(index_x - thumb_x, index_y - thumb_y)

            current_time = time.time()
            if distance < 30:
                if not pinch_active:
                    pinch_start_time = current_time
                    pinch_active = True
                
                elif current_time - pinch_start_time > 0.5:
                    if not dragging:
                        pyautogui.mouseDown()
                        dragging = True
            else:
                if pinch_active:
                    if not dragging:
                        pyautogui
                    else:
                        pyautogui.mouseUp()
                        dragging = False
                    pinch_active = False


            #map camera coords to screen coords; 1.2 is an optional scaling factor to make the cursor movement more responsive
            screen_x = int(index_finger.x * screen_w * 1.2)
            screen_y = int(index_finger.y * screen_h * 1.2)

            screen_x = max(0, min(screen_w - 1, screen_x))
            screen_y = max(0, min(screen_h - 1, screen_y))

            smoothening = 6
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            try:
                pyautogui.moveTo(curr_x, curr_y)
            except pyautogui.FailSafeException:
                pass

            prev_x, prev_y = curr_x, curr_y

    cv2.imshow("camera", frame)

    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()

