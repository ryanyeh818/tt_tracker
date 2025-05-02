import cv2
import numpy as np
from pathlib import Path
from extended_yolo_v3 import pre_process, post_process

PIXELS_PER_CM = 5.0  # 可依實際場地調整
VIDEO_NAME = "videos_2"
CAMERA_ID = 1

video_path = Path("videos") / f"{VIDEO_NAME}_{CAMERA_ID}.mp4"
ballpath = []
speeds = []

# 載入模型
modelWeights = "weights.onnx"
net = cv2.dnn.readNet(modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 讀入影片
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
ret, frame = cap.read()
frame_idx = 0
prev_center = None
prev_time = 0

while ret:
    detections = pre_process(frame, net)
    boxes, confidences, class_ids = post_process(frame, detections)

    ball_center = None
    if boxes:
        box = boxes[0]
        left, top, width, height = box
        cx = int(left + width / 2)
        cy = int(top + height / 2)
        ball_center = (cx, cy)
        ballpath.append([cx, cy, 1])
    else:
        ballpath.append([0, 0, 0])

    # 計算速度
    current_time = frame_idx / fps
    if ball_center and prev_center:
        dx = ball_center[0] - prev_center[0]
        dy = ball_center[1] - prev_center[1]
        dist_px = np.sqrt(dx ** 2 + dy ** 2)
        dist_cm = dist_px / PIXELS_PER_CM
        dt = current_time - prev_time if current_time > prev_time else 1 / fps
        speed_cm_s = dist_cm / dt
        speeds.append(speed_cm_s)
        cv2.putText(frame, f"Speed: {speed_cm_s:.1f} cm/s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    else:
        speeds.append(0)

    prev_center = ball_center
    prev_time = current_time

    if ball_center:
        cv2.circle(frame, ball_center, 5, (0, 255, 255), -1)

    cv2.putText(frame, f"Frame: {frame_idx}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Tracking + Speed", frame)

    key = cv2.waitKey(int(1000/fps))
    if key == 27:
        break

    ret, frame = cap.read()
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# 儲存結果
ballpath = np.array(ballpath)
np.save(f"data/{VIDEO_NAME}/ballpath{CAMERA_ID}.npy", ballpath)
np.save(f"data/{VIDEO_NAME}/speed{CAMERA_ID}.npy", np.array(speeds))
