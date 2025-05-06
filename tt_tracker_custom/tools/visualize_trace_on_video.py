import cv2
import numpy as np
from pathlib import Path

VIDEO_NAME = "videos_2"   # 替換為你的 video_name
CAMERA_ID = 2             # 使用 1 或 2，對應 videos_2_1.mp4 或 _2.mp4

# 載入影片和軌跡
video_path = Path("videos") / f"{VIDEO_NAME}_{CAMERA_ID}.mp4"
ball_path = Path("data") / VIDEO_NAME / f"ballpath{CAMERA_ID}.npy"

cap = cv2.VideoCapture(str(video_path))
ball_pos = np.load(str(ball_path))

trajectory_color = (0, 255, 255)  # 黃色
radius = 5
thickness = 2

# 建立一個記錄軌跡的 deque
trajectory = []

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(ball_pos):
        break

    x, y = ball_pos[frame_idx][:2]
    if x > 0 and y > 0:
        center = (int(x), int(y))
        trajectory.append(center)
        # 畫出球點
        cv2.circle(frame, center, radius, trajectory_color, -1)

    # 畫出軌跡線
    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i - 1], trajectory[i], trajectory_color, thickness)

    cv2.putText(frame, f"Frame: {frame_idx}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Ball Trace Overlay", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
