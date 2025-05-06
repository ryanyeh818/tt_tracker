import cv2
import numpy as np
from pathlib import Path

VIDEO_NAME = "videos_2"
video1_path = Path("videos") / f"{VIDEO_NAME}_1.mp4"
video2_path = Path("videos") / f"{VIDEO_NAME}_2.mp4"
ballpath1 = np.load(f"data/{VIDEO_NAME}/ballpath1.npy")
ballpath2 = np.load(f"data/{VIDEO_NAME}/ballpath2.npy")
speed3d = np.load(f"data/{VIDEO_NAME}/speed3d.npy")

cap1 = cv2.VideoCapture(str(video1_path))
cap2 = cv2.VideoCapture(str(video2_path))
fps = cap1.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

traj_color = (0, 255, 255)
font = cv2.FONT_HERSHEY_SIMPLEX
radius = 5
trajectory1, trajectory2 = [], []

frame_idx = 0
while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2 or frame_idx >= len(ballpath1):
        break

    # camera 1
    x1, y1 = ballpath1[frame_idx][:2]
    if x1 > 0 and y1 > 0:
        pt1 = (int(x1), int(y1))
        trajectory1.append(pt1)
        cv2.circle(frame1, pt1, radius, traj_color, -1)
        for i in range(1, len(trajectory1)):
            cv2.line(frame1, trajectory1[i - 1], trajectory1[i], traj_color, 2)

    # camera 2
    x2, y2 = ballpath2[frame_idx][:2]
    if x2 > 0 and y2 > 0:
        pt2 = (int(x2), int(y2))
        trajectory2.append(pt2)
        cv2.circle(frame2, pt2, radius, traj_color, -1)
        for i in range(1, len(trajectory2)):
            cv2.line(frame2, trajectory2[i - 1], trajectory2[i], traj_color, 2)

    # 加上速度資訊
    if frame_idx < len(speed3d):
        speed_text = f"3D Speed: {speed3d[frame_idx]:.1f} units/s"
        cv2.putText(frame1, speed_text, (40, 60), font, 0.8, (255, 255, 255), 2)
        cv2.putText(frame2, speed_text, (40, 60), font, 0.8, (255, 255, 255), 2)

    cv2.putText(frame1, f"Frame: {frame_idx}", (40, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(frame2, f"Frame: {frame_idx}", (40, 30), font, 0.8, (255, 255, 255), 2)

    combined = cv2.vconcat([frame1, frame2])
    cv2.imshow("Dual View with 3D Speed", combined)
    if cv2.waitKey(delay) == 27:
        break

    frame_idx += 1

cap1.release()
cap2.release()
cv2.destroyAllWindows()
