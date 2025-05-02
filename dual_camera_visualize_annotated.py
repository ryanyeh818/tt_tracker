import cv2
import numpy as np
from pathlib import Path
import time

VIDEO_NAME = "videos_2"
video1_path = Path("videos") / f"{VIDEO_NAME}_1.mp4"
video2_path = Path("videos") / f"{VIDEO_NAME}_2.mp4"
ballpath1 = np.load(f"data/{VIDEO_NAME}/ballpath1.npy")
ballpath2 = np.load(f"data/{VIDEO_NAME}/ballpath2.npy")

# 開啟影片
cap1 = cv2.VideoCapture(str(video1_path))
cap2 = cv2.VideoCapture(str(video2_path))

# 取得影片的 FPS
fps1 = cap1.get(cv2.CAP_PROP_FPS)
fps2 = cap2.get(cv2.CAP_PROP_FPS)
print(f"Camera 1 FPS: {fps1}")
print(f"Camera 2 FPS: {fps2}")

# 設定目標 FPS
target_fps = 30
frame_time = 1.0 / target_fps
print(f"Using target FPS: {target_fps}")

traj_color = (0, 255, 255)
text_color = (255, 255, 255)
radius = 5
trajectory1, trajectory2 = [], []

frame_idx = 0
start_time = time.time()
last_frame_time = start_time
frame_count = 0

while cap1.isOpened() and cap2.isOpened():
    # 計算需要等待的時間
    current_time = time.time()
    elapsed = current_time - last_frame_time
    if elapsed < frame_time:
        time.sleep(frame_time - elapsed)
    
    # 設定兩個影片到相同的幀位置
    cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2 or frame_idx >= len(ballpath1):
        break

    x1, y1 = ballpath1[frame_idx][:2]
    x2, y2 = ballpath2[frame_idx][:2]

    has1 = x1 > 0 and y1 > 0
    has2 = x2 > 0 and y2 > 0

    if has1:
        pt1 = (int(x1), int(y1))
        trajectory1.append(pt1)
        cv2.circle(frame1, pt1, radius, traj_color, -1)
        for j in range(1, len(trajectory1)):
            cv2.line(frame1, trajectory1[j - 1], trajectory1[j], traj_color, 2)
        cv2.putText(frame1, f"Ball: ({int(x1)}, {int(y1)})", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    else:
        cv2.putText(frame1, "No Ball Detected", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if has2:
        pt2 = (int(x2), int(y2))
        trajectory2.append(pt2)
        cv2.circle(frame2, pt2, radius, traj_color, -1)
        for j in range(1, len(trajectory2)):
            cv2.line(frame2, trajectory2[j - 1], trajectory2[j], traj_color, 2)
        cv2.putText(frame2, f"Ball: ({int(x2)}, {int(y2)})", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    else:
        cv2.putText(frame2, "No Ball Detected", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 計算並顯示實際 FPS
    frame_count += 1
    total_elapsed = time.time() - start_time
    actual_fps = frame_count / total_elapsed if total_elapsed > 0 else 0

    cv2.putText(frame1, f"Camera 1 - Frame {frame_idx}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    cv2.putText(frame1, f"FPS: {actual_fps:.1f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    cv2.putText(frame2, f"Camera 2 - Frame {frame_idx}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    cv2.putText(frame2, f"FPS: {actual_fps:.1f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    combined = cv2.vconcat([frame1, frame2])
    cv2.imshow("Dual Camera Trace Annotated", combined)

    if cv2.waitKey(1) == 27:  # 使用較小的等待時間
        break

    frame_idx += 1
    last_frame_time = time.time()

cap1.release()
cap2.release()
cv2.destroyAllWindows()
