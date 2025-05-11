import cv2
import numpy as np
from pathlib import Path
import time
import argparse

def on_trackbar(val):
    pass

def main(args):
    # 讀取影片路徑
    video1_path = Path("videos") / f"{args.video_name}_1.mp4"
    video2_path = Path("videos") / f"{args.video_name}_2.mp4"
    
    # 讀取球軌跡與速度數據
    ballpath1 = np.load(f"data/{args.video_name}/ballpath1.npy")
    ballpath2 = np.load(f"data/{args.video_name}/ballpath2.npy")
    speed3d = np.load(f"data/{args.video_name}/speed3d.npy")  # 3D球速數據

    # 開啟兩個影片
    cap1 = cv2.VideoCapture(str(video1_path))
    cap2 = cv2.VideoCapture(str(video2_path))

    # 取得影片的 FPS 和總幀數
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Camera 1 FPS: {fps1}, Total Frames: {total_frames1}")
    print(f"Camera 2 FPS: {fps2}, Total Frames: {total_frames2}")

    # 設定目標 FPS 為 Camera 1 的 FPS
    target_fps = fps1
    frame_time = 1.0 / target_fps
    print(f"Using target FPS: {target_fps}")

    # 設定滑桿視窗
    cv2.namedWindow("Settings")
    cv2.createTrackbar("xmin1", "Settings", 0, 1920, on_trackbar)
    cv2.createTrackbar("xmax1", "Settings", 1920, 1920, on_trackbar)
    cv2.createTrackbar("ymin1", "Settings", 0, 1080, on_trackbar)
    cv2.createTrackbar("ymax1", "Settings", 1080, 1080, on_trackbar)
    cv2.createTrackbar("xmin2", "Settings", 0, 1920, on_trackbar)
    cv2.createTrackbar("xmax2", "Settings", 1920, 1920, on_trackbar)
    cv2.createTrackbar("ymin2", "Settings", 0, 1080, on_trackbar)
    cv2.createTrackbar("ymax2", "Settings", 1080, 1080, on_trackbar)

    # 軌跡與顯示設定
    traj_color = (0, 255, 255)  # 軌跡顏色 (黃色)
    text_color = (255, 255, 255)  # 文字顏色 (白色)
    speed_color = (0, 255, 0)  # 球速顯示顏色 (綠色)
    radius = 5  # 球點半徑
    max_trajectory_length = 30  # 軌跡長度限制
    trajectory1, trajectory2 = [], []  # 存儲球軌跡

    frame_idx = 0
    start_time = time.time()
    last_frame_time = start_time
    frame_count = 0

    while cap1.isOpened() and cap2.isOpened():
        # 從滑桿讀取範圍設定
        xmin1 = cv2.getTrackbarPos("xmin1", "Settings")
        xmax1 = cv2.getTrackbarPos("xmax1", "Settings")
        ymin1 = cv2.getTrackbarPos("ymin1", "Settings")
        ymax1 = cv2.getTrackbarPos("ymax1", "Settings")
        xmin2 = cv2.getTrackbarPos("xmin2", "Settings")
        xmax2 = cv2.getTrackbarPos("xmax2", "Settings")
        ymin2 = cv2.getTrackbarPos("ymin2", "Settings")
        ymax2 = cv2.getTrackbarPos("ymax2", "Settings")
        
        # 計算目標時間
        target_time = start_time + frame_count * frame_time
        current_time = time.time()
        
        # 如果還沒到下一幀的時間，就等待
        if current_time < target_time:
            time.sleep(target_time - current_time)
        
        # 設定兩個影片到相同的幀位置
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2 or frame_idx >= len(ballpath1):
            break

        # 取得球的座標
        x1, y1 = ballpath1[frame_idx][:2]
        x2, y2 = ballpath2[frame_idx][:2]

        # Camera 1 偵測範圍
        has1 = (xmin1 < x1 < xmax1) and (ymin1 < y1 < ymax1)
        # Camera 2 偵測範圍
        has2 = (xmin2 < x2 < xmax2) and (ymin2 < y2 < ymax2)

        # 更新軌跡
        if has1:
            trajectory1.append((int(x1), int(y1)))
            if len(trajectory1) > max_trajectory_length:
                trajectory1.pop(0)
        if has2:
            trajectory2.append((int(x2), int(y2)))
            if len(trajectory2) > max_trajectory_length:
                trajectory2.pop(0)

        # 繪製軌跡
        for i in range(1, len(trajectory1)):
            cv2.line(frame1, trajectory1[i-1], trajectory1[i], traj_color, 2)
        for i in range(1, len(trajectory2)):
            cv2.line(frame2, trajectory2[i-1], trajectory2[i], traj_color, 2)

        # 顯示球速
        if frame_idx < len(speed3d):
            speed = speed3d[frame_idx]
            speed_text = f"Speed: {speed:.1f} m/s"
            color = speed_color if speed > 0 else (0, 0, 255)  # 紅色表示速度為0
            cv2.putText(frame1, speed_text, (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame2, speed_text, (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 畫出偵測區域
        cv2.rectangle(frame1, (xmin1, ymin1), (xmax1, ymax1), (255, 0, 0), 2)
        cv2.rectangle(frame2, (xmin2, ymin2), (xmax2, ymax2), (255, 0, 0), 2)

        # 合併兩個影片
        combined = cv2.vconcat([frame1, frame2])
        cv2.imshow("Dual Camera Trace Annotated", combined)

        # 檢查是否按下 ESC
        if cv2.waitKey(1) == 27:  # ESC 退出
            break

        frame_idx += 1
        frame_count += 1

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", required=True, help="Name of the video set (e.g., videos_2, videos_3)")
    args = parser.parse_args()
    main(args)
