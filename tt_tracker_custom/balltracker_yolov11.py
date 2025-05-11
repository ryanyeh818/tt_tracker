import numpy as np
import cv2
import time
import os
import argparse
from pathlib import Path
from ultralytics import YOLO
import platform
from tqdm import tqdm

def detect_ballpath(video_path, model, show_video=False):
    """使用 YOLOv11 模型檢測影片中的球
    
    Args:
        video_path: 影片路徑
        model: YOLOv11 模型
        show_video: 是否顯示檢測過程
    
    Returns:
        height: 影片高度
        width: 影片寬度
        fps: 影片幀率
        ballpath: 球的位置數據 (x, y, 1)
    """
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ballpath = np.zeros((frame_count, 3))  # x, y, 1
    trajectory = []  # 用於儲存球的軌跡
    detection_count = 0  # 計數成功檢測的幀數
    total_boxes = 0  # 計數所有檢測到的框
    max_conf = 0  # 記錄最高信心度
    min_conf = 1  # 記錄最低信心度

    # 使用 tqdm 創建進度條
    pbar = tqdm(total=frame_count, desc="Processing frames", leave=True)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 使用 YOLOv11 進行檢測，降低信心度閾值
        results = model(frame, conf=0.1, verbose=False)  # 降低信心度閾值到 0.1
        
        # 處理檢測結果
        if len(results[0].boxes) > 0:
            total_boxes += len(results[0].boxes)
            # 取第一個檢測到的球（信心最高）
            box = results[0].boxes[0].xyxy[0].cpu().numpy()  # 取得邊界框座標
            conf = float(results[0].boxes[0].conf[0])
            max_conf = max(max_conf, conf)
            min_conf = min(min_conf, conf)
            
            center_x = int((box[0] + box[2]) / 2)  # 計算中心點 x 座標
            center_y = int((box[1] + box[3]) / 2)  # 計算中心點 y 座標
            ballpath[frame_idx] = [center_x, center_y, 1]
            detection_count += 1
            
            # 更新軌跡
            trajectory.append((center_x, center_y))
            if len(trajectory) > 30:  # 只保留最近 30 幀的軌跡
                trajectory.pop(0)
        else:
            ballpath[frame_idx] = [0, 0, 0]

        if show_video:
            # 繪製檢測框和軌跡
            if len(results[0].boxes) > 0:
                # 繪製邊界框
                cv2.rectangle(frame, 
                            (int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), 
                            (0, 255, 0), 2)
                
                # 繪製中心點
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # 繪製軌跡
                for i in range(1, len(trajectory)):
                    cv2.line(frame, trajectory[i-1], trajectory[i], (255, 0, 0), 2)

            # 顯示幀數
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 顯示信心度
            if len(results[0].boxes) > 0:
                cv2.putText(frame, f"Conf: {conf:.2f}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Ball Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1
        pbar.update(1)  # 更新進度條

    pbar.close()  # 關閉進度條
    cap.release()
    if show_video:
        cv2.destroyAllWindows()

    # 輸出檢測統計資訊
    print(f"\nDetection Statistics:")
    print(f"Total frames: {frame_count}")
    print(f"Detected frames: {detection_count}")
    print(f"Detection rate: {(detection_count/frame_count)*100:.2f}%")
    print(f"Total boxes detected: {total_boxes}")
    print(f"Average boxes per detection: {total_boxes/detection_count if detection_count > 0 else 0:.2f}")
    print(f"Confidence range: {min_conf:.2f} - {max_conf:.2f}")

    return height, width, fps, ballpath

def main(args):
    """處理兩個同步影片，使用 YOLOv11 檢測乒乓球位置
    
    Args:
        args: 命令列參數，包含 video_name 和 flip 設定
    """
    vidname = args.video_name
    flipped1 = args.flipped_1
    flipped2 = args.flipped_2

    # 確保輸出目錄存在
    data_dir = Path(__file__).parent / "data" / vidname
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 設定影片路徑
    vid1_path = Path(__file__).parent / "videos" / f"{vidname}_1.mp4"
    vid2_path = Path(__file__).parent / "videos" / f"{vidname}_2.mp4"

    # 載入 YOLOv11 模型
    model = YOLO('last_train.pt')  # 使用您的訓練權重

    # 處理第一個影片
    print("\nProcessing video 1...")
    t1 = time.time()
    height1, width1, fps1, ball_pos1 = detect_ballpath(vid1_path, model, show_video=False)  # 關閉影片顯示以提高處理速度
    t2 = time.time()
    param1 = np.array([height1, width1, fps1])
    np.save(data_dir / "ballpath1", ball_pos1)
    np.save(data_dir / "param1", param1)
    print(f"Video 1 Parameters: {param1}")
    print(f"Processing time: {t2 - t1:.2f}s")

    # 處理第二個影片
    print("\nProcessing video 2...")
    t1 = time.time()
    height2, width2, fps2, ball_pos2 = detect_ballpath(vid2_path, model, show_video=False)  # 關閉影片顯示以提高處理速度
    t2 = time.time()
    param2 = np.array([height2, width2, fps2])
    np.save(data_dir / "ballpath2", ball_pos2)
    np.save(data_dir / "param2", param2)
    print(f"Video 2 Parameters: {param2}")
    print(f"Processing time: {t2 - t1:.2f}s")
    print(f"\nAll results saved to: {data_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_name",
        type=str,
        required=True,
        help="Name used when saving the cropped video.",
    )
    parser.add_argument(
        "--flipped_1",
        type=bool,
        required=False,
        default=False,
        help="set to true if first video is flipped",
    )
    parser.add_argument(
        "--flipped_2",
        type=bool,
        required=False,
        default=False,
        help="set to true if second video is flipped",
    )

    args = parser.parse_args()
    main(args) 