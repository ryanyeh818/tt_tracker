import cv2
import numpy as np
from pathlib import Path
import time

def analyze_video(video_path, video_name):
    print(f"\nAnalyzing {video_name}...")
    cap = cv2.VideoCapture(str(video_path))
    
    # 獲取影片資訊
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video Info:")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Resolution: {width}x{height}")
    
    # 分析實際播放速度
    frame_times = []
    frame_numbers = []
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = time.time()
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        frame_times.append(current_time - start_time)
        frame_numbers.append(frame_number)
        
        # 顯示當前幀資訊
        cv2.putText(frame, f"Frame: {frame_number}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {frame_times[-1]:.2f}s", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow(f"Analyzing {video_name}", frame)
        if cv2.waitKey(1) == 27:  # ESC to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 計算實際 FPS
    if len(frame_times) > 1:
        actual_fps = len(frame_times) / frame_times[-1]
        print(f"\nActual Playback Analysis:")
        print(f"  Total Playback Time: {frame_times[-1]:.2f}s")
        print(f"  Actual FPS: {actual_fps:.2f}")
        print(f"  Expected FPS: {fps}")
        print(f"  FPS Difference: {abs(actual_fps - fps):.2f}")
        
        # 計算幀間隔時間
        frame_intervals = np.diff(frame_times)
        avg_interval = np.mean(frame_intervals)
        std_interval = np.std(frame_intervals)
        print(f"\nFrame Timing Analysis:")
        print(f"  Average Frame Interval: {avg_interval*1000:.2f}ms")
        print(f"  Standard Deviation: {std_interval*1000:.2f}ms")
        print(f"  Min Interval: {np.min(frame_intervals)*1000:.2f}ms")
        print(f"  Max Interval: {np.max(frame_intervals)*1000:.2f}ms")
        
        # 檢查是否有掉幀
        expected_frames = int(fps * frame_times[-1])
        dropped_frames = expected_frames - len(frame_times)
        print(f"\nFrame Drop Analysis:")
        print(f"  Expected Frames: {expected_frames}")
        print(f"  Actual Frames: {len(frame_times)}")
        print(f"  Dropped Frames: {dropped_frames}")
        print(f"  Drop Rate: {(dropped_frames/expected_frames)*100:.2f}%")
    else:
        print("Not enough frames to analyze")

def main():
    VIDEO_NAME = "videos_2"
    video1_path = Path("videos") / f"{VIDEO_NAME}_1.mp4"
    video2_path = Path("videos") / f"{VIDEO_NAME}_2.mp4"
    
    analyze_video(video1_path, "Camera 1")
    analyze_video(video2_path, "Camera 2")

if __name__ == "__main__":
    main() 