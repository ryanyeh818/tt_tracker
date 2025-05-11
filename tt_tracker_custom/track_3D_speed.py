import numpy as np
from analysis_functions import analyzer, is_zero, inside_range
from pathlib import Path
import argparse

def calculate_speed(points, dt, max_speed=100):  # 增加速度閾值到 100 m/s
    """Calculate speed from a sequence of points"""
    speeds = np.zeros(len(points))
    valid_speeds = []
    
    for i in range(1, len(points)):
        if not is_zero(points[i]) and not is_zero(points[i - 1]):
            dist = np.linalg.norm(points[i] - points[i - 1])
            v = dist / dt
            if v < max_speed:  # 過濾掉不合理的速度
                speeds[i] = v
                valid_speeds.append(v)
    
    return speeds, valid_speeds

def main(args):
    # 讀入參數與球路徑資料
    base_path = Path("data") / args.video_name
    c1 = np.load(base_path / "c1.npy")
    c2 = np.load(base_path / "c2.npy")
    ballpath1 = np.load(base_path / "ballpath1.npy").T
    ballpath2 = np.load(base_path / "ballpath2.npy").T
    param1 = np.load(base_path / "param1.npy")
    param2 = np.load(base_path / "param2.npy")

    # 提取影像資訊
    height1, width1, fps1 = param1
    height2, width2, fps2 = param2

    print(f"Processing video: {args.video_name}")
    print(f"FPS: {fps1}")
    print(f"Total frames: {len(ballpath1[0])}")

    # 顯示校正點資訊
    print("\nCalibration Points:")
    print(f"Camera 1 points shape: {c1.shape}")
    print(f"Camera 2 points shape: {c2.shape}")
    print("\nCamera 1 points:")
    print(c1)
    print("\nCamera 2 points:")
    print(c2)

    # 檢查球路徑數據
    non_zero1 = np.sum([not is_zero(p) for p in ballpath1])
    non_zero2 = np.sum([not is_zero(p) for p in ballpath2])
    print(f"\nBall Path Statistics:")
    print(f"Camera 1 non-zero points: {non_zero1}")
    print(f"Camera 2 non-zero points: {non_zero2}")

    # 構建 analyzer 並執行 3D triangulation
    an = analyzer(int(height1), int(width1), int(height2), int(width2), c1, c2, ballpath1, ballpath2, int(fps1))
    p3d = an.p3d

    # 檢查 3D 點的有效性
    valid_points = np.sum([not is_zero(p) and inside_range(p) for p in p3d])
    print(f"\n3D Point Statistics:")
    print(f"Total points: {len(p3d)}")
    print(f"Valid points: {valid_points}")
    print(f"Valid point percentage: {(valid_points/len(p3d))*100:.2f}%")

    # 保存 3D 位置數據
    np.save(base_path / "p3d.npy", p3d)
    print(f"3D positions saved to {base_path / 'p3d.npy'}")

    # 計算每個相機的速度
    dt = 1.0 / fps1
    speed3d = np.zeros(len(p3d))
    
    # 確保所有數組長度一致
    min_length = min(len(p3d), len(ballpath1), len(ballpath2))
    p3d = p3d[:min_length]
    ballpath1 = ballpath1[:min_length]
    ballpath2 = ballpath2[:min_length]
    
    speed1, valid_speeds1 = calculate_speed(ballpath1, dt)
    speed2, valid_speeds2 = calculate_speed(ballpath2, dt)

    print(f"\nSingle Camera Speed Statistics:")
    print(f"Camera 1 valid speeds: {len(valid_speeds1)}")
    print(f"Camera 2 valid speeds: {len(valid_speeds2)}")

    # 合併速度數據
    valid_3d_speeds = 0
    for i in range(len(p3d)):
        if not is_zero(p3d[i]) and inside_range(p3d[i]):
            # 如果兩個相機都檢測到，使用平均速度
            if speed1[i] > 0 and speed2[i] > 0:
                speed3d[i] = (speed1[i] + speed2[i]) / 2
                valid_3d_speeds += 1
            # 如果只有一個相機檢測到，使用該相機的速度
            elif speed1[i] > 0:
                speed3d[i] = speed1[i]
                valid_3d_speeds += 1
            elif speed2[i] > 0:
                speed3d[i] = speed2[i]
                valid_3d_speeds += 1

    print(f"\n3D Speed Statistics:")
    print(f"Valid 3D speeds: {valid_3d_speeds}")

    # 計算速度統計資訊
    valid_speeds = [s for s in speed3d if s > 0]
    if valid_speeds:
        valid_speeds = np.array(valid_speeds)
        print("\nFinal Speed Statistics:")
        print(f"Number of valid speed measurements: {len(valid_speeds)}")
        print(f"Average speed: {np.mean(valid_speeds):.2f} m/s")
        print(f"Max speed: {np.max(valid_speeds):.2f} m/s")
        print(f"Min speed: {np.min(valid_speeds):.2f} m/s")
        print(f"Standard deviation: {np.std(valid_speeds):.2f} m/s")
        
        # 顯示每個相機的統計資訊
        if valid_speeds1:
            valid_speeds1 = np.array(valid_speeds1)
            print("\nCamera 1 Speed Statistics:")
            print(f"Number of valid measurements: {len(valid_speeds1)}")
            print(f"Average speed: {np.mean(valid_speeds1):.2f} m/s")
        
        if valid_speeds2:
            valid_speeds2 = np.array(valid_speeds2)
            print("\nCamera 2 Speed Statistics:")
            print(f"Number of valid measurements: {len(valid_speeds2)}")
            print(f"Average speed: {np.mean(valid_speeds2):.2f} m/s")
    else:
        print("\nWarning: No valid speed measurements found!")
        print("Possible reasons:")
        print("1. No valid 3D points were detected")
        print("2. Points are outside the valid range")
        print("3. Calculated speeds are too high (filtered out)")

    # 保存速度數據
    np.save(base_path / "speed3d.npy", speed3d)
    print(f"\n3D speed tracking complete. Saved to {base_path / 'speed3d.npy'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", required=True, help="Name of the video set (e.g., videos_2, videos_3)")
    args = parser.parse_args()
    main(args)
