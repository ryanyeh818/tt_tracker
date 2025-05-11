import cv2
import numpy as np
import argparse
from pathlib import Path
from extended_yolo_v3 import pre_process, post_process
import platform
from tqdm import tqdm


def detect_ballpath(video_path, net, class_name="ball"):
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ballpath = np.zeros((frame_count, 3))  # x, y, 1
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

        detections = pre_process(frame, net)
        boxes, confidences, class_ids = post_process(frame, detections)

        if len(boxes) > 0:
            total_boxes += len(boxes)
            # 取第一個檢測到的球（信心最高）
            box = boxes[0]
            conf = confidences[0]
            max_conf = max(max_conf, conf)
            min_conf = min(min_conf, conf)
            
            left, top, width_box, height_box = box
            center_x = left + width_box // 2
            center_y = top + height_box // 2
            ballpath[frame_idx] = [center_x, center_y, 1]
            detection_count += 1
        else:
            ballpath[frame_idx] = [0, 0, 0]

        frame_idx += 1
        pbar.update(1)  # 更新進度條

    pbar.close()  # 關閉進度條
    cap.release()

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
    video_name = args.video_name
    video_path1 = Path("videos") / f"{video_name}_1.mp4"
    video_path2 = Path("videos") / f"{video_name}_2.mp4"
    out_dir = Path("data") / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 載入模型
    modelWeights = "weights.onnx"
    net = cv2.dnn.readNet(modelWeights)

    # 設定 CUDA 後端
    has_cuda = False
    try:
        has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except:
        print("CUDA not available on this system")

    if has_cuda:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            print("Using CUDA acceleration")
        except Exception as e:
            print(f"CUDA setup failed: {e}")
            print("Falling back to CPU")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    else:
        print("Using CPU for inference")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    print("Processing Camera 1...")
    h1, w1, fps1, ballpath1 = detect_ballpath(video_path1, net)
    np.save(out_dir / "ballpath1.npy", ballpath1)
    np.save(out_dir / "param1.npy", np.array([h1, w1, fps1]))

    print("\nProcessing Camera 2...")
    h2, w2, fps2, ballpath2 = detect_ballpath(video_path2, net)
    np.save(out_dir / "ballpath2.npy", ballpath2)
    np.save(out_dir / "param2.npy", np.array([h2, w2, fps2]))

    print("\nDone. Results saved to:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", type=str, required=True,
                        help="Name prefix of the video, e.g. 'videos_2'")
    args = parser.parse_args()
    main(args)
