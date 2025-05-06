import cv2
import numpy as np
import argparse
from pathlib import Path
from extended_yolo_v3 import pre_process, post_process
import platform


def detect_ballpath(video_path, net, class_name="ball"):
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ballpath = np.zeros((frame_count, 3))  # x, y, 1

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = pre_process(frame, net)
        boxes, confidences, class_ids = post_process(frame, detections)

        if len(boxes) > 0:
            # 取第一個檢測到的球（信心最高）
            box = boxes[0]
            left, top, width_box, height_box = box
            center_x = left + width_box // 2
            center_y = top + height_box // 2
            ballpath[frame_idx] = [center_x, center_y, 1]
        else:
            ballpath[frame_idx] = [0, 0, 0]

        frame_idx += 1

    cap.release()
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

    if platform.system() == "Darwin":
        # 在 macOS 上強制使用 CPU
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    else:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        except:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    print("Processing Camera 1...")
    h1, w1, fps1, ballpath1 = detect_ballpath(video_path1, net)
    np.save(out_dir / "ballpath1.npy", ballpath1)
    np.save(out_dir / "param1.npy", np.array([h1, w1, fps1]))

    print("Processing Camera 2...")
    h2, w2, fps2, ballpath2 = detect_ballpath(video_path2, net)
    np.save(out_dir / "ballpath2.npy", ballpath2)
    np.save(out_dir / "param2.npy", np.array([h2, w2, fps2]))

    print("Done. Results saved to:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", type=str, required=True,
                        help="Name prefix of the video, e.g. 'videos_2'")
    args = parser.parse_args()
    main(args)
