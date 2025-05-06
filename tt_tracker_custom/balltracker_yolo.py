import numpy as np
import cv2
import time
import os
import argparse
from pathlib import Path
import process_video_yolo as pv
import platform


def main(args):
    """Processes two synced videos using YOLO to detect the position of a table
    tennis ball in each frame.

    Args:
        args: Command line arguments containing video_name and flip settings
    """
    vidname = args.video_name
    flipped1 = args.flipped_1
    flipped2 = args.flipped_2

    # 確保輸出目錄存在
    os.makedirs("data/" + vidname, exist_ok=True)
    
    # 設定影片路徑
    vid1_path = Path.joinpath(
        Path(__file__).parents[0].resolve(), "videos/" + vidname + "_1.mp4"
    )
    vid2_path = Path.joinpath(
        Path(__file__).parents[0].resolve(), "videos/" + vidname + "_2.mp4"
    )

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

    # Process video 1
    print("Processing video 1...")
    t1 = time.time()
    height1, width1, fps1, ball_pos1 = pv.detect_ballpath(vid1_path, net)
    t2 = time.time()
    param1 = np.array([height1, width1, fps1])
    np.save(
        Path.joinpath(
            Path(__file__).parents[0].resolve(), "data/" + vidname + "/ballpath1"
        ),
        ball_pos1,
    )
    np.save(
        Path.joinpath(
            Path(__file__).parents[0].resolve(), "data/" + vidname + "/param1"
        ),
        param1,
    )
    print("Video 1 Parameters:", param1)
    print("Processing time: {:.2f}s".format(t2 - t1))

    # Process video 2
    print("\nProcessing video 2...")
    t1 = time.time()
    height2, width2, fps2, ball_pos2 = pv.detect_ballpath(vid2_path, net)
    t2 = time.time()
    param2 = np.array([height2, width2, fps2])
    np.save(
        Path.joinpath(
            Path(__file__).parents[0].resolve(), "data/" + vidname + "/ballpath2"
        ),
        ball_pos2,
    )
    np.save(
        Path.joinpath(
            Path(__file__).parents[0].resolve(), "data/" + vidname + "/param2"
        ),
        param2,
    )
    print("Video 2 Parameters:", param2)
    print("Processing time: {:.2f}s".format(t2 - t1))
    print("\nAll results saved to:", Path("data") / vidname)


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