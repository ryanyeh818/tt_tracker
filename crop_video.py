import cv2
import numpy as np
import argparse
from pathlib import Path

def main(args):
    path1 = Path(__file__).parent / "videos_original" / args.path_1
    path2 = Path(__file__).parent / "videos_original" / args.path_2
    vidname = args.name_out

    cap1 = cv2.VideoCapture(str(path1))
    cap2 = cv2.VideoCapture(str(path2))
    nbr1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    nbr2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    _, f1 = cap1.read()
    _, f2 = cap2.read()
    h, w = f1.shape[:2]

    # 同步畫面（按 1/2，最後按 Enter 或 Esc）
    while True:
        disp = cv2.resize(np.hstack((f1, f2)), (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Sync & Crop", disp)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("1"):
            _, f1 = cap1.read(); nbr1 -= 1
        elif k == ord("2"):
            _, f2 = cap2.read(); nbr2 -= 1
        else:
            break

    # 確保輸出資料夾
    out_dir = Path(__file__).parent.resolve() / "videos"
    out_dir.mkdir(exist_ok=True)
    out1 = out_dir / f"{vidname}_1.mp4"
    out2 = out_dir / f"{vidname}_2.mp4"

    # 建立 FFmpeg VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    fps    = cap1.get(cv2.CAP_PROP_FPS) or 30.0
    clip1  = cv2.VideoWriter(str(out1), cv2.CAP_FFMPEG, fourcc, fps, (w, h))
    clip2  = cv2.VideoWriter(str(out2), cv2.CAP_FFMPEG, fourcc, fps, (w, h))
    print("Writer1:", clip1.isOpened(), "->", out1)
    print("Writer2:", clip2.isOpened(), "->", out2)

    # 寫入影片
    for _ in range(min(nbr1, nbr2)):
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()
        clip1.write(frame1)
        clip2.write(frame2)

    clip1.release()
    clip2.release()
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_1", required=True)
    parser.add_argument("--path_2", required=True)
    parser.add_argument("--name_out", required=True)
    args = parser.parse_args()
    main(args)
