import cv2
import numpy as np
import argparse
from pathlib import Path


def select_points_from_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    _, frame = cap.read()
    h, _, _ = frame.shape

    class CoordinateStore:
        def __init__(self):
            self.points = []

        def select_point(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                self.points.append((x, y))
                cv2.imshow("Select Points", frame)

    cs = CoordinateStore()
    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", cs.select_point)

    while True:
        cv2.imshow("Select Points", frame)
        if cv2.waitKey(20) & 0xFF == 27:  # ESC to finish
            break

    cv2.destroyAllWindows()
    cap.release()

    coords = [[p[0], h - p[1], 1] for p in cs.points]  # Flip y
    return np.array(coords)


def main(args):
    video_name = args.video_name
    data_dir = Path("data") / video_name
    data_dir.mkdir(parents=True, exist_ok=True)

    vid1_path = Path("videos") / f"{video_name}_1.mp4"
    vid2_path = Path("videos") / f"{video_name}_2.mp4"

    print("Select 6 points for Camera 1...")
    c1 = select_points_from_video(vid1_path)
    np.save(data_dir / "c1.npy", c1)

    print("Select 6 points for Camera 2...")
    c2 = select_points_from_video(vid2_path)
    np.save(data_dir / "c2.npy", c2)

    print("Corner points saved to:", data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", type=str, required=True,
                        help="Name used for saving c1 and c2 (e.g. videos_2)")
    args = parser.parse_args()
    main(args)