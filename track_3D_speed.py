import numpy as np
from analysis_functions import analyzer, is_zero
from pathlib import Path

VIDEO_NAME = "videos_2"

# 讀入參數與球路徑資料
base_path = Path("data") / VIDEO_NAME
c1 = np.load(base_path / "c1.npy")
c2 = np.load(base_path / "c2.npy")
ballpath1 = np.load(base_path / "ballpath1.npy").T
ballpath2 = np.load(base_path / "ballpath2.npy").T
param1 = np.load(base_path / "param1.npy")
param2 = np.load(base_path / "param2.npy")

# 提取影像資訊
height1, width1, fps1 = param1
height2, width2, fps2 = param2

# 構建 analyzer 並執行 3D triangulation
an = analyzer(int(height1), int(width1), int(height2), int(width2), c1, c2, ballpath1, ballpath2, int(fps1))
p3d = an.p3d

# 計算 3D 球速（每幀）
speed3d = []
dt = 1.0 / fps1
for i in range(1, len(p3d)):
    if not is_zero(p3d[i]) and not is_zero(p3d[i - 1]):
        dist = np.linalg.norm(p3d[i] - p3d[i - 1])
        v = dist / dt
        speed3d.append(v)
    else:
        speed3d.append(0)

speed3d = np.array(speed3d)
np.save(base_path / "speed3d.npy", speed3d)

print(f"3D speed tracking complete. Saved to {base_path / 'speed3d.npy'}")
