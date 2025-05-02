# 🏓 桌球追蹤系統工作流程（YOLO + 3D 球速分析）

---

## 📦 總流程總覽

```
🎥 錄影 → 🔁 裁切對齊 → 🎯 點選角點 → 🤖 YOLO 偵測球 → 🧠 3D 重建 → 🏓 球速分析 → 👀 可視化展示
```

---

## ✅ 各檔案功能說明

| 步驟 | 檔案名                          | 目的                                        | 輸入檔案                                       | 輸出檔案                                                  |
|------|---------------------------------|---------------------------------------------|------------------------------------------------|-------------------------------------------------------------|
| ①    | `crop_video.py`                | 🔁 手動同步左右兩段影片                    | `videos_original/*.mp4`                        | `videos/videos_2_1.mp4`, `videos_2_2.mp4`                    |
| ②    | `select_corners_only.py`       | 🎯 點選桌角與球網點，產生校正參考點         | `videos/videos_2_*.mp4`                        | `data/videos_2/c1.npy`, `c2.npy`                            |
| ③    | `process_video_yolo.py`        | 🤖 使用 YOLO 偵測球位置，產出球軌跡         | `videos/videos_2_*.mp4`, `weights.onnx`        | `ballpath1.npy`, `ballpath2.npy`, `param1.npy`, `param2.npy` |
| ④    | `track_3d_speed.py`            | 🧠 將雙視角軌跡轉換為 3D 並計算 3D 球速     | `c1/2.npy`, `ballpath1/2.npy`, `param1/2.npy`  | `speed3d.npy`                                               |
| ⑤    | `visualize_3d_speed_overlay.py`| 👀 同步顯示兩段影片 + 追蹤點 + 3D 球速      | `videos/videos_2_*.mp4`, `ballpath*.npy`, `speed3d.npy` | 即時可視化畫面（不輸出檔案）                   |

---

## 🔁 補充工具（可選）

| 工具名                   | 用途                                      | 輸出                                   |
|--------------------------|-------------------------------------------|----------------------------------------|
| `track_ball_speed.py`    | 用單視角偵測球 + 即時計算 2D 球速         | `ballpath1.npy`, `speed1.npy`          |
| `dual_camera_visualize.py` | 上下畫面播放，確認軌跡是否同步          | 影像視窗                               |
| `demo.py`（可跳過）      | 原始 tt_tracker 的分析與可視化工具        | 可用但非必要                           |

---

## 📁 輸出資料結構（以 `data/videos_2/` 為例）

| 檔案名稱         | 說明                            |
|------------------|---------------------------------|
| `c1.npy`, `c2.npy` | 桌角與球網點座標（每支影片各 6 點） |
| `ballpath1.npy`    | Camera 1 偵測的每幀球中心位置     |
| `ballpath2.npy`    | Camera 2 偵測的每幀球中心位置     |
| `param1.npy`       | Camera 1 的 [高, 寬, fps]        |
| `param2.npy`       | Camera 2 的 [高, 寬, fps]        |
| `speed3d.npy`      | 每一幀的三維球速（units/s）       |

---

## 🧠 建議使用順序

1. `crop_video.py` → 同步左右影片
2. `select_corners_only.py` → 點選校正角點
3. `process_video_yolo.py` → 用 YOLO 偵測球
4. `track_3d_speed.py` → 三維重建 + 球速計算
5. `visualize_3d_speed_overlay.py` → 可視化追蹤 + 3D 球速顯示

---

## ✅ 後續可擴充功能建議

- 匯出 `speed3d.npy` 成 CSV / 可圖表化分析
- 標記彈跳點、加速度變化、擊球區間
- 接入判斷出界 / 未過網 / 多球追蹤
