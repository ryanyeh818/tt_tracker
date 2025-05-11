# 🏓 桌球追蹤系統 (Table Tennis Tracker)

# Custom TT_Tracker 

使用兩個攝影機追蹤桌球在 3D 空間中的運動，並分析結果。使用 YOLO 和電腦視覺技術來識別球的位置、速度等。

這個專案主要展示如何使用 YOLO 和電腦視覺技術來分析桌球比賽。目前已在 1080p、120 fps 的雙攝影機影片上測試過，其他配置可能需要調整。

## Overview
### 主要工具
* **analysis_functions.py** - 包含 `Analyzer` 類別，用於計算 3D 軌跡、速度等，以及視覺化功能。同時包含相機校正、三角測量、插值、視覺化等功能。
  - 輸出：無直接輸出檔案，提供其他模組使用的功能
  - 被引用於：`track_3D_speed.py`

* **balltracker_yolo.py** - 使用 YOLO 模型在兩個影片的每一幀中尋找球的位置，並將這些位置儲存在 data 目錄中。
  - 輸入：`videos/videos_2_*.mp4`, `weights.onnx`
  - 輸出：`data/videos_2/ballpath1.npy`, `ballpath2.npy`, `param1.npy`, `param2.npy`
  - 依賴：`process_video_yolo.py`

* **crop_video.py** - 用於裁切和同步兩個影片，確保它們長度相同，且每一幀都大致對應到相同的時間點。
  - 輸入：`videos_original/*.mp4`
  - 輸出：`videos/videos_2_1.mp4`, `videos_2_2.mp4`

* **extended_yolo_v3.py** - 包含使用 YOLO 模型偵測桌球的功能。
  - 輸出：無直接輸出檔案，提供 YOLO 模型相關功能
  - 被引用於：`process_video_yolo.py`

* **process_video_yolo.py** - 使用 YOLO 模型處理影片並偵測球的位置。
  - 輸入：`videos/videos_2_*.mp4`, `weights.onnx`
  - 輸出：無直接輸出檔案，提供球偵測功能
  - 依賴：`extended_yolo_v3.py`
  - 被引用於：`balltracker_yolo.py`

* **track_3d_speed.py** - 將雙視角軌跡轉換為 3D 並計算球速。
  - 輸入：`data/videos_2/c1.npy`, `c2.npy`, `ballpath1.npy`, `ballpath2.npy`, `param1.npy`, `param2.npy`
  - 輸出：`data/videos_2/speed3d.npy`
  - 依賴：`analysis_functions.py`

* **visualize_3d_speed_overlay.py** - 同步顯示兩段影片，並疊加追蹤點和 3D 球速資訊。
  - 輸入：`videos/videos_2_*.mp4`, `data/videos_2/ballpath*.npy`, `speed3d.npy`
  - 輸出：即時視覺化畫面（不輸出檔案）

### 補充工具

* **tools/track_ball_speed.py** - 使用單一視角偵測球並即時計算 2D 球速。
  - 輸入：`videos/videos_2_*.mp4`, `weights.onnx`
  - 輸出：`data/videos_2/ballpath1.npy`, `speed1.npy`
  - 依賴：`extended_yolo_v3.py`

* **tools/visualize_trace_on_video.py** - 在影片上顯示球的軌跡。
  - 輸入：`videos/videos_2_*.mp4`, `data/videos_2/ballpath*.npy`
  - 輸出：即時視覺化畫面（不輸出檔案）

* **tools/analyze_pts_diff.py** - 分析兩個影片的球軌跡差異。
  - 輸入：`data/videos_2/ballpath1.npy`, `ballpath2.npy`
  - 輸出：分析結果（不輸出檔案）

### 使用方式

#### 步驟 1: 準備影片

使用兩個攝影機同時拍攝桌球比賽。在專案目錄下建立 "videos_original" 資料夾，並將兩個影片存放在這裡。

#### 步驟 2: 裁切同步 (crop_video.py)

在專案目錄下建立 "videos" 資料夾。在 crop_video.py 中設定 video_name 變數，然後執行程式。

程式會開啟視窗顯示兩個攝影機的當前幀。按 1 或 2 可以讓對應的攝影機前進。當找到同步點時，按 ESC 鍵，程式會將剩餘的影片儲存到 "videos" 資料夾中。

#### 步驟 3: 選擇角點 (select_corners_only.py)

在 select_corners_only.py 中設定相同的 video_name，然後執行程式。程式會要求你在兩個影片中標記桌角和球網位置。在視窗中，按照以下順序雙擊標記點：
1. 左上角
2. 右上角
3. 右下角
4. 左下角
5. 球網頂部（從下到上）

#### 步驟 4: 追蹤球 (balltracker_yolo.py)

使用 YOLO 模型在每一幀中偵測球的位置。結果會儲存在 "data" 資料夾中，包含球的位置、角點位置、fps 等資訊。

#### 步驟 5: 計算 3D 速度 (track_3d_speed.py)

將雙視角的球軌跡轉換為 3D 座標，並計算每一幀的 3D 球速。

#### 步驟 6: 視覺化結果 (visualize_3d_speed_overlay.py)

同步顯示兩個視角的影片，並疊加球的軌跡和 3D 球速資訊。

### 輸出資料結構（以 `data/videos_2/` 為例）

| 檔案名稱         | 說明                            |
|------------------|---------------------------------|
| `c1.npy`, `c2.npy` | 桌角與球網點座標（每支影片各 6 點） |
| `ballpath1.npy`    | Camera 1 偵測的每幀球中心位置     |
| `ballpath2.npy`    | Camera 2 偵測的每幀球中心位置     |
| `param1.npy`       | Camera 1 的 [高, 寬, fps]        |
| `param2.npy`       | Camera 2 的 [高, 寬, fps]        |
| `speed3d.npy`      | 每一幀的三維球速（units/s）       |

### 後續改進方向

- 自動偵測桌角和球網位置
- 標記彈跳點、加速度變化、擊球區間
- 接入判斷出界 / 未過網 / 多球追蹤
- 匯出 `speed3d.npy` 成 CSV / 可圖表化分析

## 環境設置

### 系統要求
- Python：3.9.22（建議使用 3.9 或更高版本）
- OpenCV：4.11.0
- Pygame：2.5.0
- NumPy：2.0.2
- PyTorch：1.13.1
- torchvision：0.14.1
- matplotlib：3.9.2
- scipy：1.13.1

### 安裝步驟

1. 確保安裝了正確版本的 Python：
```bash
python --version  # 應顯示 3.9.22
```

2. 克隆倉庫：
```bash
git clone https://github.com/ryanyeh818/tt_tracker.git
cd tt_tracker
```

3. 創建並激活虛擬環境：
```bash
conda create -n tt python=3.9
conda activate tt
```

4. 安裝依賴：
```bash
pip install -r requirements.txt
```

### 注意事項
- 本專案在 Python 3.9.22 環境下開發和測試
- 如果使用其他 Python 版本，建議使用 3.9 版本以確保最佳兼容性
- 如果遇到 OpenCV 相關問題，可能需要安裝額外的系統依賴
- 建議使用 Conda 環境來管理 Python 版本和套件依賴
---

# Original TT_Tracker

Tracking a table tennis ball in 3D using two cameras, and analyzing the result. Uses OpenCV and computer vision techniques to identify points, strokes, bounces, etc.

This project is not optimized for extended use, but rather showcases an idea for how OpenCV and computer vision can in a simple way be applied to analyze table tennis games. It has only been tested on videos captured at 720p, 30 fps on two iphones, and might need changes to work well for other configurations.

### Overview

* **analysis_functions.py** - Contains the `Analyzer` class, which computes 3D tracks, strokes, points etc. and visualization functions.
Also contains functions for camera calibration, trilateration, interpolation, visualization, etc.

* **balltracker.py** - Script that uses process_video.py to find ball positions in each frame of two videos, and storing these positions in the data-directory.

* **crop_video.py** - Script for cropping and syncing two videos so that they are of the same length, and each two frames are approximately from the same point in time.

* **demo.py** - Script demonstrating usage of Analyzer-class, and doing visualization.

* **process_video.py** - Contains functions for detecting a table tennis ball in each frame of a video.

### Demo

The data visualized below is the output from running the algorithm on two videos that are not uploaded here.

Uncomment a desired function furthest down in demo.py and run the script. This will create an "analyzer"-object. The following functions applied to this object are currently supported:

#### visualize_3d_strokes()

Plots the path of one detected stroke at a time in 3d, detected bounce positions, and the camera positions.

#### visualize_2d_strokes()

Plots the path of one detected stroke at a time in 2d and detected bounce positions.

#### animate_3d_path()

Creates an animation of the estimated 3d path taken in the whole video.

#### bounce_heatmap()

Plots a heatmap of the detected bounces on the table.

#### plot_3d_point()

Plots the estimated path of a whole point in 3d together with camera positions.

#### plot_trajectory()

Plots the estimated trajectory viewed from the perspective of one of the cameras.

### Usage

#### Step 1: Clone repository

Clone repository to desired location.

#### Step 2: Capture and save

Capture two videos simultaneously using two cameras from either side of the table.

Create a new folder called "videos_original" in the working directory and store the two videos here.

#### Step 3: Crop (crop_video.py)

Create a new folder called "videos" in the working directory. Assign the variable video_name in crop_video.py a name of the instance that will be used to store the data later. Then run the script.

The function will open a window which shows the current frame from each camera. Press 1 or 2 to run a camera forward. When a synced position in the video has been reached, press esc. and the script will save the ramainder of the videos until one has no more frames into the "videos"-folder. This ensures equal length of the two videos.

#### Step 4: Process videos (balltracker.py)

Set video_name to the chosen instance name in balltracker.py and then run the script. This will prompt you to locate corners and net positions in both videos. In the window that appears, double-click on the corner points and net positions in the order they appear here: (i.e. clock-wise from the top left to the bottom left, and then the top of the net, from bottom to top)

After doing this the videos will be processed, and the detected position in each frame will be saved in the "data"-folder under the name of the video, together with the corner positions, fps, etc. This will take some time, depending on the length, resolution, etc. of the videos.

#### Step 5: Analyze result (demo.py)

In demo.py set video_name to the name of your instance, and then choose to run the desired function.

### Improvements

Implement automatic detection of table corners and net. (Problem if background contains white lines)

Detect patterns, points of contact, etc.

# TT Tracker

桌球追蹤系統，支援單/雙鏡頭追蹤，可計算球速、軌跡等數據。

## 功能特點

- 支援單/雙鏡頭追蹤
- 自動計算球速和軌跡
- 支援 CUDA 加速（自動降級到 CPU）
- 支援影片裁剪和處理
- 支援 3D 軌跡可視化
- 支援即時追蹤和影片回放

## 環境需求

- Python 3.8+
- OpenCV
- PyTorch
- CUDA (可選，用於 GPU 加速)
- 其他依賴套件見 `requirements.txt`

## 安裝

1. 克隆專案：
```bash
git clone https://github.com/ryanyeh818/tt_tracker.git
cd tt_tracker
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

## 使用說明

### 基本追蹤
```bash
python tt_tracker_custom/balltracker_yolo.py --video path/to/video.mp4
```

### 雙鏡頭追蹤
```bash
python tt_tracker_custom/dual_camera_visualize_annotated.py --video1 path/to/video1.mp4 --video2 path/to/video2.mp4
```

### 影片裁剪
```bash
python tt_tracker_custom/crop_video.py --video path/to/video.mp4 --output path/to/output.mp4
```

### 3D 軌跡追蹤
```bash
python tt_tracker_custom/track_3D_speed.py --video1 path/to/video1.mp4 --video2 path/to/video2.mp4
```

## CUDA 支援

系統會自動檢測是否有 CUDA 可用：
- 如果有 CUDA 且 GPU 可用，將自動使用 GPU 加速
- 如果沒有 CUDA 或 GPU 不可用，將自動降級到 CPU 模式

您可以使用 `test_cuda.py` 來測試 CUDA 是否可用：
```bash
python tt_tracker_custom/test_cuda.py
```

## 注意事項

1. 確保相機已正確校準
2. 建議使用高幀率相機以獲得更好的追蹤效果
3. 如果使用 GPU 加速，請確保已安裝正確版本的 CUDA 和 cuDNN

## 授權

MIT License