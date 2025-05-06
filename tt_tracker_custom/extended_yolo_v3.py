import cv2
import numpy as np
import time
from collections import deque
import math

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5        # 分數門檻 (依需求微調)
NMS_THRESHOLD = 0.3          # 非極大值抑制門檻
CONFIDENCE_THRESHOLD = 0.3   # 信心門檻

# 軌跡相關參數
MAX_TRAJECTORY_POINTS = 30   # 保留軌跡點的最大數量
TRAJECTORY_THICKNESS = 2     # 軌跡線條粗細
TRAJECTORY_COLOR = (0, 255, 255)  # 軌跡顏色 (黃色)

# 速度計算相關參數
PIXELS_PER_CM = 5.0          # 像素到實際距離的換算比例 (需根據實際情況校準)
TARGET_FPS = 120.0           # 目標幀率 (用於速度計算)

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 1

# Colors.
BLACK  = (0, 0, 0)
BLUE   = (255, 178, 50)
YELLOW = (0, 255, 255)
RED    = (0, 0, 255)
GREEN  = (0, 255, 0)
ORANGE = (0, 165, 255)
WHITE  = (255, 255, 255)

VIDEO_FPS = 25
FRAME_SIZE = (1920, 1080)

# 用於存儲球的軌跡
trajectory_points = deque(maxlen=MAX_TRAJECTORY_POINTS)
# 存儲球體的速度數據
speed_data = deque(maxlen=10)
# 上一幀的時間戳
prev_timestamp = 0
# 速度平滑過濾器
avg_speed = 0

def draw_label(input_image, label, left, top):
    """在圖像上標記文字"""
    text_size, baseline = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    cv2.rectangle(input_image, (left, top), (left + text_size[0], top + text_size[1] + baseline), BLACK, cv2.FILLED)
    cv2.putText(input_image, label, (left, top + text_size[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

def pre_process(input_image, net):
    """前處理：生成 blob 並運行前向傳播"""
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)
    return outputs

def post_process(input_image, outputs):
    """後處理：解析網路輸出"""
    class_ids = []
    confidences = []
    boxes = []
    
    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)
            if classes_scores[class_id] > SCORE_THRESHOLD:
                confidences.append(confidence)
                class_ids.append(class_id)
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)
    return boxes, confidences, class_ids

def draw_boxes(input_image, boxes, confidences, class_ids, color):
    """
    根據檢測結果畫框，
    並僅保留信心最高的第一個框作為有效結果。
    """
    selected_box = []
    ball_center = None
    
    if len(confidences) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    else:
        indices = []
        
    for i in indices:
        box = boxes[i]
        left, top, width, height = box
        cv2.rectangle(input_image, (left, top), (left + width, top + height), color, 3 * THICKNESS)
        if confidences:
            label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
            draw_label(input_image, label, left + width, top + height)
        selected_box = box
        # 計算球體中心點
        ball_center = (left + width // 2, top + height // 2)
        break  # 僅處理第一個有效框
        
    return input_image, selected_box, ball_center

def background_subtraction_method(old_box, img, fgMask):
    """
    當偵測或追蹤失敗時，利用背景減除方法進行補救，
    裁切可能出現球體的區域後再進行一次偵測。
    """
    ball_center = None
    
    # 改進：增加形態學操作以減少噪聲
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
    
    if len(old_box) > 0:
        x1, y1, w, h = old_box
        x1 = max(x1 - 250, 0)
        y1 = max(y1 - 250, 0)
        x2 = min(x1 + w + 250, fgMask.shape[1])
        y2 = min(y1 + h + 250, fgMask.shape[0])
        
        mask = np.zeros(fgMask.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        fgMask = cv2.bitwise_and(fgMask, fgMask, mask=mask)
    
    contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boxes2 = []
    
    # 改進：增加彈性的輪廓篩選條件
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 更寬鬆的面積限制，適應不同距離的球體
        if 100 < area < 500:
            # 計算輪廓的圓形度
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                # 圓形的輪廓圓形度接近1
                if circularity > 0.5:  # 允許部分變形
                    box2 = cv2.boundingRect(cnt)
                    boxes2.append(np.array(box2))
    
    img, box, ball_center = draw_boxes(img, boxes2, [1.0] * len(boxes2), [0] * len(boxes2), ORANGE)
    return img, box.copy() if len(box) > 0 else np.empty(0, dtype=np.int32), ball_center, fgMask

def draw_trajectory(image, points):
    """在圖像上繪製球體軌跡"""
    # 繪製軌跡線
    for i in range(1, len(points)):
        if points[i] is None or points[i-1] is None:
            continue
        # 根據軌跡點的順序變化顏色，越新的點顏色越亮
        color_factor = i / len(points)
        color = (
            int(TRAJECTORY_COLOR[0] * color_factor),
            int(TRAJECTORY_COLOR[1] * color_factor),
            int(TRAJECTORY_COLOR[2] * color_factor)
        )
        cv2.line(image, points[i-1], points[i], color, TRAJECTORY_THICKNESS)
    
    # 標記軌跡點
    for point in points:
        if point is not None:
            cv2.circle(image, point, 3, YELLOW, -1)
    
    return image

def calculate_speed(prev_point, current_point, time_elapsed, avg_speed):
    """計算球體速度 (cm/s)"""
    if prev_point is None or current_point is None or time_elapsed == 0:
        return avg_speed
    
    # 計算像素距離
    distance_pixels = np.sqrt((current_point[0] - prev_point[0])**2 + (current_point[1] - prev_point[1])**2)
    
    # 轉換為實際距離 (cm)
    distance_cm = distance_pixels / PIXELS_PER_CM
    
    # 計算速度 (cm/s)
    speed = distance_cm / time_elapsed
    
    # 排除異常值 (太高或太低的速度)
    if 50 < speed < 2000:  # 一般桌球速度範圍
        speed_data.append(speed)
        # 計算平均速度
        if len(speed_data) > 0:
            avg_speed = sum(speed_data) / len(speed_data)
    
    return avg_speed

def predict_next_position(trajectory_points, frames_to_predict=5):
    """基於現有軌跡預測未來位置"""
    if len(trajectory_points) < 3 or trajectory_points[-1] is None:
        return None
    
    # 至少需要2個點來計算方向和速度
    valid_points = [p for p in trajectory_points if p is not None]
    if len(valid_points) < 2:
        return None
    
    # 計算最近兩點的移動向量
    p1 = valid_points[-2]
    p2 = valid_points[-1]
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    # 預測下一個位置
    next_x = p2[0] + dx * frames_to_predict
    next_y = p2[1] + dy * frames_to_predict
    
    return (int(next_x), int(next_y))

if __name__ == '__main__':
    # 載入類別名稱
    classesFile = "classes.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    cap = cv2.VideoCapture("test1_4.mov", cv2.CAP_AVFOUNDATION)  # 適用 macOS
    # cap = cv2.VideoCapture("test1_3.mov")  # 適用 Windows
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        exit()
    ret, frame = cap.read()
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 原始視頻的FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    # 計算時間比例因子 (從原始FPS到目標FPS的轉換)
    time_scale_factor = original_fps / TARGET_FPS
    
    # 初始化背景減除器 (調整參數以提高性能)
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=80)
    
    # 載入模型
    modelWeights = "weights.onnx"
    net = cv2.dnn.readNet(modelWeights)
    
    # 嘗試設定 CUDA 加速；改用 DNN_TARGET_CUDA_FP16
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        print("Using CUDA with FP16 for inference.")
    except Exception as e:
        print("CUDA not available or configuration error, falling back to CPU. Error:", e)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    old_box = np.empty(0, dtype=np.int32)
    old_img = np.empty_like(frame, dtype=np.uint8)
    old_fgMask = np.empty_like(frame, dtype=np.uint8)
    old_interval_diff = 0
    prev_ball_center = None
    frame_count = 0
    prev_time = time.time()

    video_cod = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = frame.shape[:2] # 動態取得視頻的高度和寬度，不把它寫死
    writer = cv2.VideoWriter('test1_4_final_v1.mp4', video_cod, VIDEO_FPS, (width, height)) #如果是openTTdataset影片的話，writer最後一個參數直接呼叫FRAME_SIZE即可
    if not writer.isOpened():
        print("❌ VideoWriter 開啟失敗，請確認路徑或格式問題")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        current_time = time.time()
        time_elapsed = (current_time - prev_time) * time_scale_factor
        prev_time = current_time

        # 嘗試用 CUDA 進行推論；若失敗則改用 CPU 模式重試
        try:
            detections = pre_process(frame, net)
        except cv2.error as e:
            print("Error during CUDA inference, falling back to CPU. Error:", e)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            detections = pre_process(frame, net)
        
        boxes, confidences, class_ids = post_process(frame.copy(), detections)

        img = frame.copy()
        # 進行背景減除，先對圖像進行 GaussianBlur 降噪
        blurred_img = cv2.GaussianBlur(img, (15, 15), 3)
        fgMask = backSub.apply(blurred_img)
        
        new_interval_diff = np.mean(cv2.absdiff(img, old_img))
        new_interval = (new_interval_diff - old_interval_diff) > 1
        old_interval_diff = new_interval_diff

        ball_center = None
        detection_method = "None"
        
        # 優先採用檢測結果 (以綠框顯示)
        img, box, ball_center = draw_boxes(img, boxes, confidences, class_ids, GREEN)
        if len(box) != 0:
            old_box = box.copy()
            detection_method = "Detection"
        elif len(old_box) > 0:
            # 當偵測失敗時，利用追蹤器 (紅框) 嘗試追蹤
            tracker = cv2.TrackerCSRT.create()  # 重新建立追蹤器
            tracker.init(frame, tuple(old_box))
            found, box2 = tracker.update(frame)
            if found:
                img, box, ball_center = draw_boxes(img, [np.array(box2)], [0.8], [0], RED)
                old_box = box.copy()
                detection_method = "Tracking"
            else:
                # 當追蹤失敗時，使用背景減除法 (橙框)
                img, old_box, ball_center, fgMask = background_subtraction_method(old_box, img, fgMask)
                if len(old_box) > 0:
                    detection_method = "BGSub"
                # 如果背景減除也失敗，嘗試使用軌跡預測
                elif len(trajectory_points) >= 3:
                    predicted_center = predict_next_position(trajectory_points)
                    if predicted_center:
                        ball_center = predicted_center
                        # 從預測位置創建一個框
                        pred_box = np.array([
                            predicted_center[0] - 15, 
                            predicted_center[1] - 15, 
                            30, 30
                        ])
                        img, _, _ = draw_boxes(img, [pred_box], [0.6], [0], BLUE)
                        old_box = pred_box.copy()
                        detection_method = "Prediction"
        else:
            # 當無法追蹤時，使用背景減除法
            img, old_box, ball_center, fgMask = background_subtraction_method(old_box, img, fgMask)
            if len(old_box) > 0:
                detection_method = "BGSub"
                
        # 如果成功檢測到球體，添加到軌跡中
        if ball_center is not None:
            trajectory_points.append(ball_center)
            
            # 計算球體速度
            if prev_ball_center is not None and time_elapsed > 0:
                avg_speed = calculate_speed(prev_ball_center, ball_center, time_elapsed, avg_speed)
            
            prev_ball_center = ball_center
        else:
            # 如果沒有檢測到球，添加None到軌跡以保持連續性
            trajectory_points.append(None)
        
        # 繪製球體軌跡
        img = draw_trajectory(img, [p for p in trajectory_points if p is not None])
        
        # 在圖像上顯示球速
        speed_text = "Ball Speed: {:.2f} cm/s ({:.2f} km/h)".format(avg_speed, avg_speed * 0.036)
        cv2.putText(img, speed_text, (50, 50), FONT_FACE, 1, WHITE, 2, cv2.LINE_AA)
        
        # 顯示當前使用的檢測方法
        method_text = "Method: {}".format(detection_method)
        cv2.putText(img, method_text, (50, 90), FONT_FACE, 1, WHITE, 2, cv2.LINE_AA)
        
        # 處理圖像和寫入視頻
        cv2.imshow('Output', img)
        writer.write(img)
        old_img = img.copy()
        old_fgMask = fgMask.copy()

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()