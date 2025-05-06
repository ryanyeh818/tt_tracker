import cv2
import numpy as np
import pygame
from pathlib import Path
import time

def main():
    VIDEO_NAME = "videos_2"
    video1_path = Path("videos") / f"{VIDEO_NAME}_1.mp4"
    video2_path = Path("videos") / f"{VIDEO_NAME}_2.mp4"
    ballpath1 = np.load(f"data/{VIDEO_NAME}/ballpath1.npy")
    ballpath2 = np.load(f"data/{VIDEO_NAME}/ballpath2.npy")

    # 初始化 Pygame
    pygame.init()
    
    # 開啟影片
    cap1 = cv2.VideoCapture(str(video1_path))
    cap2 = cv2.VideoCapture(str(video2_path))
    
    # 獲取影片資訊
    fps = cap1.get(cv2.CAP_PROP_FPS)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 設定縮放比例
    scale = 0.5  # 縮小到原來的一半
    scaled_width = int(width * scale)
    scaled_height = int(height * scale)
    
    # 設定視窗大小
    window_width = scaled_width
    window_height = scaled_height * 2
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Dual Camera Trace Annotated")
    
    # 設定時鐘
    clock = pygame.time.Clock()
    frame_time = 1.0 / fps
    
    # 初始化字型
    font = pygame.font.Font(None, 36)
    
    traj_color = (255, 255, 0)  # 黃色
    text_color = (255, 255, 255)
    radius = 5
    trajectory1, trajectory2 = [], []
    
    frame_idx = 0
    running = True
    start_time = time.time()
    frame_count = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # 讀取幀
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2 or frame_idx >= len(ballpath1):
            break
        
        # 縮放影片幀
        frame1 = cv2.resize(frame1, (scaled_width, scaled_height))
        frame2 = cv2.resize(frame2, (scaled_width, scaled_height))
        
        # 轉換 OpenCV BGR 到 RGB
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        # 轉換為 Pygame surface
        surf1 = pygame.surfarray.make_surface(frame1.swapaxes(0, 1))
        surf2 = pygame.surfarray.make_surface(frame2.swapaxes(0, 1))
        
        # 調整軌跡座標
        x1, y1 = ballpath1[frame_idx][:2]
        x2, y2 = ballpath2[frame_idx][:2]
        
        if x1 > 0 and y1 > 0:
            pt1 = (int(x1 * scale), int(y1 * scale))
            trajectory1.append(pt1)
            pygame.draw.circle(surf1, traj_color, pt1, radius)
            if len(trajectory1) > 1:
                pygame.draw.lines(surf1, traj_color, False, trajectory1, 2)
        
        if x2 > 0 and y2 > 0:
            pt2 = (int(x2 * scale), int(y2 * scale))
            trajectory2.append(pt2)
            pygame.draw.circle(surf2, traj_color, pt2, radius)
            if len(trajectory2) > 1:
                pygame.draw.lines(surf2, traj_color, False, trajectory2, 2)
        
        # 計算並顯示 FPS
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - start_time
        actual_fps = frame_count / elapsed if elapsed > 0 else 0
        
        # 繪製文字
        fps_text = font.render(f"FPS: {actual_fps:.1f}", True, text_color)
        frame1_text = font.render(f"Frame: {frame_idx}", True, text_color)
        
        # 清除畫面
        screen.fill((0, 0, 0))
        
        # 繪製文字到各自的 surface
        surf1.blit(fps_text, (30, 30))
        surf1.blit(frame1_text, (30, 70))
        surf2.blit(fps_text, (30, 30))
        surf2.blit(frame1_text, (30, 70))
        
        # 組合兩個畫面，確保正確的位置
        screen.blit(surf1, (0, 0))
        screen.blit(surf2, (0, scaled_height))
        
        # 更新顯示
        pygame.display.flip()
        
        # 控制幀率
        clock.tick(fps)
        
        frame_idx += 1
    
    # 清理資源
    cap1.release()
    cap2.release()
    pygame.quit()

if __name__ == "__main__":
    main() 