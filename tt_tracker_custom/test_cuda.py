import cv2
import numpy as np

def test_cuda():
    print("OpenCV version:", cv2.__version__)
    print("CUDA available:", cv2.cuda.getCudaEnabledDeviceCount() > 0)
    
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("CUDA device name:", cv2.cuda.getDevice())
        print("CUDA device properties:", cv2.cuda.getDeviceProperties(0))
    else:
        print("CUDA is not available. Please check your installation.")

if __name__ == "__main__":
    test_cuda() 