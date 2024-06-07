#!/usr/bin/env python3
import os
import sys
import numpy as np
import cv2
import multiprocessing
import time
import cereal.messaging as messaging
from cereal.visionipc import VisionStreamType, VisionIpcClient

V4L2_BUF_FLAG_KEYFRAME = 8

def video_display(vipc_addr, vision_stream_type, W, H, debug=False):
    print("Starting video display...")
    
    # Initialize the VisionIPC client
    vipc_client = VisionIpcClient("camerad", vision_stream_type)
    if not vipc_client.connect(False):
        print("Failed to connect to VisionIPC server")
        return
    
    print("Connected to VisionIPC server")
    
    while True:
        # Get the frame from the VisionIPC client
        frame = vipc_client.recv()
        if frame is None:
            if debug:
                print("No frame received")
            continue
        
        img_yuv = np.frombuffer(frame.data, dtype=np.uint8)
        y_size = W * H
        uv_size = y_size // 2
        
        # Extract Y, U, and V planes
        y = img_yuv[:y_size].reshape(H, W)
        uv = img_yuv[y_size:].reshape(2, uv_size // 2).T.reshape(H // 2, W)
        
        # Stack the U and V planes
        u = uv[:, :W // 2]
        v = uv[:, W // 2:]
        
        # Resize U and V planes to match Y plane
        u = cv2.resize(u, (W, H), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Merge Y, U, and V planes into one YUV image
        img_yuv = cv2.merge([y, u, v])
        
        # Convert YUV image to BGR
        img_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        # Display the image
        cv2.imshow("Video Stream", img_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    vipc_client.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    addr = "192.168.0.28"
    W = 1920  # Width of the video frame
    H = 1080  # Height of the video frame
    debug = False

    vision_stream_type = VisionStreamType.VISION_STREAM_ROAD

    p = multiprocessing.Process(target=video_display, args=(addr, vision_stream_type, W, H, debug))
    p.start()
    p.join()
