#!/usr/bin/env python3
import os
import numpy as np
import time
import cv2

def generate_random_image(W, H):
    """Generates a random BGR image."""
    return np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

def display_random_images(W, H, interval=0.1):
    """Displays random images at the specified interval."""
    while True:
        random_image = generate_random_image(W, H)
        cv2.imshow("Random Image", random_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(interval)

if __name__ == "__main__":
    W, H = 1280, 720  # You can set the desired resolution here
    display_random_images(W, H)
    cv2.destroyAllWindows()
