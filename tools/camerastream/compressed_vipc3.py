import subprocess
import time
import numpy as np
from PIL import Image
import cereal.messaging as messaging
from msgq.visionipc import VisionIpcClient, VisionStreamType
import torch
from ultralytics import YOLO

def yuv_to_rgb(y, u, v):
  ul = np.repeat(np.repeat(u, 2).reshape(u.shape[0], y.shape[1]), 2, axis=0).reshape(y.shape)
  vl = np.repeat(np.repeat(v, 2).reshape(v.shape[0], y.shape[1]), 2, axis=0).reshape(y.shape)

  yuv = np.dstack((y, ul, vl)).astype(np.int16)
  yuv[:, :, 1:] -= 128

  m = np.array([
    [1.00000,  1.00000, 1.00000],
    [0.00000, -0.39465, 2.03211],
    [1.13983, -0.58060, 0.00000],
  ])
  rgb = np.dot(yuv, m).clip(0, 255)
  return rgb.astype(np.uint8)

def extract_image(buf):
  y = np.array(buf.data[:buf.uv_offset], dtype=np.uint8).reshape((-1, buf.stride))[:buf.height, :buf.width]
  u = np.array(buf.data[buf.uv_offset::2], dtype=np.uint8).reshape((-1, buf.stride//2))[:buf.height//2, :buf.width//2]
  v = np.array(buf.data[buf.uv_offset+1::2], dtype=np.uint8).reshape((-1, buf.stride//2))[:buf.height//2, :buf.width//2]

  return yuv_to_rgb(y, u, v)

def get_snapshot():
  sm = messaging.SubMaster(['roadCameraState'])
  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)

  # wait 4 sec from camerad startup for focus and exposure
  while sm['roadCameraState'].frameId < int(4. / 0.05):  # assuming DT_MDL = 0.05
    sm.update()

  vipc_client.connect(True)
  c = vipc_client
  image = extract_image(c.recv())
  return image

def run_yolov8(image):
  model = YOLO('yolov8n.pt')  # Load pre-trained YOLOv8 model
  results = model(image)
  return results

if __name__ == "__main__":
  try:
    subprocess.check_call(["pgrep", "camerad"])
    print("Camerad already running")
  except subprocess.CalledProcessError:
    print("Starting camerad...")
    subprocess.Popen(["camerad"])

  time.sleep(2.0)  # Give camerad time to start

  image = get_snapshot()
  if image is not None:
    results = run_yolov8(image)
    results.save('/tmp/')  # Save detection results
    print("YOLOv8 detection completed")
  else:
    print("Error taking snapshot")
