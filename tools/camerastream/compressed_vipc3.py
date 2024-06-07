import cv2
import numpy as np

# 빈 이미지 생성 (512x512 크기, 3채널 RGB, 배경은 흰색)
image = np.ones((512, 512, 3), np.uint8) * 255

# 직선 그리기
start_point = (50, 50)
end_point = (400, 50)
color = (0, 0, 255)  # 빨간색
thickness = 5
cv2.line(image, start_point, end_point, color, thickness)

# 직사각형 그리기
top_left = (50, 100)
bottom_right = (400, 200)
color = (0, 255, 0)  # 초록색
thickness = 3
cv2.rectangle(image, top_left, bottom_right, color, thickness)

# 원 그리기
center = (256, 300)
radius = 50
color = (255, 0, 0)  # 파란색
thickness = -1  # 채우기
cv2.circle(image, center, radius, color, thickness)

# 타원 그리기
center = (256, 400)
axes = (100, 50)
angle = 0
startAngle = 0
endAngle = 360
color = (0, 255, 255)  # 노란색
thickness = 2
cv2.ellipse(image, center, axes, angle, startAngle, endAngle, color, thickness)

# 다각형 그리기
points = np.array([[100, 300], [200, 250], [300, 300], [200, 350]], np.int32)
points = points.reshape((-1, 1, 2))
color = (255, 0, 255)  # 자주색
thickness = 2
cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)

# 텍스트 그리기
text = "Hello OpenCV"
org = (50, 450)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (0, 0, 0)  # 검은색
thickness = 2
cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

# 이미지 창 열기
cv2.imshow('Drawn Shapes', image)
cv2.waitKey(0)  # 키 입력을 기다림
cv2.destroyAllWindows()  # 모든 창 닫기
