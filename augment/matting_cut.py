import cv2
import numpy as np

# 读取图片
image = cv2.imread(r'D:\Re_ID\dataset\Augment\test\000835470_rgba.png', cv2.IMREAD_UNCHANGED)

# 提取透明度通道
alpha_channel = image[:, :, 3]

# 创建一个二值化的掩码，用于标识非透明像素的位置
mask = np.where(alpha_channel != 0, 255, 0).astype(np.uint8)

# 查找轮廓
contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 寻找最大连通区域
max_area = 0
max_contour = None
for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        max_contour = contour

# 创建一个全黑的空白掩码
height, width = mask.shape
cleaned_mask = np.zeros((height, width), dtype=np.uint8)

# 仅在空白掩码上绘制最大连通区域的轮廓
cv2.drawContours(cleaned_mask, [max_contour], 0, 255, -1)

# 寻找最小矩形边界
x, y, w, h = cv2.boundingRect(max_contour)

# 裁剪图片
cropped_image = image[y:y+h, x:x+w]

# 保存裁剪后的图片
cv2.imwrite('cropped_image.png', cropped_image)
