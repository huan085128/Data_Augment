import cv2
import numpy as np
import os

def automsrcr(image, scales, gain=1.0, offset=0.0, weight=None):
    if weight is None:
        weight = np.ones(len(scales))
    weight = weight / np.sum(weight)

    R = image[:, :, 2].astype(np.float32)
    G = image[:, :, 1].astype(np.float32)
    B = image[:, :, 0].astype(np.float32)

    I = (R + G + B) / 3

    output = np.zeros_like(I)

    for scale, w in zip(scales, weight):
        I_blurred = cv2.GaussianBlur(I, (0, 0), scale)
        R = (I + gain * (I - I_blurred) + offset).clip(0, 255)
        output += w * R

    R = image[:, :, 2].astype(np.float32)
    G = image[:, :, 1].astype(np.float32)
    B = image[:, :, 0].astype(np.float32)

    R = (R * (output / (I + 1e-6))).clip(0, 255).astype(np.uint8)
    G = (G * (output / (I + 1e-6))).clip(0, 255).astype(np.uint8)
    B = (B * (output / (I + 1e-6))).clip(0, 255).astype(np.uint8)

    result = cv2.merge((B, G, R))

    return result

if __name__ == "__main__":
    input_image_path = "/home/PaddleSeg/Matting/demo/5.jpg"
    output_folder = "/home/PaddleSeg/Matting/demo/5_msrcr"

    image = cv2.imread(input_image_path)
    scales = [15, 80, 250]
    enhanced_image = automsrcr(image, scales)

    os.makedirs(output_folder, exist_ok=True)
    output_image_path = os.path.join(output_folder, os.path.basename(input_image_path))
    cv2.imwrite(output_image_path, enhanced_image)