import cv2
import numpy as np


"""Sigmoid contrast."""
def random_sigmoid_contrast(image, gain_range=(3, 10), cutoff_range=(0.4, 0.6), per_channel=False):
    result = np.zeros_like(image, dtype=np.float32)

    if per_channel:
        for channel in range(image.shape[2]):
            img_channel = image[:, :, channel].astype(np.float32) / 255.0
            gain = np.random.uniform(gain_range[0], gain_range[1])
            cutoff = np.random.uniform(cutoff_range[0], cutoff_range[1])
            result[:, :, channel] = 255.0 / \
                (1 + np.exp(gain * (cutoff - img_channel)))
    else:
        gain = np.random.uniform(gain_range[0], gain_range[1])
        cutoff = np.random.uniform(cutoff_range[0], cutoff_range[1])
        img = image.astype(np.float32) / 255.0
        result = 255.0 / (1 + np.exp(gain * (cutoff - img)))

    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

def sigmoid_contrast(image, input_annotation, gain_range=(3, 10), cutoff_range=(0.4, 0.6), per_channel=False):
    result = np.zeros_like(image, dtype=np.float32)

    if per_channel:
        for channel in range(image.shape[2]):
            img_channel = image[:, :, channel].astype(np.float32) / 255.0
            gain = np.random.uniform(gain_range[0], gain_range[1])
            cutoff = np.random.uniform(cutoff_range[0], cutoff_range[1])
            result[:, :, channel] = 255.0 / \
                (1 + np.exp(gain * (cutoff - img_channel)))
    else:
        gain = np.random.uniform(gain_range[0], gain_range[1])
        cutoff = np.random.uniform(cutoff_range[0], cutoff_range[1])
        img = image.astype(np.float32) / 255.0
        result = 255.0 / (1 + np.exp(gain * (cutoff - img)))

    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result, input_annotation
"""Sigmoid contrast."""


"""Histogram equalization."""
def random_apply_clahe(image, clip_limit=(1, 10), per_channel=True):
    has_alpha = image.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image[:, :, :3], image[:, :, 3]
    else:
        bgr_image = image

    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 创建一个CLAHE对象
    clahe = cv2.createCLAHE(
        clipLimit=np.random.uniform(clip_limit[0], clip_limit[1]))

    if per_channel:
        l = clahe.apply(l)
        a = clahe.apply(a)
        b = clahe.apply(b)
    else:
        l = clahe.apply(l)

    # 将处理后的通道合并为一个图像
    lab = cv2.merge((l, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if has_alpha:
        result = cv2.merge([result, alpha_channel])

    return result

def apply_clahe(image, input_annotation, clip_limit=(1, 10), per_channel=True):
    has_alpha = image.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image[:, :, :3], image[:, :, 3]
    else:
        bgr_image = image

    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 创建一个CLAHE对象
    clahe = cv2.createCLAHE(
        clipLimit=np.random.uniform(clip_limit[0], clip_limit[1]))

    if per_channel:
        l = clahe.apply(l)
        a = clahe.apply(a)
        b = clahe.apply(b)
    else:
        l = clahe.apply(l)

    # 将处理后的通道合并为一个图像
    lab = cv2.merge((l, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if has_alpha:
        result = cv2.merge([result, alpha_channel])

    return result, input_annotation
"""Histogram equalization."""
