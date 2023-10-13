import cv2
import numpy as np
import random

"""Randomly transform the image in HSV space."""
def random_hsv_transform(image, hue_shift=30, sat_scale=0.5, val_scale=0.5):
    has_alpha = image.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image[:, :, :3], image[:, :, 3]
    else:
        bgr_image = image

    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv_image)
    h_shift = random.randint(-hue_shift, hue_shift)
    s_scale = random.uniform(1 - sat_scale, 1 + sat_scale)
    v_scale = random.uniform(1 - val_scale, 1 + val_scale)

    h = (h + h_shift) % 180
    s = np.clip(s * s_scale, 0, 255)
    v = np.clip(v * v_scale, 0, 255)

    h = h.astype(np.uint8)
    s = s.astype(np.uint8)
    v = v.astype(np.uint8)

    transformed_image = cv2.merge([h, s, v])
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_HSV2BGR)

    if has_alpha:
        transformed_image = cv2.merge([transformed_image, alpha_channel])

    return transformed_image

def hsv_transform(image, input_annotation, hue_shift=30, sat_scale=0.5, val_scale=0.5):
    has_alpha = image.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image[:, :, :3], image[:, :, 3]
    else:
        bgr_image = image

    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv_image)
    h_shift = random.randint(-hue_shift, hue_shift)
    s_scale = random.uniform(1 - sat_scale, 1 + sat_scale)
    v_scale = random.uniform(1 - val_scale, 1 + val_scale)

    h = (h + h_shift) % 180
    s = np.clip(s * s_scale, 0, 255)
    v = np.clip(v * v_scale, 0, 255)

    h = h.astype(np.uint8)
    s = s.astype(np.uint8)
    v = v.astype(np.uint8)

    transformed_image = cv2.merge([h, s, v])
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_HSV2BGR)

    if has_alpha:
        transformed_image = cv2.merge([transformed_image, alpha_channel])

    return transformed_image, input_annotation
"""Randomly transform the image in HSV space."""

"""Add a random value to the Hue channel in HSV space."""
def add_hue_in_hsv(image, hue_value_range=(0, 50)):
    has_alpha = image.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image[:, :, :3], image[:, :, 3]
    else:
        bgr_image = image
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    random_hue_value = np.random.randint(hue_value_range[0], hue_value_range[1])
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + random_hue_value) % 180

    result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    if has_alpha:
        result_image = cv2.merge([result_image, alpha_channel])
    
    return result_image

def hue_in_hsv(image, input_annotation, hue_value_range=(0, 50)):
    has_alpha = image.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image[:, :, :3], image[:, :, 3]
    else:
        bgr_image = image
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    random_hue_value = np.random.randint(hue_value_range[0], hue_value_range[1])
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + random_hue_value) % 180

    result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    if has_alpha:
        result_image = cv2.merge([result_image, alpha_channel])
    
    return result_image, input_annotation
"""Add a random value to the Hue channel in HSV space."""


"""Randomly multiply the Hue and Saturation channels in HSV space."""
def random_multiply_hue_and_saturation(image, hue_range=(0.5, 1.5), saturation_range=(0.5, 1.5)):
    has_alpha = image.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image[:, :, :3], image[:, :, 3]
    else:
        bgr_image = image
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    hue_multiplier = np.random.uniform(hue_range[0], hue_range[1])
    saturation_multiplier = np.random.uniform(
        saturation_range[0], saturation_range[1])

    hsv_image = hsv_image.astype(np.float32)

    hsv_image[:, :, 0] *= hue_multiplier
    hsv_image[:, :, 1] *= saturation_multiplier

    hsv_image[:, :, 0] = np.clip(hsv_image[:, :, 0], 0, 180)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)

    hsv_image = hsv_image.astype(np.uint8)
    result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    if has_alpha:
        result_image = cv2.merge([result_image, alpha_channel])

    return result_image

def multiply_hue_and_saturation(image, input_annotation, hue_range=(0.5, 1.5), saturation_range=(0.5, 1.5)):
    has_alpha = image.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image[:, :, :3], image[:, :, 3]
    else:
        bgr_image = image
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    hue_multiplier = np.random.uniform(hue_range[0], hue_range[1])
    saturation_multiplier = np.random.uniform(
        saturation_range[0], saturation_range[1])

    hsv_image = hsv_image.astype(np.float32)

    hsv_image[:, :, 0] *= hue_multiplier
    hsv_image[:, :, 1] *= saturation_multiplier

    hsv_image[:, :, 0] = np.clip(hsv_image[:, :, 0], 0, 180)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)

    hsv_image = hsv_image.astype(np.uint8)
    result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    if has_alpha:
        result_image = cv2.merge([result_image, alpha_channel])

    return result_image, input_annotation
"""Randomly multiply the Hue and Saturation channels in HSV space."""
