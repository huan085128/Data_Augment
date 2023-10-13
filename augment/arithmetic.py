import os
import cv2
import random
import numpy as np
from utils import batch_process_images, write_results

"""Add gaussian noise to an image."""
def add_gaussian_noise(image, scale=0.2*255, per_channel=True):
    mean = 0
    if per_channel:
        noise = np.random.normal(mean, scale, image.shape)
    else:
        h, w, _ = image.shape
        noise = np.random.normal(mean, scale, (h, w, 1))
        noise = np.repeat(noise, 3, axis=2)

    noisy_image = np.clip(image.astype(np.float32) +
                          noise, 0, 255).astype(np.uint8)
    return noisy_image

def gaussian_noise(image, input_annotation, scale=0.2*255, per_channel=True):
    """Add gaussian noise to an image."""
    mean = 0
    if per_channel:
        noise = np.random.normal(mean, scale, image.shape)
    else:
        h, w, _ = image.shape
        noise = np.random.normal(mean, scale, (h, w, 1))
        noise = np.repeat(noise, 3, axis=2)

    noisy_image = np.clip(image.astype(np.float32) +
                          noise, 0, 255).astype(np.uint8)
    return noisy_image, input_annotation
"""Add gaussian noise to an image."""


"""Process an image by adjusting its brightness"""
def process_image_brightness(image, lower=0.5, upper=1.5):

    adjusted_img = image.copy().astype(np.float32)

    factor = np.random.uniform(lower, upper)

    adjusted_img *= factor
    adjusted_img = np.clip(adjusted_img, 0, 255).astype(np.uint8)

    return adjusted_img

def process_brightness(image, input_annotation, lower=0.5, upper=1.5):

    adjusted_img = image.copy().astype(np.float32)

    factor = np.random.uniform(lower, upper)

    adjusted_img *= factor
    adjusted_img = np.clip(adjusted_img, 0, 255).astype(np.uint8)

    return adjusted_img, input_annotation
"""Process an image by adjusting its brightness"""


"""Fill the image with random rectangles of the given method"""
def fill_image_rectangles(image, rectangles_numer=(1, 5), 
                    area_range=(0.01, 0.1), 
                    method='random_intensity'):
    img = image.copy()
    h, w, c = img.shape

    rectangles = random.randint(rectangles_numer[0], rectangles_numer[1])
    for _ in range(rectangles):
        area = random.uniform(area_range[0], area_range[1]) * h * w
        aspect_ratio = random.uniform(0.3, 3)

        rect_w = int(np.sqrt(area / aspect_ratio))
        rect_h = int(np.sqrt(area * aspect_ratio))

        x = random.randint(0, w - rect_w)
        y = random.randint(0, h - rect_h)

        if method == 'gaussian_noise':
            noise = np.random.normal(rect_h, rect_w, c).astype(np.uint8)
            img[y:y+rect_h, x:x+rect_w] = noise
        elif method == 'random_intensity':
            intensity = random.randint(0, 255)
            img[y:y+rect_h, x:x+rect_w] = intensity
        elif method == 'random_rgb':
            color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
            img[y:y+rect_h, x:x+rect_w] = color

    return img

def fill_rectangles(image, input_annotation, rectangles_numer=(1, 5),
                    area_range=(0.01, 0.1),
                    method='random_intensity'):
    img = image.copy()
    h, w, c = img.shape

    rectangles = random.randint(rectangles_numer[0], rectangles_numer[1])
    for _ in range(rectangles):
        area = random.uniform(area_range[0], area_range[1]) * h * w
        aspect_ratio = random.uniform(0.3, 3)

        rect_w = int(np.sqrt(area / aspect_ratio))
        rect_h = int(np.sqrt(area * aspect_ratio))

        x = random.randint(0, w - rect_w)
        y = random.randint(0, h - rect_h)

        if method == 'gaussian_noise':
            noise = np.random.normal(rect_h, rect_w, c).astype(np.uint8)
            img[y:y+rect_h, x:x+rect_w] = noise
        elif method == 'random_intensity':
            intensity = random.randint(0, 255)
            img[y:y+rect_h, x:x+rect_w] = intensity
        elif method == 'random_rgb':
            color = np.array(
                [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
            img[y:y+rect_h, x:x+rect_w] = color

    return img, input_annotation
"""Fill the image with random rectangles of the given method"""


"""Coarse dropout of the given image."""
def coarse_image_dropout(image, dropout_ratio, size_percent, per_channel=0, random_state=None):
    if random_state is None:
        random_state = np.random.default_rng()

    height, width, channels = image.shape

    if isinstance(dropout_ratio, tuple):
        dropout_ratio = random_state.uniform(
            dropout_ratio[0], dropout_ratio[1])

    if isinstance(size_percent, tuple):
        size_percent = random_state.uniform(size_percent[0], size_percent[1])

    scaled_height = int(height * size_percent)
    scaled_width = int(width * size_percent)

    # Create a mask with the given dropout ratio
    mask = random_state.random((scaled_height, scaled_width)) < dropout_ratio
    mask = cv2.resize(mask.astype(np.uint8), (width, height),
                      interpolation=cv2.INTER_NEAREST)

    if isinstance(per_channel, float):
        channel_masks = [mask if random_state.random(
        ) < per_channel else np.zeros_like(mask) for _ in range(channels)]
        mask = np.stack(channel_masks, axis=-1)

    image_with_dropout = image.copy()
    image_with_dropout[mask == 1] = 0
    
    return image_with_dropout

def coarse_dropout(image, input_annotation, dropout_ratio=(0.1, 0.5), 
                   size_percent=(0.01, 0.05), per_channel=0, random_state=None):
    if random_state is None:
        random_state = np.random.default_rng()

    height, width, channels = image.shape

    if isinstance(dropout_ratio, tuple):
        dropout_ratio = random_state.uniform(
            dropout_ratio[0], dropout_ratio[1])

    if isinstance(size_percent, tuple):
        size_percent = random_state.uniform(size_percent[0], size_percent[1])

    scaled_height = int(height * size_percent)
    scaled_width = int(width * size_percent)

    # Create a mask with the given dropout ratio
    mask = random_state.random((scaled_height, scaled_width)) < dropout_ratio
    mask = cv2.resize(mask.astype(np.uint8), (width, height),
                      interpolation=cv2.INTER_NEAREST)

    if isinstance(per_channel, float):
        channel_masks = [mask if random_state.random(
        ) < per_channel else np.zeros_like(mask) for _ in range(channels)]
        mask = np.stack(channel_masks, axis=-1)

    image_with_dropout = image.copy()
    image_with_dropout[mask == 1] = 0
    
    return image_with_dropout, input_annotation
"""Coarse dropout of the given image."""
