import os 
import cv2
import random
import numpy as np


"""Process an image by applying gaussian blur"""
def gaussian_blur_augmentation(image, kernel_size=5, sigma_range=(0.0, 3), p=1):
    if random.random() < p:
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        img = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    return img

def gaussian_blur(image, input_annotation, kernel_size=5, sigma_range=(0.0, 3), p=1):
    if random.random() < p:
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        img = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    return img, input_annotation
"""Process an image by applying gaussian blur"""


"""Apply random blur to an image."""
def random_blur_augmentation(image, k_height_range=(2, 11), k_width_range=(2, 11)):
    k_height = np.random.randint(k_height_range[0], k_height_range[1] + 1)
    k_width = np.random.randint(k_width_range[0], k_width_range[1] + 1)
    image = cv2.blur(image, (k_height, k_width))
    
    return image

def random_blur(image, input_annotation, k_height_range=(2, 11), k_width_range=(2, 11)):
    k_height = np.random.randint(k_height_range[0], k_height_range[1] + 1)
    k_width = np.random.randint(k_width_range[0], k_width_range[1] + 1)
    image = cv2.blur(image, (k_height, k_width))

    return image, input_annotation
"""Apply random blur to an image."""


"""Apply random bilateral blur to an image."""
def random_bilateral_blur(image, d_range=(3, 10), sigma_color_range=(10, 250), sigma_space_range=(10, 250)):
    d = np.random.randint(d_range[0], d_range[1] + 1)
    sigma_color = np.random.uniform(sigma_color_range[0], sigma_color_range[1])
    sigma_space = np.random.uniform(sigma_space_range[0], sigma_space_range[1])

    has_alpha = image.shape[2] == 4  # 检查图像是否具有 Alpha 通道
    if has_alpha:
        alpha_channel = image[:, :, 3]  # 保留 Alpha 通道
        image = image[:, :, :3]  # 只保留 BGR 通道

    output_image = np.zeros_like(image)
    cv2.bilateralFilter(image, d, sigma_color, sigma_space, dst=output_image)

    if has_alpha:
        output_image = cv2.merge([output_image, alpha_channel])  # 将 Alpha 通道添加回去

    return output_image

def bilateral_blur(image, input_annotation, d_range=(3, 10), sigma_color_range=(10, 250), sigma_space_range=(10, 250)):
    d = np.random.randint(d_range[0], d_range[1] + 1)
    sigma_color = np.random.uniform(sigma_color_range[0], sigma_color_range[1])
    sigma_space = np.random.uniform(sigma_space_range[0], sigma_space_range[1])

    has_alpha = image.shape[2] == 4  # 检查图像是否具有 Alpha 通道
    if has_alpha:
        alpha_channel = image[:, :, 3]  # 保留 Alpha 通道
        image = image[:, :, :3]  # 只保留 BGR 通道

    output_image = np.zeros_like(image)
    cv2.bilateralFilter(image, d, sigma_color, sigma_space, dst=output_image)

    if has_alpha:
        output_image = cv2.merge([output_image, alpha_channel])  # 将 Alpha 通道添加回去

    return output_image, input_annotation
"""Apply random bilateral blur to an image."""


"""Apply random median blur to an image."""
def motion_blur_augmentation(image, k=15, angle_range=(-45, 45), p=1):
    if random.random() < p:

        angle = random.uniform(angle_range[0], angle_range[1])
        M = cv2.getRotationMatrix2D((k/2, k/2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(k))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (k, k))
        motion_blur_kernel = motion_blur_kernel / k
        
        img = cv2.filter2D(image, -1, motion_blur_kernel)

    return img

def motion_blur(image, input_annotation, k=15, angle_range=(-45, 45), p=1):
    if random.random() < p:

        angle = random.uniform(angle_range[0], angle_range[1])
        M = cv2.getRotationMatrix2D((k/2, k/2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(k))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (k, k))
        motion_blur_kernel = motion_blur_kernel / k

        img = cv2.filter2D(image, -1, motion_blur_kernel)

    return img, input_annotation
"""Apply random median blur to an image."""
