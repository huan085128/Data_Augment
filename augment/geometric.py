import xml.etree.ElementTree as ET
import math
import cv2
import os
import random
import numpy as np
from xml.etree.ElementTree import parse
from utils import batch_process_images_trans, is_image_or_path

class RandomScale:
    """Scale an image and its annotations by a random factor."""
    def __init__(self, scale_range=(0.5, 1.5), fixed_aspect_ratio=True, annotation_type='yolo'):
        self.scale_range = scale_range
        self.fixed_aspect_ratio = fixed_aspect_ratio
        self.annotation_type = annotation_type

    def random_scale_image(self, image):
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        height, width = image.shape[:2]
        new_size = (int(width * scale_factor), int(height * scale_factor))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    def random_scale_image_aspect_ratio(self, image):
        width_scale_factor = random.uniform(0.5, 1)
        height_scale_factor = random.uniform(0.5, 1) if self.fixed_aspect_ratio else width_scale_factor
        height, width = image.shape[:2]
        new_size = (int(width * width_scale_factor), int(height * height_scale_factor))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    def paste_on_canvas(self, image, canvas_size):
        # canvas = np.zeros((*canvas_size, 3), dtype=np.uint8)
        canvas = np.full((*canvas_size, 3), 128, dtype=np.uint8)

        # Calculate the offsets
        x_offset = (canvas_size[1] - image.shape[1]) // 2
        y_offset = (canvas_size[0] - image.shape[0]) // 2

        # Calculate the boundaries
        x_start_canvas = max(0, x_offset)
        y_start_canvas = max(0, y_offset)
        x_end_canvas = x_start_canvas + min(canvas_size[1] - x_start_canvas, image.shape[1])
        y_end_canvas = y_start_canvas + min(canvas_size[0] - y_start_canvas, image.shape[0])

        x_start_image = max(0, -x_offset)
        y_start_image = max(0, -y_offset)
        x_end_image = x_start_image + (x_end_canvas - x_start_canvas)
        y_end_image = y_start_image + (y_end_canvas - y_start_canvas)

        # Paste the image onto the canvas
        canvas[y_start_canvas:y_end_canvas, x_start_canvas:x_end_canvas] = image[y_start_image:y_end_image, x_start_image:x_end_image]

        return canvas, (x_offset, y_offset)

    def scale_yolo_annotation(self, annotation_lines, original_size, canvas_size, x_offset, y_offset):
        new_lines = []
        for line in annotation_lines:
            line_parts = line.strip().split(' ')
            class_id = line_parts[0]
            x_center = (float(line_parts[1]) * canvas_size[1] + x_offset) / original_size[1]
            y_center = (float(line_parts[2]) * canvas_size[0] + y_offset) / original_size[0]
            w = float(line_parts[3]) * canvas_size[1] / original_size[1]
            h = float(line_parts[4]) * canvas_size[0] / original_size[0]

            # Add boundary checks based on the canvas size (original image size)
            x_center = max(0, min(canvas_size[1], x_center))
            y_center = max(0, min(canvas_size[0], y_center))
            w = max(0, min(canvas_size[1], w))
            h = max(0, min(canvas_size[0], h))

            new_line = f"{class_id} {x_center} {y_center} {w} {h}\n"
            new_lines.append(new_line)

        return new_lines

    def scale_voc_annotation(self, annotation_file, original_size, new_size, offset):
        tree = annotation_file
        root = tree.getroot()

        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(int(bndbox.find('xmin').text) * new_size[1] / original_size[1] + offset[0])
            ymin = int(int(bndbox.find('ymin').text) * new_size[0] / original_size[0] + offset[1])
            xmax = int(int(bndbox.find('xmax').text) * new_size[1] / original_size[1] + offset[0])
            ymax = int(int(bndbox.find('ymax').text) * new_size[0] / original_size[0] + offset[1])

            # Add boundary checks based on the canvas size (original image size)
            xmin = max(0, min(original_size[1], xmin))
            ymin = max(0, min(original_size[0], ymin))
            xmax = max(0, min(original_size[1], xmax))
            ymax = max(0, min(original_size[0], ymax))

            bndbox.find('xmin').text = str(xmin)
            bndbox.find('ymin').text = str(ymin)
            bndbox.find('xmax').text = str(xmax)
            bndbox.find('ymax').text = str(ymax)

        return tree

    def augment(self, image, input_annotation):
        original_size = image.shape[:2]

        # Randomly scale the image with or without fixed aspect ratio
        if self.fixed_aspect_ratio:
            scaled_image = self.random_scale_image_aspect_ratio(image)
        else:
            scaled_image = self.random_scale_image(image)

        # Paste the scaled image on a canvas with the same size as the original image
        canvas_image, (x_offset, y_offset) = self.paste_on_canvas(
            scaled_image, original_size)

        if input_annotation is not None:
            # Scale the bounding boxes
            if isinstance(input_annotation, list):
                if self.annotation_type == 'yolo':
                    # Scale the bounding boxes
                    scaled_annotation_lines = self.scale_yolo_annotation(
                        input_annotation, original_size, scaled_image.shape[:2], x_offset, y_offset)

                    return canvas_image, scaled_annotation_lines
                
                else:
                    raise ValueError(
                        "Invalid annotation_type. Supported types for list annotations are 'yolo'.")

            elif isinstance(input_annotation, ET.ElementTree):
                if self.annotation_type == 'voc':
                    # Scale the bounding boxes
                    scaled_tree = self.scale_voc_annotation(
                        input_annotation, original_size, scaled_image.shape[:2], (x_offset, y_offset))

                    return canvas_image, scaled_tree
                else:
                    raise ValueError(
                        "Invalid annotation_type. Supported types for XML annotations are 'voc'.")
            
            else:
                raise ValueError(
                    "Invalid annotation format. Supported formats are lists of YOLO annotation strings and ElementTree objects for VOC annotations.")

        return canvas_image


def random_scale_image(image, scale_range=(0.5, 1.5), fixed_aspect_ratio=False):
    if not fixed_aspect_ratio:
        scale_factor = random.uniform(scale_range[0], scale_range[1])
        height, width = image.shape[:2]
        new_size = (int(width * scale_factor), int(height * scale_factor))
    else:
        width_scale_factor = random.uniform(0.5, 1)
        height_scale_factor = random.uniform(0.5, 1)
        height, width = image.shape[:2]
        new_size = (int(width * width_scale_factor), int(height * height_scale_factor))
        
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)


def random_scale_image_with_bbox(image, annotation_path, scale_range=(0.5, 1.5), max_output_size=(800, 800), fixed_aspect_ratio=False):
    height, width = image.shape[:2]

    if not fixed_aspect_ratio:
        scale_factor = random.uniform(scale_range[0], scale_range[1])
        new_size = (int(width * scale_factor), int(height * scale_factor))
    else:
        width_scale_factor = random.uniform(0.5, 1)
        height_scale_factor = random.uniform(0.5, 1)
        new_size = (int(width * width_scale_factor),
                    int(height * height_scale_factor))

    # Check if the new size is larger than max_output_size and adjust if necessary
    if new_size[0] > max_output_size[0] or new_size[1] > max_output_size[1]:
        scale_factor = min(
            max_output_size[0] / new_size[0], max_output_size[1] / new_size[1])
        new_size = (int(new_size[0] * scale_factor),
                    int(new_size[1] * scale_factor))

    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    with open(annotation_path, 'r') as f:
        annotation = f.readlines()

    resized_bboxes = []
    for line in annotation:
        class_id, x_center, y_center, bbox_width, bbox_height = map(
            float, line.strip().split())
        x_center_new = x_center * (new_size[0] / width)
        y_center_new = y_center * (new_size[1] / height)
        bbox_width_new = bbox_width * (new_size[0] / width)
        bbox_height_new = bbox_height * (new_size[1] / height)

        xmin = int((x_center_new - bbox_width_new / 2) * width)
        ymin = int((y_center_new - bbox_height_new / 2) * height)
        xmax = int((x_center_new + bbox_width_new / 2) * width)
        ymax = int((y_center_new + bbox_height_new / 2) * height)

        resized_bboxes.append((xmin, ymin, xmax, ymax))

    return resized_image, resized_bboxes






class RandomTranslate:
    """Randomly translate the image and the bounding boxes."""

    def __init__(self, x_range=(-0.2, 0.2), y_range=(-0.2, 0.2), annotation_type='voc'):
        self.x_range = x_range
        self.y_range = y_range
        self.annotation_type = annotation_type

    def random_translate(self, image, x_translate, y_translate):
        matrix = np.float32([[1, 0, x_translate], [0, 1, y_translate]])
        gray_value = 128  # 设置灰色值，范围在0-255之间。128是中间灰色。
        return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderValue=(gray_value, gray_value, gray_value))

    def random_translate_yolo_annotation(self, annotation_lines, original_size, x_translate, y_translate, image_size):
        new_lines = []
        for line in annotation_lines:
            line_parts = line.strip().split(' ')
            class_id = line_parts[0]
            x_center = float(line_parts[1]) + (x_translate / image_size[1])
            y_center = float(line_parts[2]) + (y_translate / image_size[0])
            w = float(line_parts[3])
            h = float(line_parts[4])

            # Add boundary checks based on the canvas size (original image size)
            x_center = max(0, min(original_size[1], x_center))
            y_center = max(0, min(original_size[0], y_center))
            w = max(0, min(original_size[1], w))
            h = max(0, min(original_size[0], h))

            new_line = f"{class_id} {x_center} {y_center} {w} {h}\n"
            new_lines.append(new_line)

        return new_lines

    def random_translate_voc_annotation(self, annotation_file, original_size, x_translate, y_translate):
        tree = annotation_file
        root = tree.getroot()

        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text) + x_translate
            ymin = int(bndbox.find('ymin').text) + y_translate
            xmax = int(bndbox.find('xmax').text) + x_translate
            ymax = int(bndbox.find('ymax').text) + y_translate

            # Add boundary checks based on the canvas size (original image size)
            xmin = max(0, min(original_size[1], xmin))
            ymin = max(0, min(original_size[0], ymin))
            xmax = max(0, min(original_size[1], xmax))
            ymax = max(0, min(original_size[0], ymax))

            bndbox.find('xmin').text = str(xmin)
            bndbox.find('ymin').text = str(ymin)
            bndbox.find('xmax').text = str(xmax)
            bndbox.find('ymax').text = str(ymax)

        return tree

    def augment(self, image, input_annotation):
        original_size = image.shape[:2]
        height, width = image.shape[:2]

        x_translate = int(random.uniform(self.x_range[0], self.x_range[1]) * width)
        y_translate = int(random.uniform(self.y_range[0], self.y_range[1]) * height)

        translated_image = self.random_translate(image, x_translate, y_translate)

        if input_annotation is not None:
            if isinstance(input_annotation, list):
                if self.annotation_type == 'yolo':
                    translated_annotation_lines = self.random_translate_yolo_annotation(
                        input_annotation, original_size, x_translate, y_translate, (height, width))

                    return translated_image, translated_annotation_lines

                else:
                    raise ValueError(
                        "Invalid annotation_type. Supported types for list annotations are 'yolo'.")

            elif isinstance(input_annotation, ET.ElementTree):
                if self.annotation_type == 'voc':
                    translated_tree = self.random_translate_voc_annotation(
                        input_annotation, original_size, x_translate, y_translate)

                    return translated_image, translated_tree

                else:
                    raise ValueError(
                        "Invalid annotation_type. Supported types for XML annotations are 'voc'.")
            else:
                raise ValueError(
                    "Invalid annotation format. Supported formats are lists of YOLO annotation strings and ElementTree objects for VOC annotations.")
            
        return translated_image


class RandomRotate:
    """Randomly rotate the image and the bounding boxes."""
    def __init__(self, angle_range=(-45, 45), annotation_type='voc'):
        self.angle_range = angle_range
        self.annotation_type = annotation_type

    def random_rotate(self, image, angle):
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(
            image, M, (width, height), borderValue=(0, 0, 0))

        return rotated_image

    def rotate_yolo_annotation(self, annotation_lines, original_size, angle, image_shape):
        new_lines = []
        for line in annotation_lines:
            line_parts = line.strip().split(' ')
            class_id = line_parts[0]
            x_center, y_center, w, h = [float(x) for x in line_parts[1:]]

            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2

            points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

            height, width = image_shape[:2]
            center = np.array([width // 2, height // 2], dtype=np.float32)
            angle_rad = -math.radians(angle)

            R = np.array([[math.cos(angle_rad), -math.sin(angle_rad)],
                          [math.sin(angle_rad),  math.cos(angle_rad)]], dtype=np.float32)

            points_rotated = np.dot(points - center, R.T) + center

            x1_rotated, y1_rotated = np.min(points_rotated, axis=0)
            x2_rotated, y2_rotated = np.max(points_rotated, axis=0)

            x_center_rotated = (x1_rotated + x2_rotated) / 2
            y_center_rotated = (y1_rotated + y2_rotated) / 2
            w_rotated = x2_rotated - x1_rotated
            h_rotated = y2_rotated - y1_rotated

            # Add boundary checks based on the canvas size (original image size)
            x_center_rotated = max(0, min(original_size[1], x_center_rotated))
            y_center_rotated = max(0, min(original_size[0], y_center_rotated))
            w_rotated = max(0, min(original_size[1], w_rotated))
            h_rotated = max(0, min(original_size[0], h_rotated))

            new_line = f"{class_id} {x_center_rotated} {y_center_rotated} {w_rotated} {h_rotated}\n"
            new_lines.append(new_line)

        return new_lines

    def rotate_voc_annotation(self, input_annotation_path, original_size, angle, image_shape):
        tree = input_annotation_path
        root = tree.getroot()

        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            x1 = int(bbox.find('xmin').text)
            y1 = int(bbox.find('ymin').text)
            x2 = int(bbox.find('xmax').text)
            y2 = int(bbox.find('ymax').text)

            points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

            height, width = image_shape[:2]
            center = np.array([width // 2, height // 2], dtype=np.float32)
            angle_rad = -math.radians(angle)

            R = np.array([[math.cos(angle_rad), -math.sin(angle_rad)],
                          [math.sin(angle_rad),  math.cos(angle_rad)]], dtype=np.float32)

            points_rotated = np.dot(points - center, R.T) + center

            x1_rotated, y1_rotated = np.min(points_rotated, axis=0).astype(int)
            x2_rotated, y2_rotated = np.max(points_rotated, axis=0).astype(int)

            # Add boundary checks based on the canvas size (original image size)
            x1_rotated = max(0, min(original_size[1], x1_rotated))
            y1_rotated = max(0, min(original_size[0], y1_rotated))
            x2_rotated = max(0, min(original_size[1], x2_rotated))
            y2_rotated = max(0, min(original_size[0], y2_rotated))

            bbox.find('xmin').text = str(x1_rotated)
            bbox.find('ymin').text = str(y1_rotated)
            bbox.find('xmax').text = str(x2_rotated)
            bbox.find('ymax').text = str(y2_rotated)

        return tree

    def augment(self, image, input_annotation):
        original_size = image.shape[:2]
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        rotated_image = self.random_rotate(image, angle)

        if input_annotation is not None:
            # Rotate the bounding boxes
            if isinstance(input_annotation, list):
                if self.annotation_type == 'yolo':
                    rotated_annotation = self.rotate_yolo_annotation(
                        input_annotation, original_size, angle, image.shape)
                else:
                    raise ValueError(
                        "Invalid mode. Supported modes for list annotations are 'yolo'.")
            elif isinstance(input_annotation, ET.ElementTree):
                if self.annotation_type == 'voc':
                    rotated_annotation = self.rotate_voc_annotation(input_annotation, original_size, angle, image.shape)
                else:
                    raise ValueError(
                        "Invalid mode. Supported modes for XML annotations are 'voc'.")
            else:
                raise ValueError(
                    "Invalid annotation format. Supported formats are lists of YOLO annotation strings and ElementTree objects for VOC annotations.")
        else:
            return rotated_image

        return rotated_image, rotated_annotation

def random_rotate_image(image, angle_range=(-45, 45)):
    """Randomly rotate the input image and resize to fit the rotated content."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    angle = random.uniform(angle_range[0], angle_range[1])

    # 计算旋转角度的弧度值
    angle_rad = abs(math.radians(angle))

    # 计算旋转后的新尺寸
    new_width = int(math.ceil(width * math.cos(angle_rad) + height * math.sin(angle_rad)))
    new_height = int(math.ceil(height * math.cos(angle_rad) + width * math.sin(angle_rad)))

    # 计算旋转矩阵，并应用仿射变换
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_width - width) / 2
    M[1, 2] += (new_height - height) / 2
    rotated_image = cv2.warpAffine(image, M, (new_width, new_height), borderValue=(0, 0, 0))

    return rotated_image
