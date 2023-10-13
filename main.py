import os
import cv2
import random
import numpy as np

from augment.geometric import *
from augment.flip import *
from augment.arithmetic import *
from augment.blur import *
from augment.color import *
from augment.contrast import *
from augment.imgcorruptlike import *

class BatchAugmenter:
    def __init__(self, annotation_type='voc'):
        self.annotation_type = annotation_type
        self.augmentations = []

    def add_augmentations(self, augment_func, *args, **kwargs):
        self.augmentations.append((augment_func, args, kwargs))

    def process(self, input_image_folder, input_annotation_folder, output_image_folder, output_annotation_folder):
        # Create output folders if they do not exist
        os.makedirs(output_image_folder, exist_ok=True)
        os.makedirs(output_annotation_folder, exist_ok=True)

        # Process each image and its corresponding annotation
        for image_file in os.listdir(input_image_folder):
            image_path = os.path.join(input_image_folder, image_file)
            image = cv2.imread(image_path)

            # Load the corresponding annotation
            annotation_file = os.path.splitext(
                image_file)[0] + ('.txt' if self.annotation_type == 'yolo' else '.xml')
            annotation_path = os.path.join(
                input_annotation_folder, annotation_file)

            if self.annotation_type == 'yolo':
                with open(annotation_path, 'r') as f:
                    input_annotation = f.readlines()
            else:
                input_annotation = parse(annotation_path)

            # Apply augmentations
            for augment_func, args, kwargs in self.augmentations:
                image, input_annotation = augment_func(
                    image, input_annotation, *args, **kwargs)

            # Save the output image
            output_image_path = os.path.join(output_image_folder, image_file)
            cv2.imwrite(output_image_path, image)

            # Save the output annotation
            output_annotation_file = os.path.splitext(
                image_file)[0] + ('.txt' if self.annotation_type == 'yolo' else '.xml')
            output_annotation_path = os.path.join(
                output_annotation_folder, output_annotation_file)
            if self.annotation_type == 'yolo':
                with open(output_annotation_path, 'w') as f:
                    f.writelines(input_annotation)
            else:
                input_annotation.write(output_annotation_path)


class ImageBlender:
    def __init__(self, foreground_dir, background_dir, output_image_path, output_annotation_path, repeat_range=(1, 1), 
                 class_name='person', show_bbox_on_image=False, max_overlap=0.1,input_annotation_path=None,
                 input_foreground_annotation_path=None, scale_range=(0.5, 1.0), max_output_size=(640, 640)):
        self.foreground_dir = foreground_dir
        self.background_dir = background_dir
        self.output_image_path = output_image_path
        self.output_annotation_path = output_annotation_path
        self.repeat_range = repeat_range
        self.class_name = class_name
        self.show_bbox_on_image = show_bbox_on_image
        self.max_overlap = max_overlap
        self.input_annotation_path = input_annotation_path
        self.input_foreground_annotation_path = input_foreground_annotation_path
        self.scale_range = scale_range
        self.max_output_size = max_output_size

        self.augmentations = []

        os.makedirs(output_image_path, exist_ok=True)
        os.makedirs(output_annotation_path, exist_ok=True)

    def add_augmentations(self, augment_func, *args, **kwargs):
        self.augmentations.append((augment_func, args, kwargs))

    def _parse_voc_annotation(self, input_xml_path):
        tree = ET.parse(input_xml_path)
        root = tree.getroot()

        bboxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            bboxes.append((xmin, ymin, xmax, ymax))

        return bboxes

    def _parse_yolo_annotation(self, input_txt_path, foreground_image_path):
        image = cv2.imread(foreground_image_path)
        img_width = image.shape[1]
        img_height = image.shape[0]
        with open(input_txt_path, "r") as file:
            lines = file.readlines()

        bboxes = []
        for line in lines:
            line = line.strip()
            elements = line.split()

            if len(elements) != 5:
                continue

            class_id, x_center, y_center, width, height = map(float, elements)

            xmin = int((x_center - width / 2) * img_width)
            ymin = int((y_center - height / 2) * img_height)
            xmax = int((x_center + width / 2) * img_width)
            ymax = int((y_center + height / 2) * img_height)

            bboxes.append((xmin, ymin, xmax, ymax))

        return bboxes

    def _apply_augmentations(self, images):
        if images is None:
            raise FileNotFoundError(f"Input image file not found")

        for augment_func, args, kwargs in self.augmentations:
            images = augment_func(images, *args, **kwargs)
        
        return images
    
    def _calculate_iou(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)

        # 计算相交区域的坐标
        xi1, yi1, xi2, yi2 = max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4)

        if xi1 >= xi2 or yi1 >= yi2:
            return 0

        intersection_area = (xi2 - xi1) * (yi2 - yi1)
        union_area = area1 + area2 - intersection_area
        iou = intersection_area / union_area

        # 根据边界框的面积，计算一个大小比例
        size_ratio = min(area1, area2) / max(area1, area2)

        # 使用大小比例调整IoU
        adjusted_iou = iou * size_ratio

        return adjusted_iou


    def _paste_foreground(self, foreground, background, x_offset, y_offset, bboxes, max_overlap):
        h, w = foreground.shape[:2]
        blended = background.copy()
        foreground_area = blended[y_offset:y_offset+h, x_offset:x_offset+w]

        if foreground.shape[2] == 4:  # 如果前景图像有 alpha 通道
            alpha_channel = foreground[:, :, 3] / 255.0
            alpha_3ch = np.stack([alpha_channel] * 3, axis=-1)
            foreground = foreground[:, :, :3]  # 只使用前三个颜色通道
        else:
            alpha_3ch = 1.0

        new_bbox = (x_offset, y_offset, x_offset + w, y_offset + h)

        for bbox in bboxes:
            iou = self._calculate_iou(new_bbox, bbox)
            if iou > max_overlap:
                return None, None

        blended[y_offset:y_offset+h, x_offset:x_offset+w] = alpha_3ch * \
            foreground + (1 - alpha_3ch) * foreground_area

        return blended, new_bbox

    def _paste_foreground_with_annotation(self, foreground, background, x_offset, y_offset, bboxes, foreground_bboxes, max_overlap):
        h, w = foreground.shape[:2]
        blended = background.copy()
        foreground_area = blended[y_offset:y_offset+h, x_offset:x_offset+w]

        if foreground.shape[2] == 4:  # 如果前景图像有 alpha 通道
            alpha_channel = foreground[:, :, 3] / 255.0
            alpha_3ch = np.stack([alpha_channel] * 3, axis=-1)
            foreground = foreground[:, :, :3]  # 只使用前三个颜色通道
        else:
            alpha_3ch = 1.0

        new_bbox = (x_offset, y_offset, x_offset + w, y_offset + h)

        for bbox in bboxes:
            iou = self._calculate_iou(new_bbox, bbox)
            if iou > max_overlap:
                return None, None, None

        blended[y_offset:y_offset+h, x_offset:x_offset+w] = alpha_3ch * \
            foreground + (1 - alpha_3ch) * foreground_area

        # 计算新的前景标注框位置
        new_foreground_bboxes = [(x_min + x_offset, y_min + y_offset, x_max + x_offset, y_max + y_offset) for x_min, y_min, x_max, y_max in foreground_bboxes]
        # new_foreground_bboxes = new_foreground_bboxes[0]

        return blended, new_bbox, new_foreground_bboxes   

    def _create_voc_annotation(self, output_filename, img_shape, bboxes):
        xml_path = os.path.join(
            self.output_annotation_path, f"{output_filename.split('.')[0]}.xml")

        with open(xml_path, "w") as f:
            f.write("<annotation>\n")
            f.write("    <folder>{}</folder>\n".format('smoke_detection'))
            f.write("    <filename>{}</filename>\n".format(output_filename))
            f.write("    <size>\n")
            f.write("        <width>{}</width>\n".format(img_shape[1]))
            f.write("        <height>{}</height>\n".format(img_shape[0]))
            f.write("        <depth>{}</depth>\n".format(img_shape[2]))
            f.write("    </size>\n")
            for bbox in bboxes:
                f.write("    <object>\n")
                f.write("        <name>{}</name>\n".format(self.class_name))
                f.write("        <bndbox>\n")
                f.write("            <xmin>{}</xmin>\n".format(bbox[0]))
                f.write("            <ymin>{}</ymin>\n".format(bbox[1]))
                f.write("            <xmax>{}</xmax>\n".format(bbox[2]))
                f.write("            <ymax>{}</ymax>\n".format(bbox[3]))
                f.write("        </bndbox>\n")
                f.write("    </object>\n")
            f.write("</annotation>")


    def _generate_random_color(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def blend_images(self):
        """将前景图像粘贴到背景图像上"""
        
        background_paths = [os.path.join(self.background_dir, img) for img in os.listdir(
            self.background_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

        for background_path in background_paths:
            background = cv2.imread(background_path)
            print(f"Processed {background_path}")
            bboxes = []

            # 如果提供了输入的标注文件，读取边界框信息并添加到bboxes列表中
            if self.input_annotation_path is not None:
                input_xml_path = os.path.join(
                    self.input_annotation_path, f"{os.path.splitext(os.path.basename(background_path))[0]}.xml")
                if os.path.exists(input_xml_path):
                    existing_bboxes = self._parse_voc_annotation(input_xml_path)
                    for bbox in existing_bboxes:
                        bboxes.append((bbox, (0, 0, 255)))

            foreground_paths = [os.path.join(self.foreground_dir, img) for img in os.listdir(self.foreground_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

            # 每次随机抽取的前景图片数量范围
            foreground_sample_range = (3, 6)

            # 随机选择一部分前景图片
            num_foreground_samples = random.randint(foreground_sample_range[0], foreground_sample_range[1])
            num_foreground_samples = min(num_foreground_samples, len(foreground_paths))
            selected_foreground_paths = random.sample(foreground_paths, num_foreground_samples)

            foreground_max_overlaps = {foreground_path: random.uniform(0, self.max_overlap) for foreground_path in selected_foreground_paths}

            # 为每个前景生成随机颜色
            foreground_colors = {foreground_path: self._generate_random_color() for foreground_path in selected_foreground_paths}

            for foreground_path in selected_foreground_paths:
                repeat_times = random.randint(self.repeat_range[0], self.repeat_range[1])
                for _ in range(repeat_times):
                    foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)

                    # 图像增强
                    foreground = self._apply_augmentations(foreground)

                    scale_factor = min(background.shape[0] / foreground.shape[0],
                                    background.shape[1] / foreground.shape[1])
                    scale_factor = min(scale_factor, 1)
                    resized_foreground = cv2.resize(foreground, None, fx=scale_factor,
                                                    fy=scale_factor, interpolation=cv2.INTER_AREA)

                    max_attempts = 10
                    for _ in range(max_attempts):
                        max_x_offset = background.shape[1] - resized_foreground.shape[1]
                        max_y_offset = background.shape[0] - resized_foreground.shape[0]
                        x_offset = random.randint(0, max_x_offset)
                        y_offset = random.randint(0, max_y_offset)

                        blended, new_bbox = self._paste_foreground(resized_foreground, background,
                        x_offset, y_offset, [bbox[0] for bbox in bboxes], max_overlap=foreground_max_overlaps[foreground_path])
                        if blended is not None:
                            break

                    if blended is None:
                        continue

                    background = blended
                    bboxes.append((new_bbox, foreground_colors[foreground_path]))

            output_filename = f"{os.path.splitext(os.path.basename(background_path))[0]}{os.path.splitext(background_path)[1]}"
            cv2.imwrite(f"{self.output_image_path}/{output_filename}", background)
            print(f"Processed {self.output_image_path}/{output_filename}")

            self._create_voc_annotation(output_filename, background.shape, [bbox[0] for bbox in bboxes])

            if self.show_bbox_on_image:
                for bbox, color in bboxes:
                    cv2.rectangle(
                        background, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.imwrite(f"{self.output_image_path}/{output_filename}", background)

    def blend_images_with_annotation(self):
        """将前景图片混合到背景图片上，输入的前景图片必须带有标注信息"""

        background_paths = [os.path.join(self.background_dir, img) for img in os.listdir(
            self.background_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

        for background_path in background_paths:
            background = cv2.imread(background_path)
            print(f"Processed {background_path}")
            bboxes = []
            foreground_bbox_list = []  # 用于存储每次迭代中更新的前景本身的标注框

            # 如果提供了输入的标注文件，读取边界框信息并添加到bboxes列表中
            if self.input_annotation_path is not None:
                input_xml_path = os.path.join(
                    self.input_annotation_path, f"{os.path.splitext(os.path.basename(background_path))[0]}.xml")
                if os.path.exists(input_xml_path):
                    existing_bboxes = self._parse_voc_annotation(input_xml_path)
                    for bbox in existing_bboxes:
                        bboxes.append((bbox, (0, 0, 255)))

            foreground_paths = [os.path.join(self.foreground_dir, img) for img in os.listdir(
                self.foreground_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

            # 每次随机抽取的前景图片数量范围
            foreground_sample_range = (3, 6)

            # 随机选择一部分前景图片
            num_foreground_samples = random.randint(foreground_sample_range[0], foreground_sample_range[1])
            num_foreground_samples = min(num_foreground_samples, len(foreground_paths))
            selected_foreground_paths = random.sample(foreground_paths, num_foreground_samples)

            foreground_max_overlaps = {foreground_path: random.uniform(
                0, self.max_overlap) for foreground_path in selected_foreground_paths}

            # 为每个前景生成随机颜色
            foreground_colors = {foreground_path: self._generate_random_color(
            ) for foreground_path in selected_foreground_paths}

            for foreground_path in selected_foreground_paths:
                # 读取前景标注框
                foreground_annotation_path = os.path.join(
                    self.input_foreground_annotation_path, f"{os.path.splitext(os.path.basename(foreground_path))[0]}.txt")

                repeat_times = random.randint(self.repeat_range[0], self.repeat_range[1])

                for _ in range(repeat_times):
                    foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)

                    # 控制最大缩放因子
                    scale_factor = min(background.shape[0] / foreground.shape[0],
                                       background.shape[1] / foreground.shape[1])
                    scale_factor = min(scale_factor, 1)

                    # 随机缩放前景图片
                    res_foreground, res_bboxes = random_scale_image_with_bbox(foreground, foreground_annotation_path, 
                        self.scale_range, self.max_output_size)

                    # 图像增强
                    augment_foreground = self._apply_augmentations(res_foreground)

                    max_attempts = 10
                    for _ in range(max_attempts):

                        max_x_offset = background.shape[1] - augment_foreground.shape[1]
                        max_y_offset = background.shape[0] - augment_foreground.shape[0]
                        x_offset = random.randint(0, max_x_offset)
                        y_offset = random.randint(0, max_y_offset)
                        
                        blended, new_bbox, new_foreground_bboxes = self._paste_foreground_with_annotation(augment_foreground, background, x_offset, y_offset, 
                        [bbox for bbox in foreground_bbox_list], res_bboxes, max_overlap=foreground_max_overlaps[foreground_path])
                        if blended is not None:
                            break

                    if blended is None:
                        continue

                    background = blended
                    for item in new_foreground_bboxes:
                        bboxes.append((item, foreground_colors[foreground_path]))
                    foreground_bbox_list.append(new_bbox)  # 将新的前景边界框添加到列表中

            if self.show_bbox_on_image:
                for bbox, color in bboxes:
                    cv2.rectangle(
                        background, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            output_path = os.path.join(self.output_image_path, os.path.basename(background_path))
            cv2.imwrite(output_path, background)

            if self.output_annotation_path is not None:
                output_xml_path = os.path.join(
                    self.output_annotation_path, f"{os.path.splitext(os.path.basename(output_path))[0]}.xml")
                self._create_voc_annotation(
                    output_xml_path, background.shape, [bbox[0] for bbox in bboxes])


# foreground_dir = r"D:\Re_ID\dataset\Augment_copy\apply2\foreground"
# background_dir = r"D:\Re_ID\dataset\Augment_copy\apply2\background"
# # 前景图片所带的标注文件
# input_foreground_annotation_path = r"D:\Re_ID\dataset\Augment_copy\apply2\foreground_annotations"
# # 背景图片所带的标注文件
# input_annotation_path = r"D:\Re_ID\dataset\Augment_copy\test_paste_augment\input_annotations"
# output_image_path = r"D:\Re_ID\dataset\Augment_copy\apply2\output_images"
# output_annotation_path = r"D:\Re_ID\dataset\Augment_copy\apply2\output_annotations"

# # 若前景图片不带标注文件，则将input_foreground_annotation_path设置为None
# # image_blender = ImageBlender(foreground_dir, b ackground_dir, output_image_path, output_annotation_path,
# #                              repeat_range=(1,3), class_name='person', show_bbox_on_image=True, 
# #                              max_overlap=0.1, input_annotation_path=input_annotation_path,
# #                              input_foreground_annotation_path=None, scale_range=None,
# #                              max_output_size=None)

# # 若前景图片带标注文件，则将input_foreground_annotation_path设置为前景图片所带的标注文件路径
# image_blender = ImageBlender(foreground_dir, background_dir, output_image_path, output_annotation_path,
#                              repeat_range=(2,3), class_name='smoke', show_bbox_on_image=False, 
#                              max_overlap=0.00, input_annotation_path=None, 
#                              input_foreground_annotation_path=input_foreground_annotation_path,
#                              scale_range=(0.8, 1.2), max_output_size=(250, 250))

# # 注意：随机缩放增强无法应用于blend_images_with_annotation函数
# # image_blender.add_augmentations(random_scale_image, scale_range=(0.8, 1.2), fixed_aspect_ratio=False)
# # image_blender.add_augmentations(random_rotate_image, angle_range=(-45, 45))
# # image_blender.add_augmentations(flip_image, flip_type='random', percentage=1)
# # image_blender.add_augmentations(add_gaussian_noise, scale=0.2*255, per_channel=True)
# # image_blender.add_augmentations(process_image_brightness, lower=0.5, upper=1.5)
# # image_blender.add_augmentations(fill_image_rectangles, rectangles_numer=(1, 5), area_range=(0.01, 0.1), method='random_intensity')
# # image_blender.add_augmentations(gaussian_blur_augmentation, kernel_size=5, sigma_range=(0.0, 3), p=1)
# # image_blender.add_augmentations(random_blur_augmentation, k_height_range=(2, 11), k_width_range=(2, 11))
# # image_blender.add_augmentations(random_bilateral_blur, d_range=(3, 10), sigma_color_range=(10, 250), sigma_space_range=(10, 250))
# # image_blender.add_augmentations(motion_blur_augmentation, k=15, angle_range=(-45, 45), p=1)
# # image_blender.add_augmentations(random_hsv_transform, hue_shift=30, sat_scale=0.5, val_scale=0.5)
# # image_blender.add_augmentations(add_hue_in_hsv, hue_value_range=(0, 50))
# # image_blender.add_augmentations(random_multiply_hue_and_saturation, hue_range=(0.5, 1.5), saturation_range=(1, 1))
# # image_blender.add_augmentations(random_sigmoid_contrast, gain_range=(3, 10), cutoff_range=(0.4, 0.6), per_channel=False)
# # image_blender.add_augmentations(random_apply_clahe, clip_limit=(1, 10), per_channel=True)

# # image_blender.add_augmentations(apply_fog_augmentation, severity=1, seed=None)
# # image_blender.add_augmentations(apply_frost_augmentation, severity=1, seed=None)
# # image_blender.add_augmentations(apply_snow_augmentation, severity=1, seed=None)
# # image_blender.add_augmentations(apply_spatter_augmentation, severity=1, seed=None)
# # image_blender.add_augmentations(apply_contrast_augmentation, severity=1, seed=None)
# # image_blender.add_augmentations(apply_brightness_augmentation, severity=1, seed=None)
# # image_blender.add_augmentations(apply_saturate_augmentation, severity=1, seed=None)
# # image_blender.add_augmentations(apply_jpeg_compression_augmentation, severity=1, seed=None)
# # image_blender.add_augmentations(apply_pixelate_augmentation, severity=1, seed=None)
# # image_blender.add_augmentations(apply_elastic_transform_augmentation, severity=1, seed=None)

# # image_blender.blend_images()
# # blend_images_with_annotation自带随机缩放增强
# # 若不需要随机缩放增强，可将scale_range设置为（1，1）, max_output_size设置为大于背景图片尺寸的值
# image_blender.blend_images_with_annotation()


input_image_folder = r'D:\projects\Data_augment\Augment_copy\apply1\input_images'
input_annotation_folder = r'D:\projects\Data_augment\Augment_copy\apply1\input_labels'
output_image_folder = r'D:\projects\Data_augment\Augment_copy\apply1\output_image'
output_annotation_folder = r'D:\projects\Data_augment\Augment_copy\apply1\output_annotation'

random_scale = RandomScale(scale_range=(0.5, 1.2), fixed_aspect_ratio=False, annotation_type='yolo')
random_trans = RandomTranslate(x_range=(-0.2, 0.2), y_range=(-0.2, 0.2), annotation_type='yolo')
random_rotate = RandomRotate(angle_range=(-45, 45), annotation_type='yolo')
flip = ImageFlipper(percentage=1, annotation_type='yolo')
batch_augmenter = BatchAugmenter(annotation_type='yolo')

batch_augmenter.add_augmentations(random_scale.augment)
# batch_augmenter.add_augmentations(random_trans.augment)
batch_augmenter.add_augmentations(flip.augment)
# batch_augmenter.add_augmentations(random_rotate.augment)
# batch_augmenter.add_augmentations(gaussian_noise, scale=0.2*255, per_channel=True)
# batch_augmenter.add_augmentations(process_brightness, lower=0.5, upper=1.5)
# batch_augmenter.add_augmentations(fill_rectangles, rectangles_numer=(1, 5), area_range=(0.01, 0.05), method='random_rgb')
# batch_augmenter.add_augmentations(coarse_dropout, dropout_ratio=(0.1, 0.5), size_percent=(0.01, 0.05), per_channel=0, random_state=None)
# batch_augmenter.add_augmentations(gaussian_blur, kernel_size=5, sigma_range=(0.0, 3), p=1)
# batch_augmenter.add_augmentations(random_blur, k_height_range=(2, 11), k_width_range=(2, 11))
# batch_augmenter.add_augmentations(bilateral_blur, d_range=(3, 10), sigma_color_range=(10, 250), sigma_space_range=(10, 250))
# batch_augmenter.add_augmentations(motion_blur, k=15, angle_range=(-45, 45), p=1)
# batch_augmenter.add_augmentations(hsv_transform, hue_shift=30, sat_scale=0.5, val_scale=0.5)
# batch_augmenter.add_augmentations(hue_in_hsv, hue_value_range=(0, 50))
# batch_augmenter.add_augmentations(multiply_hue_and_saturation, hue_range=(0.5, 1.5), saturation_range=(0.5, 1.5))
# batch_augmenter.add_augmentations(sigmoid_contrast, gain_range=(3, 10), cutoff_range=(0.4, 0.6), per_channel=False)

# batch_augmenter.add_augmentations(apply_clahe, clip_limit=(1, 10), per_channel=True)
# batch_augmenter.add_augmentations(apply_frost, severity=3, seed=None)
# batch_augmenter.add_augmentations(apply_fog, severity=3, seed=None)
# batch_augmenter.add_augmentations(apply_snow, severity=3, seed=None)
# batch_augmenter.add_augmentations(apply_spatter, severity=3, seed=None)
# batch_augmenter.add_augmentations(apply_contrast, severity=3, seed=None)
# batch_augmenter.add_augmentations(apply_brightness, severity=3, seed=None)
# batch_augmenter.add_augmentations(apply_saturate, severity=3, seed=None)
# batch_augmenter.add_augmentations(apply_jpeg_compression, severity=3, seed=None)
# batch_augmenter.add_augmentations(apply_pixelate, severity=3, seed=None)
# batch_augmenter.add_augmentations(apply_elastic_transform, severity=3, seed=None)

batch_augmenter.process(input_image_folder, input_annotation_folder, output_image_folder, output_annotation_folder)