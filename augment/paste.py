import cv2
import os
import random
import numpy as np
from color_transfer import color_transfer
import xml.etree.cElementTree as ET


class ImageBlender:
    def __init__(self, foreground_dir, background_path, output_path, augment=None, repeat_times=1, class_name='person'):
        self.foreground_dir = foreground_dir
        self.background_path = background_path
        self.output_path = output_path
        self.augment = augment
        self.repeat_times = repeat_times
        self.class_name = class_name

        os.makedirs(output_path, exist_ok=True)

    def paste_foreground(self, foreground, background, x_offset, y_offset):
        h, w = foreground.shape[:2]
        blended = background.copy()
        foreground_area = blended[y_offset:y_offset+h, x_offset:x_offset+w]

        if foreground.shape[2] == 4:  # 如果前景图像有 alpha 通道
            alpha_channel = foreground[:, :, 3] / 255.0
            alpha_3ch = np.stack([alpha_channel] * 3, axis=-1)
            foreground = foreground[:, :, :3]  # 只使用前三个颜色通道
        else:
            alpha_3ch = 1.0

        blended[y_offset:y_offset+h, x_offset:x_offset+w] = alpha_3ch * \
            foreground + (1 - alpha_3ch) * foreground_area

        return blended

    def create_voc_annotation(self, output_filename, img_shape, bboxes):
        xml_path = os.path.join(
            self.output_path, f"{output_filename.split('.')[0]}.xml")

        with open(xml_path, "w") as f:
            f.write("<annotation>\n")
            f.write("    <folder>{}</folder>\n".format('person_detection'))
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

    def blend_images(self, *args, **kwargs):
        background = cv2.imread(self.background_path)
        bboxes = []

        # 获取文件夹中的所有前景图片
        foreground_paths = [os.path.join(self.foreground_dir, img) for img in os.listdir(
            self.foreground_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

        for foreground_path in foreground_paths:
            for _ in range(self.repeat_times):
                foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)

                # 图像增强
                if self.augment is not None:
                    foreground = self.augment(*args, **kwargs)

                scale_factor = min(background.shape[0] / foreground.shape[0],
                                   background.shape[1] / foreground.shape[1])
                scale_factor = min(scale_factor, 1)
                resized_foreground = cv2.resize(foreground, None, fx=scale_factor,
                                                fy=scale_factor, interpolation=cv2.INTER_AREA)

                max_x_offset = background.shape[1] - resized_foreground.shape[1]
                max_y_offset = background.shape[0] - resized_foreground.shape[0]
                x_offset = random.randint(0, max_x_offset)
                y_offset = random.randint(0, max_y_offset)

                background = self.paste_foreground(resized_foreground, background, x_offset, y_offset)

                bbox = (x_offset, y_offset, x_offset +
                        resized_foreground.shape[1], y_offset + resized_foreground.shape[0])
                bboxes.append(bbox)

        output_filename = "blended_output.png"
        cv2.imwrite(f"{self.output_path}/{output_filename}", background)

        self.create_voc_annotation(output_filename, background.shape, bboxes)

        show_bbox_on_image = True

        if show_bbox_on_image:
            for bbox in bboxes:
                cv2.rectangle(background, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.imwrite(f"{self.output_path}/{output_filename}", background)

foreground_dir = r"D:\Re_ID\dataset\Augment\blend\object"
background_path = r"D:\Re_ID\dataset\Augment\blend\background\vlcsnap-2023-03-23-12h13m47s841.png"
output_path = r"D:\Re_ID\dataset\Augment\blend\output"

image_blender = ImageBlender(foreground_dir, background_path, output_path, repeat_times=5, class_name='person')
image_blender.blend_images()

