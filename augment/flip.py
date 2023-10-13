import random
import cv2
import xml.etree.ElementTree as ET


class ImageFlipper:

    def __init__(self, percentage=1, flip_type='random', annotation_type='voc'):
        self.flip_type = flip_type
        self.annotation_type = annotation_type
        self.percentage = percentage

    def flip_horizontal(self, image):
        return cv2.flip(image, 1)

    def flip_vertical(self, image):
        return cv2.flip(image, 0)

    def flip_both(self, image):
        return cv2.flip(image, -1)

    def flip_voc_annotation(self, tree, flip_type, width, height):
        root = tree.getroot()
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            if flip_type == 'horizontal':
                bbox.find('xmin').text = str(width - xmax)
                bbox.find('xmax').text = str(width - xmin)
            elif flip_type == 'vertical':
                bbox.find('ymin').text = str(height - ymax)
                bbox.find('ymax').text = str(height - ymin)
            elif flip_type == 'both':
                bbox.find('xmin').text = str(width - xmax)
                bbox.find('ymin').text = str(height - ymax)
                bbox.find('xmax').text = str(width - xmin)
                bbox.find('ymax').text = str(height - ymin)

        return tree

    def flip_yolo_annotation(self, lines, flip_type):
        new_lines = []
        for line in lines:
            line_parts = line.strip().split(' ')
            class_id = line_parts[0]
            x_center = float(line_parts[1])
            y_center = float(line_parts[2])
            w = float(line_parts[3])
            h = float(line_parts[4])

            if flip_type == 'horizontal':
                x_center = 1 - x_center
            elif flip_type == 'vertical':
                y_center = 1 - y_center
            elif flip_type == 'both':
                x_center = 1 - x_center
                y_center = 1 - y_center

            new_line = f"{class_id} {x_center} {y_center} {w} {h}\n"
            new_lines.append(new_line)

        return new_lines

    def augment(self, image, input_annotation):
        height, width, _ = image.shape

        if input_annotation is not None:
            if random.random() < self.percentage:
                if self.flip_type == 'random':
                    flip_choices = ['horizontal', 'vertical', 'both']
                    flip_type = random.choice(flip_choices)
                else:
                    flip_type = self.flip_type

                if flip_type == 'horizontal':
                    flipped_image = self.flip_horizontal(image)
                elif flip_type == 'vertical':
                    flipped_image = self.flip_vertical(image)
                elif flip_type == 'both':
                    flipped_image = self.flip_both(image)

                if isinstance(input_annotation, ET.ElementTree):
                    if self.annotation_type == 'voc' and isinstance(input_annotation, ET.ElementTree):
                        flipped_annotation = self.flip_voc_annotation(
                            input_annotation, flip_type, width, height)
                        
                        return flipped_image, flipped_annotation
                    else:
                        raise ValueError(
                            "Invalid annotation_type or input_annotation format")
                
                elif isinstance(input_annotation, list):
                    if self.annotation_type == 'yolo' and isinstance(input_annotation, list):
                        flipped_annotation = self.flip_yolo_annotation(
                            input_annotation, flip_type)
                        
                        return flipped_image, flipped_annotation
                    else:
                        raise ValueError(
                            "Invalid annotation_type. Supported types for XML annotations are 'voc'.")
            
                else:
                    raise ValueError(
                        "Invalid annotation format. Supported formats are lists of YOLO annotation strings and ElementTree objects for VOC annotations.")

        else:
            return image


def flip_image(image, flip_type='random', percentage=1):
    
    def flip_horizontal(image):
        return cv2.flip(image, 1)

    def flip_vertical(image):
        return cv2.flip(image, 0)

    def flip_both(image):
        return cv2.flip(image, -1)

    if random.random() < percentage:
        if flip_type == 'random':
            flip_choices = ['horizontal', 'vertical', 'both']
            flip_type = random.choice(flip_choices)
        
        if flip_type == 'horizontal':
            flipped_image = flip_horizontal(image)
        elif flip_type == 'vertical':
            flipped_image = flip_vertical(image)
        elif flip_type == 'both':
            flipped_image = flip_both(image)
        
        return flipped_image
    else:
        return image