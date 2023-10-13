import os
import cv2
import xml.etree.ElementTree as ET


class PedestrianCropper:
    def __init__(self, dataset_type, image_dir, annotation_dir, output_dir, target_label):
        self.dataset_type = dataset_type
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.output_dir = output_dir
        self.target_label = target_label

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def crop_pedestrians_voc(self, image_path, xml_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image file not found: {image_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        person_idx = 0
        for obj in root.findall("object"):
            label = obj.find("name").text

            if label == self.target_label:
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)

                cropped = image[ymin:ymax, xmin:xmax]
                output_file_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_person_{person_idx}{os.path.splitext(image_path)[1]}"
                output_path = os.path.join(self.output_dir, output_file_name)
                print(f"Saving cropped person to: {output_path}")
                cv2.imwrite(output_path, cropped)
                person_idx += 1

    def crop_pedestrians_yolo(self, image_path, txt_path):
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        with open(txt_path, "r") as f:
            person_idx = 0
            for line in f:
                data = line.strip().split()
                label = int(data[0])

                if label == self.target_label:
                    x_center = float(data[1]) * width
                    y_center = float(data[2]) * height
                    box_width = float(data[3]) * width
                    box_height = float(data[4]) * height

                    xmin = int(x_center - box_width / 2)
                    ymin = int(y_center - box_height / 2)
                    xmax = int(x_center + box_width / 2)
                    ymax = int(y_center + box_height / 2)

                    cropped = image[ymin:ymax, xmin:xmax]
                    output_file_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_person_{person_idx}{os.path.splitext(image_path)[1]}"
                    output_path = os.path.join(
                        self.output_dir, output_file_name)
                    print(f"Saving cropped person to: {output_path}")
                    cv2.imwrite(output_path, cropped)
                    person_idx += 1

    def process_images(self):
        image_exts = [".jpg", ".jpeg", ".png"]
        for file_name in os.listdir(self.image_dir):
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in image_exts:
                print(f"Processing image: {file_name}")
                image_path = os.path.join(self.image_dir, file_name)

                if self.dataset_type == "voc":
                    xml_name = os.path.splitext(file_name)[0] + ".xml"
                    xml_path = os.path.join(self.annotation_dir, xml_name)
                    self.crop_pedestrians_voc(image_path, xml_path)

                elif self.dataset_type == "yolo":
                    txt_name = os.path.splitext(file_name)[0] + ".txt"
                    txt_path = os.path.join(self.annotation_dir, txt_name)
                    self.crop_pedestrians_yolo(image_path, txt_path)


dataset_type = "voc"
image_dir = "/home/dataset/pedestrian_voc/Images"
annotation_dir = "/home/dataset/pedestrian_voc/Annotations"
output_dir = "/home/dataset/cropped_pedestrians"
target_label = "person"

cropper = PedestrianCropper(dataset_type, image_dir, annotation_dir, output_dir, target_label)
cropper.process_images()