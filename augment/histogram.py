import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class HistogramEqualizer:
    def __init__(self):
        pass
    
    def equalize(self, image, method="HE"):
        if method == "HE":
            return self.traditional_equalization(image)
        elif method == "CLAHE":
            return self.clahe_equalization(image)
        else:
            raise ValueError("Invalid method name. Choose from ['HE', 'CLAHE']")

    def traditional_equalization(self, image):
        equalized_image = np.zeros_like(image)
        for i in range(3):
            equalized_image[:, :, i] = cv2.equalizeHist(image[:, :, i])
        return equalized_image

    def clahe_equalization(self, image):
        equalized_image = np.zeros_like(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for i in range(3):
            equalized_image[:, :, i] = clahe.apply(image[:, :, i])
        return equalized_image

    def plot_histogram(self, image, title="Histogram", output_path=None):
        channels = ["Blue", "Green", "Red"]
        plt.figure()
        plt.title(title)
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")

        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=channels[i])
            plt.xlim([0, 256])

        if output_path:
            plt.savefig(output_path)
        plt.show()

if __name__ == "__main__":

    input_image_path = "/home/dataset/pedestrian_voc/Images/1.jpeg"
    output_image_path = "/home/dataset/HE"

    os.makedirs(output_image_path, exist_ok=True)

    # 读取输入图像
    input_image = cv2.imread(input_image_path)

    # 创建HistogramEqualizer对象
    equalizer = HistogramEqualizer()

    # 选择要应用的方法（'HE', 'CLAHE')
    method = "HE"

    # 应用直方图均衡化
    output_image = equalizer.equalize(input_image, method=method)

    # 获取输入图像的扩展名
    input_ext = os.path.splitext(input_image_path)[1]

    # 在输出路径中包含文件名和扩展名
    output_image_path_ = os.path.join(output_image_path, f"output{input_ext}")

    # 保存输出图像
    cv2.imwrite(output_image_path_, output_image)

    # 绘制并保存直方图
    histogram_filename = f"{method}_histogram.png"
    equalizer.plot_histogram(output_image, title=method, output_path=os.path.join(output_image_path, histogram_filename))
