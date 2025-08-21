"""
@Project ：MVANet 
@File    ：trans_mask.py
@IDE     ：PyCharm 
@Author  ：chengxuLiu
@Date    ：2025/6/4 20:12 
"""
from PIL import Image
import numpy as np
import os


def convert_to_binary_black_white(image_path, output_path, black_threshold=0):
    """
    直接以灰度模式读取图像，将所有非黑色像素转换为白色，
    黑色像素保持不变，然后保存为PNG图像。

    参数:
        image_path (str): 输入图像的路径。
        output_path (str): 处理后图像的保存路径。
        black_threshold (int): 定义“黑色”的阈值。灰度值小于等于此阈值的像素将被视为黑色。
                               默认为0，表示只有纯黑 (灰度值为0) 才被视为黑色。
                               可以适当调高此值 (例如 5 或 10) 以将非常暗的灰色也视为背景黑。
    返回:
        bool: 如果成功处理并保存图像则返回 True，否则返回 False。
    """
    try:
        # 1. 直接以灰度模式 ('L' mode) 读取图像
        # 'L'模式表示 luminance (亮度)，即灰度图像
        img_gray = Image.open(image_path).convert('L')

        # 2. 将灰度图像转换为Numpy数组以便进行像素操作
        gray_data = np.array(img_gray)

        # 3. 创建一个新的空白图像数据 (初始为全黑)
        #   或者可以直接在原数据上修改，但创建副本更安全
        #   这里我们直接修改，因为目标是二值化
        processed_data = np.zeros_like(gray_data, dtype=np.uint8)  # 确保背景是黑色 (0)

        # 4. 识别非黑色像素并将其设置为白色 (255)
        #    灰度值大于 black_threshold 的像素被认为是前景 (非黑色)
        foreground_pixels = gray_data > black_threshold
        processed_data[foreground_pixels] = 255
        # foreground_pixels = np.where(gray_data == 127, 0, 255).astype(np.uint8)
        # processed_data[foreground_pixels] = 255
        # 5. 将处理后的Numpy数组转换回Pillow图像对象
        #    因为我们处理的是灰度数据，所以模式也是 'L'
        output_img = Image.fromarray(foreground_pixels)

        # 6. 保存为PNG图像
        output_img.save(output_path)
        print(f"图像已成功处理并保存到: {output_path}")
        return True

    except FileNotFoundError:
        print(f"错误: 图像文件未找到于路径 '{image_path}'")
        return False
    except Exception as e:
        print(f"处理或保存图像时发生错误: {e}")
        return False


def binarize_image_array(gray_array: np.ndarray) -> np.ndarray:
    """
    检测一个灰度图NumPy数组，将值为127的像素点变为黑色(0)，
    其余像素点变为白色(255)。

    参数:
        gray_array (np.ndarray): 输入的灰度图NumPy数组 (应为2维)。

    返回:
        np.ndarray: 处理后的二值化图像NumPy数组。
    """
    if gray_array.ndim != 2:
        raise ValueError("输入数组必须是二维的灰度图数组。")

    # 使用 np.where 高效地进行条件赋值
    # 条件: 数组中的元素值是否等于 127
    # 如果等于 127，则新值为 0 (黑色)
    # 如果不等于 127，则新值为 255 (白色)
    binary_array = np.where(gray_array == 127, 0, 255).astype(np.uint8)

    return binary_array


if __name__ == '__main__':
    image_file_path = '/22liuchengxu/dataset/foodpixcompleted/annotations/training/'
    new_image_file_path = '/22liuchengxu/dataset/foodpixcompleted/annotations/training_l/'
    # image_file_path = '/22liuchengxu/dataset/FoodSeg103/Images/ann_dir/training/'
    # new_image_file_path = '/22liuchengxu/dataset/FoodSeg103/Images/ann_dir/training_l/'
    # image_file_path = '/22liuchengxu/MVANet-main/MVANet-main/model_test/mvegn2_seg103_3/mvegn2_seg103_loss_2.7673.pth/seg103/'
    # new_image_file_path = '/22liuchengxu/MVANet-main/MVANet-main/model_test/mvegn2_seg103_3/mvegn2_seg103_loss_2.7673.pth/seg103_l/'
    image_filenames = os.listdir(image_file_path)
    for image in image_filenames:
        convert_to_binary_black_white(image_file_path + image, new_image_file_path + image)