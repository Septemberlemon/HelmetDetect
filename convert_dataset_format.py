import os
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET


# 设置路径
voc_images_dir = "C:\\Users\\septemberlemon\\.cache\\ultralytics\\datasets\\VOCdevkit\\VOC2012\\JPEGImages"
voc_annotations_dir = "C:\\Users\\septemberlemon\\.cache\\ultralytics\\datasets\\VOCdevkit\\VOC2012\\Annotations"
output_dir = "C:\\Users\\septemberlemon\\.cache\\ultralytics\\datasets\\helmet-detect"  # 输出 YOLO 数据集的根目录


# 创建 YOLO 数据集文件夹结构
def create_yolo_folders(base_dir):
    for subset in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_dir, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, subset, "labels"), exist_ok=True)


# 转换 VOC 标注文件为 YOLO 格式
def convert_voc_to_yolo(xml_file, classes, output_txt):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    img_width = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)

    with open(output_txt, "w") as f:
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in classes:
                continue
            class_id = classes.index(class_name)

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # 计算 YOLO 格式 (x_center, y_center, width, height)
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


# 主函数
def create_yolo_dataset(voc_images_dir, voc_annotations_dir, output_dir, classes, test_size=0.1, val_size=0.1):
    # 获取所有图片文件名
    images = [f for f in os.listdir(voc_images_dir) if f.endswith((".jpg", "png"))]
    images.sort()  # 确保文件名排序一致

    # 拆分训练集、验证集、测试集
    train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
    train_images, val_images = train_test_split(train_images, test_size=val_size, random_state=42)

    subsets = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }

    create_yolo_folders(output_dir)

    # 复制图片并生成标注文件
    for subset, image_list in subsets.items():
        for image in image_list:
            # 图片路径
            src_img_path = os.path.join(voc_images_dir, image)
            dst_img_path = os.path.join(output_dir, subset, "images", image)

            # 对应的 VOC XML 文件路径
            annotation_file = image.replace(".jpg", ".xml")
            src_annotation_path = os.path.join(voc_annotations_dir, annotation_file)

            # YOLO 标注文件路径
            dst_label_path = os.path.join(output_dir, subset, "labels", annotation_file.replace(".xml", ".txt"))

            # 复制图片
            shutil.copy(src_img_path, dst_img_path)

            # 转换标注
            if os.path.exists(src_annotation_path):
                convert_voc_to_yolo(src_annotation_path, classes, dst_label_path)
            else:
                print(f"Warning: Missing annotation for {image}")


# VOC 数据集的类别列表
voc_classes = ["head", "helmet", "person"]  # 替换为你自己的类别列表

# 执行
create_yolo_dataset(voc_images_dir, voc_annotations_dir, output_dir, voc_classes)
