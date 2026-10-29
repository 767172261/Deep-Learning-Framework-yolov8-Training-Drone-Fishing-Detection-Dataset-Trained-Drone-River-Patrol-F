深度学习框架yolov8训练无人机钓鱼检测数据集   训练出的无人机河流巡检钓鱼人检测数据集 建立基于无人机基于深度学习的无人机钓鱼人检测系统
@[toc]

无人机河道河流巡检疑似钓鱼人目标检测数据集

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cadbdf627dd6498b9558b1023bd4da15.png)

### 🎣 无人机河道巡检疑似钓鱼人目标检测数据集统计表

#### 数据集概述

| 项目               | 详情                           |
|--------------------|--------------------------------|
| **无人机型号**      | DJI MAVIC 3                    |
| **数据类型**        | 原始图片 + XML格式标签         |
| **总内存大小**      | 5.34 GB                        |
| **图片总数**        | 1,308 张                       |
| **图片分辨率**      | 4K                             |
| **采集高度**        | 15～40 米                      |
| **采集角度**        | 30°～90°                       |
| **拍摄时间**        | 2025-04-20                     |
| **类别**            | 钓鱼人                         |

#### 类别统计
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0fbc8a8a534b42529aeb32654566dd7c.png)

| 序号 | 类别名称 | 英文类别名 | 备注 |
|------|----------|------------|------|
| 1    | 钓鱼人   | Fishing Person | 数据集中唯一的目标检测类别 |

---

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/106eb720f2d441cdaa47f97350ebcb1f.jpeg)
1

使用无人机（如DJI MAVIC 3）采集的数据集进行钓鱼人目标检测，代码示例，以下步骤：数据准备、格式转换、数据划分、环境搭建、数据配置、模型训练、超参数配置、模型推理以及性能评估。代码示例，仅供参考。

### 数据准备

首先，确保tx 同学你的数据集已经准备好，并且包含原始图片和XML格式的标签
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/15a41af2ab7f4d73964e070da12b8417.png)

### 格式转换

通常情况下，深度学习框架要求标签以特定格式存在，比如YOLO使用的`.txt`文件或COCO格式。使用YOLOv8作为模型，我们同学需要将XML标签转换为YOLO格式。

使用Python脚本完成这个转换：

```python
import xml.etree.ElementTree as ET
import os

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(xml_files, output_path):
    for xml_file in xml_files:
        in_file = open(xml_file)
        tree=ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult)==1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            out_file = open(os.path.join(output_path, os.path.basename(xml_file).replace('.xml','.txt')), 'a')
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            out_file.close()

# 假设classes=['钓鱼人']
classes = ['钓鱼人']
xml_files = ['path/to/xml_files']  # 替换为你的XML文件路径列表
output_path = 'path/to/output/labels'  # 输出路径
convert_annotation(xml_files, output_path)
```

### 数据划分

接下来是将数据划分为训练集、验证集和测试集。可以使用如下脚本来随机划分数据集：

```python
import os
import random
import shutil

dataset_dir = 'path/to/dataset'
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
random.shuffle(image_files)

total = len(image_files)
train_size = int(total * train_ratio)
val_size = int(total * val_ratio)

train_set = image_files[:train_size]
val_set = image_files[train_size:train_size+val_size]
test_set = image_files[train_size+val_size:]

os.makedirs('images/train', exist_ok=True)
os.makedirs('images/val', exist_ok=True)
os.makedirs('images/test', exist_ok=True)

for file_name in train_set:
    shutil.move(os.path.join(dataset_dir, file_name), os.path.join('images/train', file_name))
    shutil.move(os.path.join(dataset_dir, file_name.replace('.jpg', '.txt')), os.path.join('labels/train', file_name.replace('.jpg', '.txt')))

for file_name in val_set:
    shutil.move(os.path.join(dataset_dir, file_name), os.path.join('images/val', file_name))
    shutil.move(os.path.join(dataset_dir, file_name.replace('.jpg', '.txt')), os.path.join('labels/val', file_name.replace('.jpg', '.txt')))

for file_name in test_set:
    shutil.move(os.path.join(dataset_dir, file_name), os.path.join('images/test', file_name))
    shutil.move(os.path.join(dataset_dir, file_name.replace('.jpg', '.txt')), os.path.join('labels/test', file_name.replace('.jpg', '.txt')))
```

### 环境搭建与模型训练

请参照前面的回答中关于环境配置的部分安装必要的库。然后，下载并设置YOLOv8或其他适合的目标检测模型。

对于模型训练，你可以使用如下命令：

```bash
yolo task=detect mode=train model=yolov8n.pt data=path/to/data.yaml epochs=100 imgsz=640
```

其中`data.yaml`需要定义训练集、验证集和测试集的路径及类别信息。

### 模型推理与批量推理

使用训练好的模型进行推理，可以参考之前提供的代码片段。对于批量推理，只需遍历所有测试图像并应用相同的推理逻辑。

### 性能评估

在模型训练完成后，可以使用以下命令来评估模型性能：

```bash
yolo task=detect mode=val model=path/to/best.pt data=path/to/data.yaml
```

为了根据训练好的权重建立一个深度学习无人机钓鱼检测系统，你需要完成一系列步骤，从准备环境、加载模型到执行推理及用户界面开发。以下是一个详细的指南：

### 1. 环境配置

首先确保你的环境中安装了必要的库和依赖项。这里以YOLOv8为例进行说明，你可以使用`pip`来安装这些依赖：

```bash
pip install ultralytics opencv-python-headless
```

### 2. 加载训练好的模型

假设你已经有了针对钓鱼人检测训练好的YOLOv8模型权重文件（例如`best.pt`），你可以通过下面的Python代码加载模型：

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('path/to/best.pt')  # 替换为你的模型路径
```

### 3. 执行推理

接下来是实现对图像或视频流进行推理的功能。以下是针对单张图片和视频流的推理示例代码。

#### 对单张图片进行推理

```python
import cv2

def detect_fishing_person(image_path):
    img = cv2.imread(image_path)
    results = model(img)
    
    for result in results:
        annotated_frame = result.plot()  # 自动绘制边界框和标签
        cv2.imshow("Fishing Person Detection", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
```

#### 对视频流进行推理

对于无人机拍摄的实时视频流或录像，可以使用如下代码进行处理：

```python
def process_video(video_path=0):  # 使用0表示使用摄像头，或者指定视频文件路径
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 进行推理
        results = model(frame)

        # 绘制结果
        for result in results:
            annotated_frame = result.plot()
            cv2.imshow("Fishing Person Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 Q 键退出
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

### 4. 开发用户界面

为了让操作员能够轻松地监控无人机的状态以及钓鱼人检测的结果，我们可以创建一个简单的图形用户界面（GUI）。这里我们使用`tkinter`作为示例框架。

```python
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading

class FishingPersonDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("无人机钓鱼人检测系统")
        
        # 加载模型
        self.model = YOLO('path/to/best.pt')  # 替换为你的模型路径
        
        # GUI组件
        self.select_image_button = tk.Button(root, text="选择图片", command=self.detect_image)
        self.select_image_button.pack(pady=10)
        
        self.select_video_button = tk.Button(root, text="选择视频", command=self.detect_video)
        self.select_video_button.pack(pady=10)
        
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)
    
    def detect_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self._run_detection(file_path, is_video=False)
    
    def detect_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
        if file_path:
            threading.Thread(target=self._run_detection, args=(file_path, True)).start()
    
    def _run_detection(self, file_path, is_video):
        if is_video:
            cap = cv2.VideoCapture(file_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = self.model(frame)
                annotated_frame = results[0].plot()
                
                # 转换颜色通道并显示
                img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.image_label.imgtk = imgtk
                self.image_label.configure(image=imgtk)
                self.root.update_idletasks()
                self.root.update()
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            results = self.model(file_path)
            annotated_image = results[0].plot()
            
            # 转换颜色通道并调整大小以适应窗口
            img = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((640, 480), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.image_label.imgtk = imgtk
            self.image_label.configure(image=imgtk)
            messagebox.showinfo("完成", "图像处理完成")

if __name__ == '__main__':
    root = tk.Tk()
    app = FishingPersonDetectionApp(root)
    root.mainloop()
```

### 总结

，从加载训练好的模型权重到数据推理以及用户界面的开发，帮助你构建一个基于深度学习的无人机钓鱼人检测系统。请根据实际情况替换所有路径为你自己的实际路径，并根据需要调整参数和优化模型。请注意，具体实现细节可能会有所变化，请参考最新的官方文档或资源进行调整。

以上文字及代码仅供参考。学习，


