import numpy as np
import os
import cv2
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    # 加载 YOLOv12n 模型
    # model_path = os.path.join(os.path.abspath(os.path.join('models', 'yolov12n.pt')))
    model_path = 'yolov12n.pt'
    print("模型路径:", model_path)
    model = YOLO(model_path)
    
    # 训练我们自己的 YOLOv12n 模型
    model.train(data=os.path.join('datasets', 'pipe', 'data.yaml'), epochs=500, imgsz=640, batch=16, workers=4, device='0')

    # 测试模型效果
    model.val(data=os.path.join('datasets', 'pipe', 'data.yaml'), conf=0.4, iou=0.65, save_json=True, save_conf=True, save_hybrid=True)

    # 直接用图片的形式输入 YOLOv12, 可以是 BGR 的也可以是 RGB 的
    image_handlers = os.listdir(os.path.join('datasets', 'pipe', 'test', 'images'))
    img = cv2.imread(os.path.join('datasets', 'pipe', 'test', 'images', random.choice(image_handlers)))
    results = model(img)
    pf = pd.DataFrame({
        "cls": [model.names[cls] for cls in results[0].boxes.cls.cpu().numpy()],
        "xywh": ["{} {} {} {}".format(*xywh) for xywh in results[0].boxes.xywhn.cpu().numpy()],
        "conf": results[0].boxes.conf.cpu().numpy()
    })
    print(pf, '\n')
    
    # 绘制后的图片
    draw = results[0].plot()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(draw)
    plt.axis('off')
    plt.show()
    
    # 是否保存模型
    ctrl = input("是否保存模型？(y/n): ")
    if ctrl.lower() == 'y':
        model.save(os.path.join('models', 'nanoka_model.pt'))
        print("模型已保存.")
    else:
        print("模型未保存.")