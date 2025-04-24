from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import cv2
import pandas as pd
from ultralytics import YOLO

# 我们的 HTTP 服务器
app = Flask(__name__)
CORS(app)

model_path = os.path.join(os.path.abspath(os.path.join('models', 'nanoka_model.pt')))
lst, thres_arg = ['pipe'], 0.4

# 加载 YOLOv12n 模型
model = YOLO(model_path)

def image_threshold(image):
    '''
    对图片进行灰度化处理, 然后对图片进行二值化处理
    '''
    global thres_arg
    # 将图片转换为灰度图并对图片进行二值化处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thres_arg, 255, cv2.THRESH_BINARY)
    
    white_pixels = np.sum(binary == 255)
    total_pixels = binary.size
    
    # 计算白色像素点占总像素点的比例
    ratio = white_pixels / total_pixels
    return ratio

def detect_img(img):
    '''
    直接用图片的形式输入 YOLOv12, 可以是 BGR 的也可以是 RGB 的
    '''
    global model
    results = model.predict(img, device='0')
    
    pf = pd.DataFrame({
        "cls": [model.names[cls] for cls in results[0].boxes.cls.cpu().numpy()],
        "xywh": ["{} {} {} {}".format(*xywh) for xywh in results[0].boxes.xywhn.cpu().numpy()],
        "conf": results[0].boxes.conf.cpu().numpy()
    })
    # print(pf, '\n')
    
    # 绘制后的图片
    draw = results[0].plot()    
    return pf, draw

# 配置检测类型
@app.route('/config', methods=['POST'])
def config():
    '''
    配置检测类型
    '''
    global lst, thres_arg
    # 获取 'type' 字段并拆分为列表
    type_list = request.form.get('type', '').split(',')
    lst = type_list
    # 获取 'thres_arg' 字段
    thres_arg = float(request.form.get('thres_arg', thres_arg))
    print("类型修改为:{}, 灰度阈值修改为:{}".format(lst, thres_arg))
    
    return jsonify({"status": 0, "message": "Configuration successful"}), 200

# 我们的检测函数
@app.route('/detect', methods=['POST'])
def detect():
    '''
    开始检测我们的图片是否存在我们需要的类型
    '''
    global lst, thres_arg
    # 读取图片并开始计算
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    pf, _ = detect_img(img)
    
    # 将检测到的图片进行灰度化处理
    ratio = image_threshold(img)

    # 检测缓存
    pf = pf[pf['cls'].isin(lst)]
    print(pf, '\n')
    
    # 如果没有检测到我们需要的类型, 那就值返回一个信息即可
    if pf.empty:
        return jsonify({"status": 1, "ratio": ratio, "message": "No target detected"}), 200
    # 将我们的所有检测信息返回
    else:
        return jsonify({"status": 0, "ratio": ratio, "message": "Target detected", "data": pf.to_dict(orient='records')}), 200
    
if __name__ == '__main__':
    app.run('0.0.0.0', port=80, debug=False)