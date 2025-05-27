# YOLOv12 后端服务案例
以 YOLO 为后端服务的服务器，本案例为拍照服务器

### 安装方法
首先我们安装好 CUDA 版本的 PyTorch 和对应版本的 TorchVision

```
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121
pip install -r requirements.txt
```

之后我们大概率会遇到 flash_attr 安装不上的情况, 我们可以下载之后本地安装

```
wget https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu124torch2.6.0cxx11abiFALSE-cp312-cp312-win_amd64.whl
```

其实吧我看着 torch 版本是 2.6.0 但是 cu121 而 flash_attr 是 cu124 但是居然没报错我也就没管了

### 训练模型

先下载你要使用的模型文件

```
wget https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12s.pt
```

```python
# 修改这一部分到 train.py
model_path = os.path.abspath('yolov12s.pt')
```

去 RoboFlow 上训练自己的模型, 然后导出成 YOLOv12 的格式

```python
# 把这一部分修改为你的文件夹名字
FOLDER_PATH = "oil"
```

之后就可以直接运行 train.py 了

```
python train.py
```

### 使用方法

前后端分离的, 主要考虑到树莓派 4B 运行 YOLO 不太行, 所以计算任务交给服务器了

```python
# 运行前记得把模型文件改了
model_path = os.path.join(os.path.abspath(os.path.join('models', 'nanoka_model.pt')))
```

```
# 开启后端 HTTP 服务, 主要用于运行 YOLOv12, 降低边缘服务器压力 
python app.py

# 开启拍照程序
python capture.py --show_vid --save_origin --debug --url http://127.0.0.1:80 --type person --speed 2
```

上面的 show_vid 是实时显示图像, save_origin 是保存原图片, url 是服务器地址, type 是要检测的类型, speed 是每秒拍摄多少张（主要是用来限速的, 最大速度不会超过这个）
