import os
import time
import argparse
import requests
import uuid
import sys
import shutil

# 程序状态码
status_code = -1
cache_name = None
files = None

def parse_opt():
    '''
    获取程序需要的所有参数, 直接配置参数
    '''
    parser = argparse.ArgumentParser()
    # 图片保存信息
    parser.add_argument('--save_path', type=str, default=os.path.abspath(os.path.join('images', 'det')), help='图片保存文件夹')
    parser.add_argument('--save_origin', action='store_true', help='是否保存摄像原图片')
    parser.add_argument('--origin_save_path', type=str, default=os.path.abspath(os.path.join('images', 'origin')), help='原始图片保存文件夹')
    # 请求信息
    parser.add_argument('--url', type=str, default='http://localhost:80', help='需要请求的服务器地址')
    parser.add_argument('--type', type=str,nargs='+', default=['pipe'], help='图片需要检测的目标')
    # 摄像头信息
    parser.add_argument('--img', type=tuple, default=(1024, 768), help='拍摄照片的宽高数据')
    parser.add_argument('--speed', type=int, default=10000, help='拍摄照片的保存速度（每秒多少张）')
    # 图像处理信息
    parser.add_argument('--thres_arg', type=int, default=0.4, help='图片需要超过的灰度阈值, 用于二值化处理')
    parser.add_argument('--threshold', type=float, default=0.3, help='图片需要超过的明亮比例, 图片过于灰暗时会被丢弃')
    # 其他信息
    parser.add_argument('--show_vid', action='store_true', help='是否显示视频流')
    parser.add_argument('--debug', action='store_true', help='是否开启调试模式')
    
    opt = parser.parse_args()
    
    # 如果只给了一个宽或者高, 那么我们扩张成正方形
    opt.img *= 2 if len(opt.img) == 1 else 1
    return opt


if __name__ == "__main__":
    '''
    主函数入口, 首先获取参数, 然后再进行判断
    '''
    # 解析命令行参数
    opt = parse_opt()
    
    # 创建保存图片的文件夹, 如果不保存那就将 opt.origin_save_path 当做缓冲目录
    for save_path in [opt.save_path, opt.origin_save_path]:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
    # 初始化摄像头
    camera = None
    if opt.debug:
        import cv2
        
        # 使用 cv2 进行摄像头拍照
        camera = cv2.VideoCapture(0)
        # 设置摄像头分辨率
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, opt.img[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.img[1])
    else:
        from picamera2 import Picamera2
        
        camera = Picamera2()
        # 配置摄像头
        config = camera.create_still_configuration(main={"size": opt.img})
        camera.configure(config)
        camera.start()
    

    # 设置拍照间隔时间（秒）
    interval = 1 / opt.speed   

    try:
    # if 1:
        # 我们首先配置识别物体类型
        json_data = {
            'type': ','.join(opt.type),     # 如果 opt.type 是一个列表，确保将其转换为字符串
            'thres_arg': opt.thres_arg      # 灰度阈值
        }
        response = requests.post(opt.url + '/config', data=json_data)
        
        # 检查响应状态码
        if response.status_code != 200:
            print(f"配置请求失败, 状态码: {response.status_code}, 使用默认配置")
        
        while True:
            t1 = time.time()
            # 获取当前时间，并格式化为字符串，用于文件名, 最后添加 uuid 防止吞照片
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            cache_name = os.path.join(opt.origin_save_path, f"photo_{timestamp}_{uuid.uuid4()}.jpg")

            if opt.debug:
                # 使用 cv2 进行拍照
                ret, image = camera.read()
                if not ret:
                    raise RuntimeError("cv2 无法读取摄像头图像")
                cv2.imwrite(cache_name, image)
            else:
                # 拍照并保存
                camera.capture_file(cache_name)
            
            # 如果要展示图片 (先展示图片)
            if opt.show_vid:
                # 读取并在图片上显示视频
                image = cv2.imread(cache_name)
                
                cv2.imshow('frame', image)
                if cv2.waitKey(1) == ord('q'):
                    if not opt.save_origin:
                        os.remove(cache_name)
                    break
                
            # 发送 POST 请求上传图片
            files = {'image': open(cache_name, 'rb')}
            response = requests.post(opt.url + '/detect', files=files)
            # 关闭占用
            files['image'].close()
            
            # 检查响应状态码
            if response.status_code != 200:
                print(f"请求失败, 状态码: {response.status_code}")
                os.remove(cache_name)
                continue
            
            # 将响应内容转换为 JSON 格式
            response_json = response.json()
            save_name = os.path.join(opt.save_path, f"photo_{timestamp}_{uuid.uuid4()}.jpg")
            if response_json['status'] == 0:
                # 灰度信息
                ratio = response_json['ratio']
                print(f"图片 {cache_name} 明亮点的比例为: {ratio:.4f}", end='')
                # 如果图片过于灰暗, 那么就丢弃
                if ratio < opt.threshold:
                    print(", 图片灰暗, 丢弃")
                    os.remove(cache_name)
                    continue
                else:
                    print(", 图片明亮, 正在保存...")
                
                # 将识别后的图片保存
                shutil.copy(cache_name, save_name)
                print(f"图片 {save_name} 检测到了相关内容, 保存成功", end=', ')
            else:
                print(f"图片 {save_name} 未检测到相关内容, 取消保存", end=', ')
            
            # 如果没有保存, 那我们就删除缓存图片
            if not opt.save_origin:
                os.remove(cache_name)
            
            t2 = time.time()
            print("程序用时: {:.4f}秒".format(t2 - t1))
            # 等待时间
            time.sleep(interval)

    # 异常处理和报错
    except KeyboardInterrupt:
        # 捕获中断异常，例如用户按下Ctrl+C
        print("接收到 KeyboardInterrupt 中断, 程序已被用户中断")
        status_code = 0
    except RuntimeError as e:
        # 捕获运行时异常
        print(f"接收到 RuntimeError 异常, 具体内容为: {e}")
        status_code = 1
    except Exception as e:
        # 捕获其他异常
        print(f"接收到其他 Exception 未知错误, 具体内容为: {e}")
        status_code = 2
    finally:
        # 最终清理工作     
        if os.path.exists(cache_name):
            try:
                os.remove(cache_name)
            except PermissionError as e:
                # 捕获权限错误
                print(f"删除缓存图片时遇到 PermissionError, 具体内容为: {e}, 正在尝试解决...")
                files['image'].close()
                os.remove(cache_name)
            print("删除缓存图片: {}".format(cache_name))
        
        # 关闭摄像头
        if opt.debug:
            camera.release()
        else:
            camera.close()
        print("摄像头已关闭, 程序退出 (退出码: {})".format(status_code))
        sys.exit(status_code)
    
    # camera.release()
