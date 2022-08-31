from __future__ import print_function
# 我们将使用高效且线程化的VideoStream 使我们可以同时访问内置/ USB网络摄像头和Raspberry Pi摄像头模块。
# VideoStream 类在imutils Python包内部实现。您可以阅读有关VideoStream的更多信息【https://www.pyimagesearch.com/2016/01/04/unifying-picamera-and-cv2-videocapture-into-a-single-class-with-opencv/】   类，它如何访问多个摄像机输入，并在本教程中以线程方式有效读取帧。
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# 构建命令行参数
# --output  输出视频存储的磁盘路径
# --picamera  指定是否要使用Raspberry Pi摄像头模块而不是内置/ USB摄像头。提供> 0的值以访问Pi摄像机模块
# --fps 控制输出视频所需的FPS
# --codec 我们提供FourCC或四个字符的代码，视频编解码器的标识符，压缩格式以及视频文件中的颜色/像素格式。 不同的组合很可能奏效，也可能不奏效；
# MJPG的组合 和.avi 开箱即用，既可以在OSX机器上运行，也可以在Raspberry Pi上工作，因此，如果在将视频写入文件时遇到问题，请务必先尝试这些组合！
# 注意 codec: MJPG output: example.avi； codec: MP4V output: baby.avi

# 初始化视频流，让相机📷传感器 预热2s
print("[INFO] warming up camera...")
vs = VideoStream(usePiCamera=-1> 0).start()
time.sleep(2.0)

# 初始化 FourCC, 视频writer，帧窗口的宽度，高度，0的数组
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = None
(h, w) = (None, None)
zeros = None

# 遍历视频流
while True:
    # 获取视频流的一帧 并且resize窗口宽为300
    frame = vs.read()
    frame = imutils.resize(frame, width=300)

    # 检查writer是否为None
    if writer is None:
        # 获取帧的空间尺寸（宽度和高度），实例化视频流videoWriter
        (h, w) = frame.shape[:2]
        writer = cv2.VideoWriter("F:\\", fourcc, 20,
                                 (w * 2, h * 2), True)
        zeros = np.zeros((h, w), dtype="uint8")

    # 我们将frame 分离为红色，绿色和蓝色通道， 然后我们使用Numpy 零数组分别构造每个通道的表示形式
    (B, G, R) = cv2.split(frame)
    R = cv2.merge([zeros, zeros, R])
    G = cv2.merge([zeros, G, zeros])
    B = cv2.merge([B, zeros, zeros])
    # 构建输出帧  原图在左上角 红色通道右上角 绿色通道右下角 蓝色通道左下角
    output = np.zeros((h * 2, w * 2, 3), dtype="uint8")
    output[0:h, 0:w] = frame
    output[0:h, w:w * 2] = R
    output[h:h * 2, w:w * 2] = G
    output[h:h * 2, 0:w] = B
    # 将帧写入视频
    writer.write(output)

    # 展示帧
    cv2.imshow("Frame", frame)
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF
    # 按下q键 将结束播放
    if key == ord("q"):
        break

# 清理，释放资源
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
writer.release()
