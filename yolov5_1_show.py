import os
import numpy as np
import mindspore as ms
from src.yolo import YOLOV5s
import cv2
import matplotlib.pyplot as plt


def nms(pred, conf_thres, iou_thres):
    # 置信度抑制，小于置信度阈值则删除
    conf = pred[..., 4] > conf_thres
    box = pred[conf == True]
    # 类别获取
    cls_conf = box[..., 5:]
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))
    # 获取类别
    total_cls = list(set(cls))  # 删除重复项，获取出现的类别标签列表,example=[0, 17]
    output_box = []  # 最终输出的预测框
    # 不同分类候选框置信度
    for i in range(len(total_cls)):
        clss = total_cls[i]  # 当前类别标签
        # 从所有候选框中取出当前类别对应的所有候选框
        cls_box = []
        for j in range(len(cls)):
            if cls[j] == clss:
                box[j][5] = clss
                cls_box.append(box[j][:6])
        cls_box = np.array(cls_box)
        box_conf = cls_box[..., 4]  # 取出候选框置信度
        box_conf_sort = np.argsort(box_conf)  # 获取排序后索引
        max_conf_box = cls_box[box_conf_sort[len(box_conf) - 1]]
        output_box.append(max_conf_box)  # 将置信度最高的候选框输出为第一个预测框
        cls_box = np.delete(cls_box, 0, 0)  # 删除置信度最高的候选框
        while len(cls_box) > 0:
            max_conf_box = output_box[len(output_box) - 1]  # 将输出预测框列表最后一个作为当前最大置信度候选框
            del_index = []
            for j in range(len(cls_box)):
                current_box = cls_box[j]  # 当前预测框
                interArea = getInter(max_conf_box, current_box)  # 当前预测框与最大预测框交集
                iou = getIou(max_conf_box, current_box, interArea)  # 计算交并比
                if iou > iou_thres:
                    del_index.append(j)  # 根据交并比确定需要移出的索引
            cls_box = np.delete(cls_box, del_index, 0)  # 删除此轮需要移出的候选框
            if len(cls_box) > 0:
                output_box.append(cls_box[0])
                cls_box = np.delete(cls_box, 0, 0)
    return output_box


# 计算并集
def getIou(box1, box2, inter_area):
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - inter_area
    iou = inter_area / union
    return iou


# 计算交集
def getInter(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[0], box1[1], \
                                         box1[0] + box1[2], box1[1] + box1[3]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[0], box2[1], \
                                         box2[0] + box2[2], box2[1] + box2[3]
    if box1_x1 > box2_x2 or box1_x2 < box2_x1:
        return 0
    if box1_y1 > box2_y2 or box1_y2 < box2_y1:
        return 0
    x_list = [box1_x1, box1_x2, box2_x1, box2_x2]
    x_list = np.sort(x_list)
    x_inter = x_list[2] - x_list[1]
    y_list = [box1_y1, box1_y2, box2_y1, box2_y2]
    y_list = np.sort(y_list)
    y_inter = y_list[2] - y_list[1]
    inter = x_inter * y_inter
    return inter


def draw(img, pred):
    img_ = img.copy()
    if len(pred):
        for detect in pred:
            x1 = int(detect[0])
            y1 = int(detect[1])
            x2 = int(detect[0] + detect[2])
            y2 = int(detect[1] + detect[3])
            score = detect[4]
            cls = detect[5]
            # hat person
            # mask face
            labels = [ 'person'] #类别名，改成自己的，注意要是没改yolo的类别数。。就让labels数量别变，把前几个标签改成你的就行

            print(x1, y1, x2, y2, score, cls)

            img_ = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

            text = labels[int(cls)]
            cv2.putText(img, text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, )
    return img_


def load_parameters(network, filename):
    param_dict = ms.load_checkpoint(filename)
    param_dict_new = {}
    for key, values in param_dict.items():
        if key.startswith('moments.'):
            continue
        elif key.startswith('yolo_network.'):
            param_dict_new[key[13:]] = values
        else:
            param_dict_new[key] = values
    ms.load_param_into_net(network, param_dict_new)


def main(ckpt_file, img):
    orig_h, orig_w = img.shape[:2]
    ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target='CPU', device_id=0)
    network = YOLOV5s(is_training=False)
    if os.path.isfile(ckpt_file):
        load_parameters(network, ckpt_file)
    else:
        raise FileNotFoundError(f"{ckpt_file} is not a filename.")
    network.set_train(False)
    input_shape = ms.Tensor(tuple([640, 640]), ms.float32)
    img = cv2.resize(img, (640, 640), cv2.INTER_LINEAR)
    img = img[:, :, ::-1].transpose((2, 0, 1))
    img = img / 255.
    img = np.expand_dims(img, axis=0)
    image = np.concatenate((img[..., ::2, ::2], img[..., 1::2, ::2],
                            img[..., ::2, 1::2], img[..., 1::2, 1::2]), axis=1)
    image = ms.Tensor(image, dtype=ms.float32)
    output_big, output_me, output_small = network(image, input_shape)
    output_big = output_big.asnumpy()
    output_me = output_me.asnumpy()
    output_small = output_small.asnumpy()
    output_small = np.squeeze(output_small)
    output_small = np.reshape(output_small, [19200, 85])
    output_me = np.squeeze(output_me)
    output_me = np.reshape(output_me, [4800, 85])
    output_big = np.squeeze(output_big)
    output_big = np.reshape(output_big, [1200, 85])
    result = np.vstack([output_small, output_me, output_big])
    for i in range(len(result)):
        x = result[i][0] * orig_w
        y = result[i][1] * orig_h
        w = result[i][2] * orig_w
        h = result[i][3] * orig_h
        x_top_left = x - w / 2.
        y_top_left = y - h / 2.
        x_left, y_left = max(0, x_top_left), max(0, y_top_left)
        wi, hi = min(orig_w, w), min(orig_h, h)
        result[i][0], result[i][1], result[i][2], result[i][3] = x_left, y_left, wi, hi
    return result


if __name__ == '__main__':
    img = cv2.imread('F:\\mypicture\\z5.jpg')
    pred = main('F:\\0-300_1200.ckpt', img)
    print(pred[..., 4])
    print("=============")
    pred = nms(pred, 0.05, 0.4)
    print(pred)
    ret_img = draw(img, pred)
    print("ceshi")
    print(ret_img)
    ret_img = ret_img[:, :, ::-1]

    plt.imshow(ret_img)
    plt.show()