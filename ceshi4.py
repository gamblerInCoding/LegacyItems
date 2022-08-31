
import os
import numpy as np
import mindspore as ms
from src.yolo import YOLOV5s
import cv2
import math
import argparse

numbers = 0
marks1 = np.zeros((320, 240), int)
marks2 = np.zeros((320, 240), int)





def nms(pred, conf_thres, iou_thres):

    conf = pred[..., 4] > conf_thres
    box = pred[conf == True]

    cls_conf = box[..., 5:]
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))

    total_cls = list(set(cls))
    output_box = []

    for i in range(len(total_cls)):
        clss = total_cls[i]

        cls_box = []
        for j in range(len(cls)):
            if cls[j] == clss:
                box[j][5] = clss
                cls_box.append(box[j][:6])
        cls_box = np.array(cls_box)
        box_conf = cls_box[..., 4]
        box_conf_sort = np.argsort(box_conf)
        max_conf_box = cls_box[box_conf_sort[len(box_conf) - 1]]
        output_box.append(max_conf_box)
        cls_box = np.delete(cls_box, 0, 0)
        while len(cls_box) > 0:
            max_conf_box = output_box[len(output_box) - 1]
            del_index = []
            for j in range(len(cls_box)):
                current_box = cls_box[j]
                interArea = getInter(max_conf_box, current_box)
                iou = getIou(max_conf_box, current_box, interArea)
                if iou > iou_thres:
                    del_index.append(j)
            cls_box = np.delete(cls_box, del_index, 0)
            if len(cls_box) > 0:
                output_box.append(cls_box[0])
                cls_box = np.delete(cls_box, 0, 0)
    return output_box



def getIou(box1, box2, inter_area):
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - inter_area
    iou = inter_area / union
    return iou



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

            labels = ['box', 'schoolbag']

            print(x1, y1, x2, y2, score, cls)

            img_ = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

            text = labels[int(cls)]
            cv2.putText(img, text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, )
    return img_

def draw2(img, pred):
    img_ = img.copy()

    if len(pred):
        for detect in pred:
            x1 = int(detect[0])
            y1 = int(detect[1])
            x2 = int(detect[0] + detect[2])
            y2 = int(detect[1] + detect[3])
            score = detect[4]
            cls = detect[5]

            labels = [ 'person']

            print(x1, y1, x2, y2, score, cls)

            img_ = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

            text = labels[int(cls)]
            cv2.putText(img, text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, )
    return img_



def draw3(img,pred,pred2):
    marks1 = np.zeros((320, 240), int)
    img_ = img.copy()
    personList = []
    if len(pred2):
        for detect in pred2:
            x1 = int(detect[0])
            y1 = int(detect[1])
            x2 = int(detect[0] + detect[2])
            y2 = int(detect[1] + detect[3])
            score = detect[4]
            cls = detect[5]
            labels = ['person']
            arr = ((x1+x2)/2,(y1+y2)/2,math.sqrt((x1-x2)/2*(x1-x2)/2+(y1-y2)/2*(y1-y2)/2))
            personList.append(arr)
    print("识别出的人:=====")
    print(personList)
    print("===========")
    if len(pred):
        for detect in pred:
            mark = 0
            x1 = int(detect[0])
            y1 = int(detect[1])
            x2 = int(detect[0] + detect[2])
            y2 = int(detect[1] + detect[3])
            score = detect[4]
            cls = detect[5]
            labels = ['box']
            thingx = (x1+x2)/2
            thingy = (y1+y2)/2
            distance2 = math.sqrt((x1-x2)/2*(x1-x2)/2+(y1-y2)/2*(y1-y2)/2)
            print("识别出的物品:=====")
            arrak = (thingx, thingy, distance2)
            print(arrak)
            print("===========")
            for i in range(len(personList)):
                arrAy = personList[i]
                personx = arrAy[0]
                persony = arrAy[1]
                distance1 = arrAy[2]
                if(((thingx-personx)*(thingx-personx)+(thingy-persony)*(thingy-persony))<=((distance1+distance2)*(distance1+distance2))):
                    mark = 1
            if(mark==1):
                print("非遗留物品+1")
            else:

                marks1[math.ceil((x1+x2)/2)][math.ceil((y1+y2)/2)]=1
                marks2[math.ceil((x1+x2)/2)][math.ceil((y1+y2)/2)]= marks2[math.ceil((x1+x2)/2)][math.ceil((y1+y2)/2)]+1
                if marks2[math.ceil((x1+x2)/2)][math.ceil((y1+y2)/2)]>=5:
                    img_ = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255),3)
                    text = labels[int(cls)]
                    cv2.putText(img, "items", (int((x1+x2)/2)-10,int((y1+y2)/2)-10), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 255), 3 )

    for i in range(320):
        for k in range(240):
            if marks1[i][k] == 0:
                marks2[i][k] = 0
    return img_


def draw4(img,pred):
    marks1 = np.zeros((320, 240), int)
    img_ = img.copy()
    personList = []
    if len(pred):
        for detect in pred:
            x1 = int(detect[0])
            y1 = int(detect[1])
            x2 = int(detect[0] + detect[2])
            y2 = int(detect[1] + detect[3])
            cls = detect[5]
            labels = ['person','backpack','umbrella','handbag','tie','suitcase','bottle','laptop','mouse','remote','cell phone','book']
            if labels[int(cls)]== 'person':
               arr = ((x1+x2)/2,(y1+y2)/2,math.sqrt((x1-x2)/2*(x1-x2)/2+(y1-y2)/2*(y1-y2)/2))
               personList.append(arr)
    print("识别出的人:=====")
    print(personList)
    print("===========")
    if len(pred):
        for detect in pred:
            mark = 0
            x1 = int(detect[0])
            y1 = int(detect[1])
            x2 = int(detect[0] + detect[2])
            y2 = int(detect[1] + detect[3])
            cls = detect[5]
            if labels[int(cls)] is not 'person':
               labels = ['person','backpack','umbrella','handbag','tie','suitcase','bottle','laptop','mouse','remote','cell phone','book']
               thingx = (x1+x2)/2
               thingy = (y1+y2)/2
               distance2 = math.sqrt((x1-x2)/2*(x1-x2)/2+(y1-y2)/2*(y1-y2)/2)
               print("识别出的物品:=====")
               arrak = (thingx, thingy, distance2)
               print(arrak)
               print("===========")
               for i in range(len(personList)):
                   arrAy = personList[i]
                   personx = arrAy[0]
                   persony = arrAy[1]
                   distance1 = arrAy[2]
                   if(((thingx-personx)*(thingx-personx)+(thingy-persony)*(thingy-persony))<=((distance1+distance2)*(distance1+distance2))):
                       mark = 1
               if(mark==1):
                   print("非遗留物品+1")
               else:
                   marks1[math.ceil((x1+x2)/2)][math.ceil((y1+y2)/2)]=1
                   marks2[math.ceil((x1+x2)/2)][math.ceil((y1+y2)/2)]= marks2[math.ceil((x1+x2)/2)][math.ceil((y1+y2)/2)]+1
                   if marks2[math.ceil((x1+x2)/2)][math.ceil((y1+y2)/2)]>=5:
                      img_ = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255),1.5)
                      text = labels[int(cls)]
                      cv2.putText(img, "item", (int((x1+x2)/2)-10,int((y1+y2)/2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1.5 )
    for i in range(320):
        for k in range(240):
            if marks1[i][k] == 0:
                marks2[i][k] = 0
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
def pred_tuili(imgk):
    img = cv2.imread(imgk)
    # pred1 = main('F:\\0-300_1200.ckpt', img)
    # pred2 = main('F:\\0-29_87600.ckpt', img)
    # pred2 = nms(pred2, 0.1, 0.4)
    # pred1 = nms(pred1, 0.005, 0.4)
    # ret_img1 = draw3(img,pred1,pred2)
    # ret_img1 = ret_img1[:, :, ::-1]
    predx = main('F:\\.ckpt',img)
    predx = nms(predx,0.1,0.4)
    ret_img = draw4(img,predx)
    ret_img = ret_img[:, :, ::-1]
    return ret_img












ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target='CPU', device_id=0)
video_path = 'F:\\viptrain.avi'
video_outpath = "results.avi"
timeF = 1
images_path = video_path.split('.',1)[0]
if not os.path.exists(images_path):  # ⽂件夹不存在则新建
    os.mkdir(images_path)
vc = cv2.VideoCapture(video_path) # 读入视频文件c=1
c=1
rat=1
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
witer = cv2.VideoWriter(video_outpath,fourcc,25,(320,240),True)
if vc.isOpened():
   while rat:
       rat, frame = vc.read()
       if not rat:break
       cv2.imwrite("F:\\1.jpg",frame)
       imgk = "F:\\1.jpg"
       if((c%5)==0):
           newframw = pred_tuili(imgk)
           witer.write(newframw)
       c=c+1
   vc.release()
witer.release()
