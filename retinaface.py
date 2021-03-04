import colorsys
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

from nets.retinaface import RetinaFace
from utils.anchors import Anchors
from utils.box_utils import (decode, decode_landm, letterbox_image,
                             non_max_suppression, retinaface_correct_boxes)
from utils.config import cfg_mnet, cfg_re50


def preprocess_input(image):
    image -= np.array((104, 117, 123),np.float32)
    return image

#------------------------------------#
#   请注意主干网络与预训练权重的对应
#   即注意修改model_path和backbone
#------------------------------------#
class Retinaface(object):
    _defaults = {
        "model_path"        : 'model_data/Retinaface_mobilenet0.25.pth',
        "backbone"          : 'mobilenet',
        "confidence"        : 0.5,
        "nms_iou"           : 0.45,
        "cuda"              : True,
        #----------------------------------------------------------------------#
        #   是否需要进行图像大小限制。
        #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
        #   可根据输入图像的大小自行调整input_shape，注意为32的倍数，如[640, 640, 3]
        #----------------------------------------------------------------------#
        "input_shape"       : [1280, 1280, 3],
        "letterbox_image"   : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Retinaface
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        if self.backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50
        self.generate()
        if self.letterbox_image:
            self.anchors = Anchors(self.cfg, image_size=[self.input_shape[0], self.input_shape[1]]).get_anchors()

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        self.net = RetinaFace(cfg=self.cfg, mode='eval').eval()

        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        print('Loading weights into state dict...')
        state_dict = torch.load(self.model_path)
        self.net.load_state_dict(state_dict)
        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
        print('Finished!')

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, picpath, frameNo):
        #---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        #---------------------------------------------------#
        old_image = image.copy()

        image = np.array(image,np.float32)

        #---------------------------------------------------#
        #   计算scale，用于将获得的预测框转换成原图的高宽
        #---------------------------------------------------#
        scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
        scale_for_landmarks = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0]]

        im_height, im_width, _ = np.shape(image)
        #---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        if self.letterbox_image:
            image = np.array(letterbox_image(image, [self.input_shape[1], self.input_shape[0]]), np.float32)
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
            
        with torch.no_grad():
            #-----------------------------------------------------------#
            #   图片预处理，归一化。
            #-----------------------------------------------------------#
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0)

            if self.cuda:
                self.anchors = self.anchors.cuda()
                image = image.cuda()

            loc, conf, landms = self.net(image)
            
            #-----------------------------------------------------------#
            #   将预测结果进行解码
            #-----------------------------------------------------------#
            boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            boxes = boxes.cpu().numpy()

            conf = conf.data.squeeze(0)[:,1:2].cpu().numpy()
            
            landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])
            landms = landms.cpu().numpy()

            boxes_conf_landms = np.concatenate([boxes, conf, landms],-1)
            #到这里为止我们已经利用Retinaface_pytorch的预训练模型检测完了人脸，并获得了人脸框和人脸五个特征点的坐标信息，全保存在dets中，接下来为人脸剪切部分
            # 用来储存生成的单张人脸的路径
            path_save = "./curve/faces/" #你可以将这里的路径换成你自己的路径
            '''
        #剪切图片
        #if args.show_cutting_image：
            for num, b in enumerate(boxes_conf_landms): # dets中包含了人脸框和五个特征点的坐标
                #if b[4] < 0.6:
                #    continue
                b = list(map(int, b))

                # landms，在人脸上画出特征点，要是你想保存不显示特征点的人脸图，你可以把这里注释掉
                cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)

                #计算人脸框矩形大小
                Height = b[3] - b[1]
                Width = b[2] - b[0]
              
                # 显示人脸矩阵大小
                print("人脸数 / faces in all:", str(num+1), "\n")
                print("窗口大小 / The size of window:"
                      , '\n', "高度 / height:", Height
                      , '\n', "宽度 / width: ", Width)
                
                #根据人脸框大小，生成空白的图片
                img_blank = np.zeros((Height, Width, 3), np.uint8)
                # 将人脸填充到空白图片
                for h in range(Height):
                    for w in range(Width):
                        img_blank[h][w] = old_image[b[1] + h][b[0] + w]
                       
                cv2.namedWindow("img_faces")  # , 2)
                #cv2.imshow("img_faces", img_blank)  #显示图片
                cv2.imwrite(path_save + "img_face_4" + str(num + 1) + ".jpg", img_blank)  #将图片保存至你指定的文件夹
                print("Save into:", path_save + "img_face_4" + str(num + 1) + ".jpg")
                cv2.waitKey(0)

            '''
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
            if len(boxes_conf_landms)<=0:
                return False, old_image
            #---------------------------------------------------------#
            #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
            #---------------------------------------------------------#
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                    np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
            
        boxes_conf_landms[:,:4] = boxes_conf_landms[:,:4]*scale
        boxes_conf_landms[:,5:] = boxes_conf_landms[:,5:]*scale_for_landmarks
        num = 0
        for b in boxes_conf_landms:            
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))

            # b[0]-b[3]为人脸框的坐标，b[4]为得分
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(old_image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            print(b[0], b[1], b[2], b[3], b[4])

            c = [-50, -50,50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            #b += c
            c = np.array(b) + np.array(c)
            if (c > 0).all() :
                    b = c
            else:
                for i, ind in enumerate(b[:4]):
                    if ind < 0:
                        print(i, b[i])
                        b[i] = 0

            print(b[0], b[1], b[2], b[3], b[4])
            #计算人脸框矩形大小
            Height = b[3] - b[1]
            Width = b[2] - b[0]

            print("窗口大小 / The size of window:"
                      , '\n', "高度 / height:", Height
                      , '\n', "宽度 / width: ", Width)

            img_blank = old_image[int(b[1]):int(b[3]), int(b[0]):int(b[2])] # height, width
                    
            savepath = picpath + "/" + str(frameNo) + "_" + str(num)+".jpg"        
            #cv2.namedWindow("img_faces")  # , 2)
            #cv2.imshow("img_faces", img_blank)  #显示图片
            #img_blank = cv2.cvtColor(img_blank,cv2.COLOR_RGB2BGR)
            cv2.imwrite(savepath, img_blank)  #将图片保存至你指定的文件夹
            print("Save into:", savepath)
            #cv2.waitKey(0)          
            '''
            # b[5]-b[14]为人脸关键点的坐标
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)
            '''
            num += 1
        return True, old_image
