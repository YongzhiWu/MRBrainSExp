#-*-coding:utf-8-*-

import numpy as np
import cv2 as cv

# 训练时类别
# 0.    Background
# 1.    Cortical gray matter (皮质灰质)
# 2.    Basal ganglia (基底神经节 )
# 3.    White matter (白质)
# 4.    White matter lesions （白质组织）
# 5.    Cerebrospinal fluid in the extracerebral space （脑脊液）
# 6.    Ventricles （脑室）
# 7.    Cerebellum （小脑）
# 8.    Brainstem （脑干）
# 测试时类别，类别合并
# 0.    Background
# 1.    Cerebrospinal fluid (including ventricles)
# 2.    Gray matter (cortical gray matter and basal ganglia)
# 3.    White matter (including white matter lesions)
# class_train: [0, 1, 2, 3, 4, 5, 6, 7, 8]
# class_test:[0, 2, 2, 3, 3, 1, 1, 0, 0]

class_train = [0, 1, 2, 3, 4, 5, 6, 7, 8]
class_test = [0, 2, 2, 3, 3, 1, 1, 0, 0]

#color_space = np.array([[0,0,0],[45,45,45],[180,180,180],[127,127,127],[37,165,165],\
#        [255,255,255],[44,172,172],[190,60,60],[187,59,187]]).astype(np.uint8)
color_space = np.asarray([[0,0,0],[0,0,255],[0,255,0],[0,255,255],[255,0,0],\
        [255,0,255],[255,255,0],[255,255,255],[0,0,128],[0,128,0],[128,0,0]]).astype(np.uint8)
color_space_test = np.asarray([[0,0,0],[0,0,255],[0,255,0],[255,0,0]]).astype(np.uint8)

def merge_class(label):
    label = label.astype(np.int)
    for i in range(len(class_train)):
        label[label == class_train[i]] = class_test[i]
    return label.astype(np.uint8)

def gray2rgb(mask, color=color_space):
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            rgb_mask[i, j] = color[mask[i, j]]
    return rgb_mask.astype(np.uint8)

if __name__ == "__main__":
    img = cv.imread("../dataloader/sample/7_mask.png", 0)
    cv.imshow("Mask", img)
    cv.imshow("RGB_Mask", gray2rgb(img))
    cv.imshow("Merge_Mask", gray2rgb(merge_class(img), color_space_test))
    cv.waitKey(0)
    cv.destroyAllWindows()
    