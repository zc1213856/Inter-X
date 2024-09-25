#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   common_utils.py
@Time    :   2023/07/20 22:16:15
@Author  :   Chen Zhu 
@Version :   1.0
@Contact :   zc1213856@qq.com
'''

import math
import os
import pickle
import sys
from copy import deepcopy

import cv2
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R




def save_pkl(path, result):
    """"
    save pkl file
    """
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(path, 'wb') as result_file:
        pickle.dump(result, result_file, protocol=2)

def extract():
    # with open(R'.\trackingseqout\S9__Walking__cam_0.pkl','rb') as f:
    with open(R'E:\ChenZhu\data\h36m_annot\test.pkl','rb') as f:
        content = pickle.load(f)
    # seq = deepcopy(content[0])
    for i,item in enumerate(content):
        seq = deepcopy(content[i]) # images/0029/Camera05
        # seq = deepcopy(content[6]) # 'images/0027/Camera00/00000.jpg'
        seqname = seq[0]['img_path'].split('/')[1]
        motionname = seq[0]['img_path'].split('/')[2]
        camname = seq[0]['img_path'].split('/')[3]
        filename = seqname + '__' + motionname + '__' + camname + '.pkl'
        print(filename)
        temp = []
        temp.append(seq)
        save_pkl('E:\\ChenZhu\\data\\h36m_annot_single\\'+filename,temp)

def cutclip(pklpath,s,e):
    data = joblib.load(pklpath)
    new = []
    for i in range(s,e+1):
        new.append(data[0][i])
    save_pkl(pklpath.rstrip('.pkl')+f'clip{s}_{e}'+'.pkl', new)

def img2mp4(image_folder_dir,fps=30):
    print(f'image_folder_dir = {image_folder_dir}')
    # size = (w, h)     # (width, height) 数值可根据需要进行调整
    image_list = sorted([name for name in os.listdir(image_folder_dir)])     # 获取文件夹下所有格式为 jpg 图像的图像名，并按时间戳进行排序
    image0 = cv2.imread(os.path.join(image_folder_dir, image_list[0]))
    size = (image0.shape[1],image0.shape[0]) # w, h
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4','v') # 编码为 mp4v 格式，注意此处字母为小写，大写会报错
    video = cv2.VideoWriter(image_folder_dir +'.mp4', fourcc, fps, size, isColor=True)
    for image_name in image_list:     # 遍历 image_list 中所有图像并添加进度条
        image_full_path = os.path.join(image_folder_dir, image_name)     # 获取图像的全路经
        # print(image_full_path)
        image = cv2.imread(image_full_path)     # 读取图像
        video.write(image)     # 将图像写入视频
    video.release()
    cv2.destroyAllWindows()
    print('video ok!')

def video2img(path, target_path):
    # VideoCapture视频读取类
    videoCapture = cv2.VideoCapture()
    videoCapture.open(path)
    # 帧率
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    # 总帧数
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps=", int(fps), "frames=", int(frames))
    
    if not os.path.isdir(target_path): 
        os.makedirs(target_path)
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        # cv2.imwrite(os.path.join(target_path,'%5d.jpg'%(i)), frame)
        cv2.imwrite(os.path.join(target_path,str(i).zfill(5)+'.jpg'), frame)
    return

def project2d(jt_3d, intri, extri=None):
    # extri np.4*4, intri np.3*3, jt_3d np.len*3
    if extri is None:
        f = [intri[0,0],intri[1,1],1]
        c = [intri[0,2],intri[1,2],0]
        jt_2d_1 = []
        for p in jt_3d:
            p = p/p[2]
            p = p *f+c
            jt_2d_1.append(p.tolist())

    else:
        extri = extri[0:3]
        intri = intri
        jt_3d_4t = np.column_stack((jt_3d,np.array([1]*jt_3d.shape[0]))).T
        # halpe_jt_3d = frame[key]['halpe_joints_3d']
        jt_2d = (intri @ extri @ jt_3d_4t).T
        jt_2d_1 = []
        for p in jt_2d:
            p = np.rint(p/p[2])
            jt_2d_1.append(p.tolist())
    return jt_2d_1
                        
def matrix2euler(rm):
    # rm :list
    Rm = np.array(rm)
    r3 = R.from_matrix(Rm)
    # qua = r3.as_quat()
    euler = r3.as_euler('yzx',degrees=False)
    return euler.tolist()



def main():
    pass

if __name__ == '__main__':
    main()

