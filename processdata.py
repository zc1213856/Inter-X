'''
@File    :   processdata.py
@Time    :   2024/09/21 15:06:49
@Author  :   Chen Zhu 
@Version :   1.0
@Contact :   zc1213856@163.com
'''

#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   chi3d.py
@Time    :   2023/07/21 15:15:40
@Author  :   Chen Zhu 
@Version :   1.0
@Contact :   zc1213856@qq.com
'''

import json
import os
import pickle
from copy import deepcopy

import cv2
import joblib
import numpy as np
import torch
from tqdm import tqdm

import smplx
from utils.common_utils import *
from utils.FileLoaders import *
from utils.render import Renderer
from utils.rotation_conversions import *
from utils.smpl_torch_batch import SMPLModel


from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.renderables.plane import Plane
from aitviewer.viewer import Viewer
from scipy.spatial.transform import Rotation as R

from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence



C.update_conf({'smplx_models':'./body_models'})


def extract_image():
    datadir = R'G:\zc_data\origin_CHI3D'
    targetpath = R'G:\zc_data\CHI3D'
    # splitlist = ['train','test']


    # train split
    if False:
        splitpath = os.path.join(datadir,'train')
        for seq in os.listdir(splitpath):
            videopath = os.path.join(splitpath,seq,'videos')
            for camdix, cam in enumerate(os.listdir(videopath)):
                campath = os.path.join(videopath,cam)
                # camname = 'cam_'+str(camdix)
                for motion in os.listdir(campath):
                    filepath = os.path.join(campath,motion)

                    newmotion = motion.replace('.mp4','',1).replace('\x20','_',1)
                    print(filepath,newmotion)
                    imgpath = os.path.join(targetpath,seq,newmotion,cam)
                    video2img(filepath,imgpath)

    # test split
    if True:
        splitpath = os.path.join(datadir,'test')
        for seq in os.listdir(splitpath):
            videopath = os.path.join(splitpath,seq,'videos')
            for motion in os.listdir(videopath):
                filepath = os.path.join(videopath,motion)

                newmotion = motion.replace('.mp4','',1).replace('\x20','_',1)
                print(filepath,newmotion)
                imgpath = os.path.join(targetpath,seq,newmotion)
                video2img(filepath,imgpath)



def main1():
    smpl = SMPLModel(device=torch.device('cpu'),
                      model_path='data/smpl/SMPL_NEUTRAL.pkl',
                      data_type=torch.float)
    smplx_model = smplx.create('data', model_type='smplx',
                         gender='neutral', use_face_contour=False,
                         num_betas=10,
                         num_expression_coeffs=10,
                         ext='npz')
    smplx_layer = smplx.build_layer(
        'data', model_type='smplx',
        gender='neutral', use_face_contour=False,
        num_betas=10,
        num_expression_coeffs=10,
        ext='npz')
    datadir = R'G:\zc_data\origin_CHI3D'
    # imagedir = R'G:\zc_data\CHI3D'
    savepath = '.\\'
    # split = ['train','validation','test']
    # with open(R'G:\zc_data\origin_CHI3D\chi3d_info.json') as f:
    #     j2 = json.load(f)    
    # for train split
    d = []
    splitpath = os.path.join(datadir,'train')
    for seq in os.listdir(splitpath):
        camparam_path = os.path.join(splitpath,seq,'camera_parameters')
        for camdix, cam in enumerate(os.listdir(camparam_path)):
            campath = os.path.join(camparam_path,cam)
            # camname = 'cam_'+str(camdix)
            for motion in os.listdir(campath):
                camparam_filepath = os.path.join(campath,motion)
                with open(camparam_filepath) as f1:
                    camparam = json.load(f1)
                extri = camparam['extrinsics']['R']
                e_np = np.array(extri)
                t_np = np.array([camparam['extrinsics']['T'][0][0],
                                camparam['extrinsics']['T'][0][1],
                                camparam['extrinsics']['T'][0][2]])
                t_new = - e_np @ t_np
                extri[0].append(t_new[0])
                extri[1].append(t_new[1])
                extri[2].append(t_new[2])
                extri.append([0,0,0,1])
                intri = [[1,0,0],[0,1,0],[0,0,1]]
                intri[0][0] = camparam['intrinsics_wo_distortion']['f'][0]
                intri[1][1] = camparam['intrinsics_wo_distortion']['f'][1]
                intri[0][2] = camparam['intrinsics_wo_distortion']['c'][0]
                intri[1][2] = camparam['intrinsics_wo_distortion']['c'][1]
                # with open(os.path.join(splitpath,seq,'interaction_contact_signature.json')) as f4:
                #     test = json.load(f4)
                # gpppath = os.path.join(splitpath,seq,'gpp',motion)
                # with open(gpppath) as f4:
                #     gpp = json.load(f4)
                smplx_filepath = os.path.join(splitpath,seq,'smplx',motion)
                with open(smplx_filepath) as f2:
                    smplxp = json.load(f2)
                transls = smplxp['transl']
                betas = smplxp['betas']
                global_orients = smplxp['global_orient']
                body_poses = smplxp['body_pose']
                poses = []
                for person, p_global_orients in enumerate(global_orients):
                    p_poses = []
                    for idx,bpos in enumerate(p_global_orients):
                        pose = []
                        pose.extend(matrix_to_axis_angle(torch.from_numpy(np.array(bpos[0])).to(torch.float32)).tolist())
                        # pose.append(matrix2euler(bpos[0]))
                        for joint in body_poses[person][idx]:
                            pose.extend((matrix_to_axis_angle(torch.from_numpy(np.array(joint)).to(torch.float32)).tolist()))
                        pose.extend([0,0,0])
                        pose.extend([0,0,0])
                        p_poses.append(pose)
                    poses.append(p_poses)
                joint3d_filepath = os.path.join(splitpath,seq,'joints3d_25',motion)
                with open(joint3d_filepath) as f3:
                    jtparam = json.load(f3)
                joints3d = jtparam['joints3d_25']
                newmotion = motion.replace('.json','',1).replace('\x20','_',1)
                relpath = os.path.join('images','train',seq,newmotion,cam)
                imgnames = os.listdir(os.path.join(datadir,relpath))
                image0 = cv2.imread(os.path.join(datadir,relpath,imgnames[0]))
                hw = (image0.shape[0],image0.shape[1])
                render = Renderer(resolution=(int(hw[1]), int(hw[0])))
                s = []
                pose_lenth = len(poses[0])
                for f_idx,imgname in enumerate(imgnames):
                    if f_idx >= pose_lenth:
                        break
                    img_path = os.path.join(relpath,imgname)
                    f = dict(h_w=hw,img_path=img_path)
                    debug = True # render mesh to img
                    if debug and f_idx ==1:
                        ipath = os.path.join(datadir,img_path)
                        img = cv2.imread(ipath)
                    for person, p_poses in enumerate(poses):
                        p = torch.from_numpy(np.array(p_poses[f_idx])).reshape(-1, 72).to(torch.float32)
                        t = torch.from_numpy(np.array(transls[person][f_idx])).reshape(-1, 3).to(torch.float32)
                        b = torch.from_numpy(np.array(betas[person][f_idx])).reshape(-1, 10).to(torch.float32)
                        verts, smpl_jt_3d = smpl(b, p, t)
                        write_obj(verts[0].detach().cpu().numpy(), smpl.faces, os.path.join('.\\outmesh', 'smpl_%05d.obj' %f_idx))
                        smpl_jt_3d = smpl_jt_3d.tolist()[0]
                        _, halpe_jt_3d = smpl(b, p, t,halpe=True)
                        halpe_jt_3d = halpe_jt_3d.tolist()[0]
                        smpl_jt_2d = project2d(np.array(smpl_jt_3d),np.array(intri),np.array(extri))
                        halpe_jt_2d = project2d(np.array(halpe_jt_3d),np.array(intri),np.array(extri))
                        jt_2d = project2d(np.array(joints3d[person][f_idx]),np.array(intri),np.array(extri))
                        
                        debug1 = False #测试计算的矩阵是否和原来的处理方式等价
                        if debug1 : 
                            with open(camparam_filepath) as ff:
                                camparam = json.load(ff)
                            mat1 = np.array(joints3d[f_idx]) - np.array(camparam['extrinsics']['T'])
                            mat2 = np.transpose(np.array(camparam['extrinsics']['R']))
                            j3d_in_camera = np.matmul(mat1,mat2)
                            # if intrinsics_type == 'w_distortion':
                            p = np.array(camparam['intrinsics_w_distortion']['p'])[:, [1, 0]]
                            x = j3d_in_camera[:, :2] / j3d_in_camera[:, 2:3]
                            r2 = np.sum(x**2, axis=1)
                            radial = 1 + np.transpose(np.matmul(np.array(camparam['intrinsics_w_distortion']['k']), np.array([r2, r2**2, r2**3])))
                            tan = np.matmul(x, np.transpose(p))
                            xx = x*(tan + radial) + r2[:, np.newaxis] * p
                            jt2d_w = np.array(camparam['intrinsics_w_distortion']['f']) * xx + np.array(camparam['intrinsics_w_distortion']['c'])
                            # elif intrinsics_type == 'wo_distortion':
                            xx = j3d_in_camera[:, :2] / j3d_in_camera[:, 2:3]
                            jt2d_wo = np.array(camparam['intrinsics_wo_distortion']['f']) * xx + np.array(camparam['intrinsics_wo_distortion']['c'])
                        
                        # smpl_jt_2d = project2d(np.array(smpl_jt_3d),np.array(intri))
                        # halpe_jt_2d = project2d(np.array(halpe_jt_3d),np.array(intri))
                        # jt_2d = project2d(np.array(joints3d[f_idx]),np.array(intri))
                        debug2 = True #使用smplx库进行处理
                        if debug2 and person==0:
                            g_o = torch.from_numpy(np.array(p_poses[f_idx][0:3])).reshape(-1, 3).to(torch.float32)
                            bp = torch.from_numpy(np.array(p_poses[f_idx][3:66])).reshape(-1, 63).to(torch.float32)
                            tr = torch.from_numpy(np.array(transls[person][f_idx])).reshape(-1, 3).to(torch.float32)
                            be = torch.from_numpy(np.array(betas[person][f_idx])).reshape(-1, 10).to(torch.float32)
                            # expr =  torch.from_numpy(np.array(smplxp['expression'][person][f_idx])).reshape(-1, 10).to(torch.float32)
                            output = smplx_model(global_orient = g_o,betas = be,body_pose=bp, transl= tr)
                            verts = output.vertices
                            write_obj(verts[0].detach().cpu().numpy(), smplx_model.faces, os.path.join('.\\outmesh', 'smplx_%05d.obj' %f_idx))
                            print('debug')
                            # world_smplx_params = {key: torch.from_numpy(np.array(smplxp[key][0]).astype(np.float32)) for key in smplxp}
                            # output = smplx_layer(**world_smplx_params)
                            # verts = output.vertices[0:1]
                            # print('debug')


                        f.update({str(person):dict(pose=poses[person][f_idx],
                                                betas=betas[person][f_idx],
                                                trans=transls[person][f_idx],
                                                joint_3d=joints3d[person][f_idx],
                                                joint_2d = jt_2d,
                                                intri=intri,
                                                extri=extri,
                                                smpl_joints_3d=smpl_jt_3d,
                                                halpe_joints_3d=halpe_jt_3d,
                                                smpl_joints_2d=smpl_jt_2d,
                                                halpe_joints_2d=halpe_jt_2d )})
                        if debug and f_idx ==1:
                            # ipath = os.path.join(datadir,img_path)
                            # img = cv2.imread(ipath)
                            for point in smpl_jt_2d:
                                cv2.circle (img, (int(point[0]),int(point[1])), 5, (0, 0, 255) ,thickness=-1)
                            for point in halpe_jt_2d:
                                cv2.circle (img, (int(point[0]),int(point[1])), 5, (0, 255, 0) ,thickness=-1)
                            for point in jt_2d:
                                cv2.circle (img, (int(point[0]),int(point[1])), 5, (255, 255, 0) ,thickness=-1)
                            # for point in jt2d_w:
                            #     cv2.circle (img, (int(point[0]),int(point[1])), 3, (255, 0, 0) ,thickness=-1)
                            # for point in jt2d_wo:
                            #     cv2.circle (img, (int(point[0]),int(point[1])), 1, (0, 255, 0) ,thickness=-1)

                            render2img = True
                            if render2img and f_idx==1:
                                rot = np.array(extri)[0:3,0:3]
                                # rot = np.array([[1,0,0],[0,1,0],[0,0,1]])
                                trans_render = np.array(extri)[0:3,3]
                                # trans_render = np.array([0,0,0])
                                intri = np.array(intri)
                                # ef= 0.5* math.sqrt(intri[0,0] **2 + intri[1,1]**2) 
                                # intri[0,0] = intri[1,1] = ef
                                # rot = np.array([[0,0,-1],[0,1,0],[1,0,0]])
                                if debug2 and person ==0:
                                    img = render.render_multiperson(verts.detach().cpu().numpy(), smplx_model.faces, rot, trans_render , intri.copy(), img.copy(), viz=False)
                                else :
                                    img = render.render_multiperson(verts.detach().cpu().numpy(), smpl.faces, rot, trans_render , intri.copy(), img.copy(), viz=False)
                                # render.vis_img('img', img)
                                # output_path = os.path.join(savepath,str(idx).zfill(5)+'.jpg')
                                # cv2.imwrite(output_path,img)
                                if person == 1:
                                    cv2.imwrite('.\\renderimg\\'+seq+'_'+newmotion+'_'+cam+'_'+imgnames[f_idx], img)

                    s.append(f)
                print(smplx_filepath)
                d.append(s)
    save_pkl(savepath+'train'+'.pkl',d)

def main2():
    
    # for test split
    with open(R'G:\zc_data\origin_CHI3D\chi3d_template.json') as f:
        data = json.load(f)
    for seq_key in data:
        for motion_key in data[seq_key]:
            
            # for p in d
            pass
    smplx_model = smplx.create('data', model_type='smplx',
                         gender='neutral', use_face_contour=False,
                         num_betas=10,
                         num_expression_coeffs=10,
                         ext='npz')
    smplx_layer = smplx.build_layer(
        'data', model_type='smplx',
        gender='neutral', use_face_contour=False,
        num_betas=10,
        num_expression_coeffs=10,
        ext='npz')
    datadir = R'G:\zc_data\origin_CHI3D'
    # imagedir = R'G:\zc_data\CHI3D'
    savepath = '.\\'
    # split = ['train','validation','test']
    # with open(R'G:\zc_data\origin_CHI3D\chi3d_info.json') as f:
    #     j2 = json.load(f)    
    # for train split
    d = []
    splitpath = os.path.join(datadir,'train')
    for seq in os.listdir(splitpath):
        camparam_path = os.path.join(splitpath,seq,'camera_parameters')
        for camdix, cam in enumerate(os.listdir(camparam_path)):
            campath = os.path.join(camparam_path,cam)
            # camname = 'cam_'+str(camdix)
            for motion in os.listdir(campath):
                camparam_filepath = os.path.join(campath,motion)
                with open(camparam_filepath) as f1:
                    camparam = json.load(f1)
                extri = camparam['extrinsics']['R']
                e_np = np.array(extri)
                t_np = np.array([camparam['extrinsics']['T'][0][0],
                                camparam['extrinsics']['T'][0][1],
                                camparam['extrinsics']['T'][0][2]])
                t_new = - e_np @ t_np
                extri[0].append(t_new[0])
                extri[1].append(t_new[1])
                extri[2].append(t_new[2])
                extri.append([0,0,0,1])
                intri = [[1,0,0],[0,1,0],[0,0,1]]
                intri[0][0] = camparam['intrinsics_wo_distortion']['f'][0]
                intri[1][1] = camparam['intrinsics_wo_distortion']['f'][1]
                intri[0][2] = camparam['intrinsics_wo_distortion']['c'][0]
                intri[1][2] = camparam['intrinsics_wo_distortion']['c'][1]
                # with open(os.path.join(splitpath,seq,'interaction_contact_signature.json')) as f4:
                #     test = json.load(f4)
                # gpppath = os.path.join(splitpath,seq,'gpp',motion)
                # with open(gpppath) as f4:
                #     gpp = json.load(f4)
                smplx_filepath = os.path.join(splitpath,seq,'smplx',motion)
                with open(smplx_filepath) as f2:
                    smplxp = json.load(f2)
                transls = smplxp['transl']
                betas = smplxp['betas']
                global_orients = smplxp['global_orient']
                body_poses = smplxp['body_pose']
                poses = []
                for person, p_global_orients in enumerate(global_orients):
                    p_poses = []
                    for idx,bpos in enumerate(p_global_orients):
                        pose = []
                        pose.extend(matrix_to_axis_angle(torch.from_numpy(np.array(bpos[0])).to(torch.float32)).tolist())
                        # pose.append(matrix2euler(bpos[0]))
                        for joint in body_poses[person][idx]:
                            pose.extend((matrix_to_axis_angle(torch.from_numpy(np.array(joint)).to(torch.float32)).tolist()))
                        pose.extend([0,0,0])
                        pose.extend([0,0,0])
                        p_poses.append(pose)
                    poses.append(p_poses)
                joint3d_filepath = os.path.join(splitpath,seq,'joints3d_25',motion)
                with open(joint3d_filepath) as f3:
                    jtparam = json.load(f3)
                joints3d = jtparam['joints3d_25']
                newmotion = motion.replace('.json','',1).replace('\x20','_',1)
                relpath = os.path.join('images','train',seq,newmotion,cam)
                imgnames = os.listdir(os.path.join(datadir,relpath))
                image0 = cv2.imread(os.path.join(datadir,relpath,imgnames[0]))
                hw = (image0.shape[0],image0.shape[1])
                render = Renderer(resolution=(int(hw[1]), int(hw[0])))
                s = []
                pose_lenth = len(poses[0])
                for f_idx,imgname in enumerate(imgnames):
                    if f_idx >= pose_lenth:
                        break
                    img_path = os.path.join(relpath,imgname)
                    f = dict(h_w=hw,img_path=img_path)
                    debug = True # render mesh to img
                    if debug and f_idx ==1:
                        ipath = os.path.join(datadir,img_path)
                        img = cv2.imread(ipath)
                    for person, p_poses in enumerate(poses):
                        p = torch.from_numpy(np.array(p_poses[f_idx])).reshape(-1, 72).to(torch.float32)
                        t = torch.from_numpy(np.array(transls[person][f_idx])).reshape(-1, 3).to(torch.float32)
                        b = torch.from_numpy(np.array(betas[person][f_idx])).reshape(-1, 10).to(torch.float32)
                        verts, smpl_jt_3d = smpl(b, p, t)
                        write_obj(verts[0].detach().cpu().numpy(), smpl.faces, os.path.join('.\\outmesh', 'smpl_%05d.obj' %f_idx))
                        smpl_jt_3d = smpl_jt_3d.tolist()[0]
                        _, halpe_jt_3d = smpl(b, p, t,halpe=True)
                        halpe_jt_3d = halpe_jt_3d.tolist()[0]
                        smpl_jt_2d = project2d(np.array(smpl_jt_3d),np.array(intri),np.array(extri))
                        halpe_jt_2d = project2d(np.array(halpe_jt_3d),np.array(intri),np.array(extri))
                        jt_2d = project2d(np.array(joints3d[person][f_idx]),np.array(intri),np.array(extri))
                        
                        debug1 = False #测试计算的矩阵是否和原来的处理方式等价
                        if debug1 : 
                            with open(camparam_filepath) as ff:
                                camparam = json.load(ff)
                            mat1 = np.array(joints3d[f_idx]) - np.array(camparam['extrinsics']['T'])
                            mat2 = np.transpose(np.array(camparam['extrinsics']['R']))
                            j3d_in_camera = np.matmul(mat1,mat2)
                            # if intrinsics_type == 'w_distortion':
                            p = np.array(camparam['intrinsics_w_distortion']['p'])[:, [1, 0]]
                            x = j3d_in_camera[:, :2] / j3d_in_camera[:, 2:3]
                            r2 = np.sum(x**2, axis=1)
                            radial = 1 + np.transpose(np.matmul(np.array(camparam['intrinsics_w_distortion']['k']), np.array([r2, r2**2, r2**3])))
                            tan = np.matmul(x, np.transpose(p))
                            xx = x*(tan + radial) + r2[:, np.newaxis] * p
                            jt2d_w = np.array(camparam['intrinsics_w_distortion']['f']) * xx + np.array(camparam['intrinsics_w_distortion']['c'])
                            # elif intrinsics_type == 'wo_distortion':
                            xx = j3d_in_camera[:, :2] / j3d_in_camera[:, 2:3]
                            jt2d_wo = np.array(camparam['intrinsics_wo_distortion']['f']) * xx + np.array(camparam['intrinsics_wo_distortion']['c'])
                        
                        # smpl_jt_2d = project2d(np.array(smpl_jt_3d),np.array(intri))
                        # halpe_jt_2d = project2d(np.array(halpe_jt_3d),np.array(intri))
                        # jt_2d = project2d(np.array(joints3d[f_idx]),np.array(intri))
                        debug2 = True #使用smplx库进行处理
                        if debug2 and person==0:
                            g_o = torch.from_numpy(np.array(p_poses[f_idx][0:3])).reshape(-1, 3).to(torch.float32)
                            bp = torch.from_numpy(np.array(p_poses[f_idx][3:66])).reshape(-1, 63).to(torch.float32)
                            tr = torch.from_numpy(np.array(transls[person][f_idx])).reshape(-1, 3).to(torch.float32)
                            be = torch.from_numpy(np.array(betas[person][f_idx])).reshape(-1, 10).to(torch.float32)
                            # expr =  torch.from_numpy(np.array(smplxp['expression'][person][f_idx])).reshape(-1, 10).to(torch.float32)
                            output = smplx_model(global_orient = g_o,betas = be,body_pose=bp, transl= tr)
                            verts = output.vertices
                            write_obj(verts[0].detach().cpu().numpy(), smplx_model.faces, os.path.join('.\\outmesh', 'smplx_%05d.obj' %f_idx))
                            print('debug')
                            # world_smplx_params = {key: torch.from_numpy(np.array(smplxp[key][0]).astype(np.float32)) for key in smplxp}
                            # output = smplx_layer(**world_smplx_params)
                            # verts = output.vertices[0:1]
                            # print('debug')


                        f.update({str(person):dict(pose=poses[person][f_idx],
                                                betas=betas[person][f_idx],
                                                trans=transls[person][f_idx],
                                                joint_3d=joints3d[person][f_idx],
                                                joint_2d = jt_2d,
                                                intri=intri,
                                                extri=extri,
                                                smpl_joints_3d=smpl_jt_3d,
                                                halpe_joints_3d=halpe_jt_3d,
                                                smpl_joints_2d=smpl_jt_2d,
                                                halpe_joints_2d=halpe_jt_2d )})
                        if debug and f_idx ==1:
                            # ipath = os.path.join(datadir,img_path)
                            # img = cv2.imread(ipath)
                            for point in smpl_jt_2d:
                                cv2.circle (img, (int(point[0]),int(point[1])), 5, (0, 0, 255) ,thickness=-1)
                            for point in halpe_jt_2d:
                                cv2.circle (img, (int(point[0]),int(point[1])), 5, (0, 255, 0) ,thickness=-1)
                            for point in jt_2d:
                                cv2.circle (img, (int(point[0]),int(point[1])), 5, (255, 255, 0) ,thickness=-1)
                            # for point in jt2d_w:
                            #     cv2.circle (img, (int(point[0]),int(point[1])), 3, (255, 0, 0) ,thickness=-1)
                            # for point in jt2d_wo:
                            #     cv2.circle (img, (int(point[0]),int(point[1])), 1, (0, 255, 0) ,thickness=-1)

                            render2img = True
                            if render2img and f_idx==1:
                                rot = np.array(extri)[0:3,0:3]
                                # rot = np.array([[1,0,0],[0,1,0],[0,0,1]])
                                trans_render = np.array(extri)[0:3,3]
                                # trans_render = np.array([0,0,0])
                                intri = np.array(intri)
                                # ef= 0.5* math.sqrt(intri[0,0] **2 + intri[1,1]**2) 
                                # intri[0,0] = intri[1,1] = ef
                                # rot = np.array([[0,0,-1],[0,1,0],[1,0,0]])
                                if debug2 and person ==0:
                                    img = render.render_multiperson(verts.detach().cpu().numpy(), smplx_model.faces, rot, trans_render , intri.copy(), img.copy(), viz=False)
                                else :
                                    img = render.render_multiperson(verts.detach().cpu().numpy(), smpl.faces, rot, trans_render , intri.copy(), img.copy(), viz=False)
                                # render.vis_img('img', img)
                                # output_path = os.path.join(savepath,str(idx).zfill(5)+'.jpg')
                                # cv2.imwrite(output_path,img)
                                if person == 1:
                                    cv2.imwrite('.\\renderimg\\'+seq+'_'+newmotion+'_'+cam+'_'+imgnames[f_idx], img)

                    s.append(f)
                print(smplx_filepath)
                d.append(s)
    save_pkl(savepath+'train'+'.pkl',d)



    print('debug')


def main():

    # path = R'E:\zc_codes\e-react\data\train_0_9.pkl'
    # p = joblib.load(path)
    datadir = R'F:\zc_data\Inter-X_Dataset'
    textdir = os.path.join(datadir, 'texts')
    motionsdir = os.path.join(datadir, 'motions')
    skelsdir = os.path.join(datadir, 'skeletons')
    names = os.listdir(motionsdir)
    smpl_joints_dir = os.path.join(datadir, 'smpl_joints')
    # data = []
    for name in tqdm(names):
        npy_folder = os.path.join(motionsdir, name)
        # skels_folder = os.path.join(skelsdir, name)
        smpl_joints_path = os.path.join(smpl_joints_dir, name)
        os.makedirs(smpl_joints_path, exist_ok=True)
        smpl_joints_path_p1 =  os.path.join(smpl_joints_path, 'P1.npy')
        smpl_joints_path_p2 =  os.path.join(smpl_joints_path, 'P2.npy')
        if os.path.exists(smpl_joints_path_p1) and os.path.exists(smpl_joints_path_p2):
            continue
        # load smplx
        smplx_path_p1 = os.path.join(npy_folder, 'P1.npz')
        smplx_path_p2 = os.path.join(npy_folder, 'P2.npz')
        params_p1 = np.load(smplx_path_p1, allow_pickle=True)
        params_p2 = np.load(smplx_path_p2, allow_pickle=True)
        nf_p1 = params_p1['pose_body'].shape[0]
        nf_p2 = params_p2['pose_body'].shape[0]

        betas_p1 = params_p1['betas']
        poses_root_p1 = params_p1['root_orient']
        poses_body_p1 = params_p1['pose_body'].reshape(nf_p1,-1)
        poses_lhand_p1 = params_p1['pose_lhand'].reshape(nf_p1,-1)
        poses_rhand_p1 = params_p1['pose_rhand'].reshape(nf_p1,-1)
        transl_p1 = params_p1['trans']
        gender_p1 = str(params_p1['gender'])

        betas_p2 = params_p2['betas']
        poses_root_p2 = params_p2['root_orient']
        poses_body_p2 = params_p2['pose_body'].reshape(nf_p2,-1)
        poses_lhand_p2 = params_p2['pose_lhand'].reshape(nf_p2,-1)
        poses_rhand_p2 = params_p2['pose_rhand'].reshape(nf_p2,-1)
        transl_p2 = params_p2['trans']
        gender_p2 = str(params_p2['gender'])
        # create body models
        smplx_layer_p1 = SMPLLayer(model_type='smplx',gender=gender_p1,num_betas=10,device=C.device)
        smplx_layer_p2 = SMPLLayer(model_type='smplx',gender=gender_p2,num_betas=10,device=C.device)
        # create smplx sequence for two persons   p1:blue  p2:red
        smplx_seq_p1 = SMPLSequence(poses_body=poses_body_p1,
                            smpl_layer=smplx_layer_p1,
                            poses_root=poses_root_p1,
                            betas=betas_p1,
                            trans=transl_p1,
                            poses_left_hand=poses_lhand_p1,
                            poses_right_hand=poses_rhand_p1,
                            device=C.device,
                            color=(0.11, 0.53, 0.8, 1.0)
                            )
        smplx_seq_p2 = SMPLSequence(poses_body=poses_body_p2, 
                            smpl_layer=smplx_layer_p2,
                            poses_root=poses_root_p2,
                            betas=betas_p2,
                            trans=transl_p2,
                            poses_left_hand=poses_lhand_p2,
                            poses_right_hand=poses_rhand_p2,
                            device=C.device,
                            # color=(0, 0.0, 0, 1.0)
                            color=(1.0, 0.27, 0, 1.0)
                            )
        _, joints_1, _, lines = smplx_seq_p1.fk()
        _, joints_2, _, lines = smplx_seq_p2.fk()
        np.save(smpl_joints_path_p1, joints_1, allow_pickle=True)
        np.save(smpl_joints_path_p2, joints_2, allow_pickle=True)
        
        # skel_path_p1 = os.path.join(skels_folder, 'P1.npy')
        # skel_p1 = np.load(skel_path_p1, allow_pickle=True)
        print(npy_folder)
        # seq = []
        # for t in range(len(transl_p1)):
        #     frame = dict()
        #     trans_1 = transl_p1[t]
        #     frame.update('length')
        #     # seq.append(frame)
        #     print('d')
        
        # skels_p1 = Skeletons(
        #     joint_positions = joints,
        #     joint_connections= lines,
        #     gui_affine = False,
        #     color=(0.11, 0.53, 0.8, 1.0),
        #     name="Skeleton1",
        #     )
        # print('d')
    print('d')

    


if __name__ == '__main__':
    main()

