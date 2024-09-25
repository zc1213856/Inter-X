'''
@File    :   plot_utils.py
@Time    :   2023/09/21 21:32:18
@Author  :   Chen Zhu 
@Version :   1.0
@Contact :   zc1213856@163.com
'''

import os
import shutil
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def draw_skeletons(skeleton_list, color_list=None, bias=[1.0, 0, 0], frame_id=0, savepath=None, prefix= 'visual', label='class'):
    # skeleton = [j1[x,y,z],j2[], ...]
    # skeleton_list = [s1, s2, ...]
    # color_list = ['r', 'g', 'b', ...]
    if color_list is None:
        color_list = ['r']*len(skeleton_list)
    # fig = plt.figure()
    # fig = plt.figure(figsize = (12,8))
    # ax = fig.add_subplot(1,1,1,projection = "3d")

    fig, ax = plt.subplots(nrows=1, ncols=1,subplot_kw=dict(projection="3d"))
    plt.ioff()
    # fig, axs = plt.subplots(nrows=1, ncols=2,subplot_kw=dict(projection="3d"))
    fig.set_size_inches(18.5, 10.5)
    # fig.suptitle(label,fontsize=24)
    for i, skeleton in enumerate(skeleton_list):
        draw_single_skeleton(skeleton,ax,color_list[i], bias)
    
    ax.axis('off')

    #for sbu
    ax.set_xlim3d([-0, 2])
    ax.set_ylim3d([-2, 2])
    ax.set_zlim3d([0, 1])
    #ax.set_title('generated')
        
    # 绘制X坐标轴
    ax.plot([0, 1], [0, 0], [0, 0], 'r', linewidth=2)
    ax.text(1.1, 0, 0, 'X', color='red')

    # 绘制Y坐标轴
    ax.plot([0, 0], [0, 1], [0, 0], 'g', linewidth=2)
    ax.text(0, 1.1, 0, 'Y', color='green')

    # 绘制Z坐标轴
    ax.plot([0, 0], [0, 0], [0, 1], 'b', linewidth=2)
    ax.text(0, 0, 1.1, 'Z', color='blue')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(-90,90)
    # ax.view_init(180,90) # for SBU  NO!
    plt.axis('off')

    ax.text2D(0.5, 0.8, label, ha='center', va='bottom', transform=ax.transAxes, fontsize=15)

    if savepath is None:
        # plt.show()
        # plt.close('all')
        return ax 
    else:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        name =savepath + '//' + prefix +'_' + str(frame_id).zfill(5) + '.png'
        plt.savefig(name)
        plt.close('all')
        return imageio.imread(name)
        # if np.sum(action[:,i])==45.0 and i>1:
        #     break

def draw_single_skeleton(skeleton_, ax, color='r', bias=[0, 0, 0]):
    joint_num = len(skeleton_)
    skeleton =  skeleton_ + np.array([bias] * joint_num)
    if joint_num==15:# sbu
        # all joints
        ax.plot(skeleton[:,0], skeleton[:,1], skeleton[:,2], color + 'o')
        # [0, 1, 2] for head and body
        ax.plot(skeleton[:3,0], skeleton[:3, 1],skeleton[:3, 2], color + '-')
        # [1, 3, 4, 5] and [1, 6, 7, 8] for arms
        ax.plot(np.append(skeleton[1, 0], skeleton[3:6, 0]), np.append(skeleton[1,1], skeleton[3:6,1]), np.append(skeleton[1,2], skeleton[3:6,2]), color + '-', alpha=0.3)
        ax.plot(np.append(skeleton[1, 0], skeleton[6:9, 0]),np.append(skeleton[1, 1], skeleton[6:9, 1]), np.append(skeleton[1, 2], skeleton[6:9,2]), color + '-')
        # [2, 9, 10, 11] et [2, 12, 13, 14] for legs
        ax.plot(np.append(skeleton[2, 0], skeleton[9:12, 0]), np.append(skeleton[2, 1], skeleton[9:12, 1]), np.append(skeleton[2, 2], skeleton[9:12, 2]), color + '-', alpha=0.3)
        ax.plot(np.append(skeleton[2, 0], skeleton[12:15, 0]), np.append(skeleton[2, 1], skeleton[12:15, 1]), np.append(skeleton[2, 2], skeleton[12:15, 2]), color + '-')
    elif joint_num == 25:# ntu
        # trunk_joints = [0, 1, 20, 2, 3]
        # arm_joints = [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
        # leg_joints = [19, 18, 17, 16, 0, 12, 13, 14, 15]
        # body = [trunk_joints, arm_joints, leg_joints]
        # j_ind=[2, 21, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 23, 8, 25, 12]
        j_ind=[1, 20, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 20, 22, 7, 24, 11]
        ax.plot(skeleton[:,0], skeleton[:,1], skeleton[:,2], color + 'o')
        for i, j in enumerate(j_ind):
            ax.plot([skeleton[i, 0],skeleton[j, 0]] ,[skeleton[i, 1],skeleton[j, 1]],[skeleton[i, 2],skeleton[j, 2]], color + '-')

def draw_skeletons_compare(skeletons_list, color_list=None, bias=[1.0, 0, 0], frame_id=0, savepath=None, prefix= 'visual', label=None):
    # skeleton = [j1[x,y,z],j2[], ...]
    # skeleton_list = [s1, s2, ...]
    # skeletons_list = [s_list_1, s_list_2]
    # color_list = ['r', 'g', 'b', ...]
    if color_list is None:
        color_list = ['r']*len(skeletons_list[0])
    # fig = plt.figure()
    # fig = plt.figure(figsize = (12,8))
    # ax = fig.add_subplot(1,1,1,projection = "3d")
    if label is None:
        label = ['compare']*len(skeletons_list)
    fig, axs = plt.subplots(nrows=1, ncols=2,subplot_kw=dict(projection="3d"))
    plt.ioff()
    # fig, axs = plt.subplots(nrows=1, ncols=2,subplot_kw=dict(projection="3d"))
    fig.set_size_inches(18.5, 10.5)
    # fig.suptitle(label,fontsize=24)
    for ax_id, ax in enumerate(axs):
        for i, skeleton in enumerate(skeletons_list[ax_id]):
            draw_single_skeleton(skeleton,ax,color_list[i], bias)
        
        ax.axis('off')

        #for sbu
        ax.set_xlim3d([-0, 2])
        ax.set_ylim3d([-2, 2])
        ax.set_zlim3d([0, 1])
        #ax.set_title('generated')
            
        # 绘制X坐标轴
        ax.plot([0, 1], [0, 0], [0, 0], 'r', linewidth=2)
        ax.text(1.1, 0, 0, 'X', color='red')

        # 绘制Y坐标轴
        ax.plot([0, 0], [0, 1], [0, 0], 'g', linewidth=2)
        ax.text(0, 1.1, 0, 'Y', color='green')

        # 绘制Z坐标轴
        ax.plot([0, 0], [0, 0], [0, 1], 'b', linewidth=2)
        ax.text(0, 0, 1.1, 'Z', color='blue')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(-90,90)
        # ax.view_init(180,90) # for SBU  NO!
        plt.axis('off')

        ax.text2D(0.5, 0.8, label[ax_id], ha='center', va='bottom', transform=ax.transAxes, fontsize=15)

    if savepath is None:
        plt.show()
        # plt.close('all')
        return 0 
    else:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        name =savepath + '//' + prefix +'_' + str(frame_id).zfill(5) + '.png'
        plt.savefig(name)
        plt.close('all')
        return imageio.imread(name)
        # if np.sum(action[:,i])==45.0 and i>1:
        #     break

def build_gif(images_dir, savepath):
    # build gif and delete image tmp path
    # images_dir : $output$/images_tmp
    # savepath : $output$/xxx.gif
    images = []
    for item in os.listdir(images_dir):
        fullpath = os.path.join(images_dir, item)
        images.append(imageio.imread(fullpath))    
    imageio.mimsave(savepath, images)
    shutil.rmtree(images_dir)
    plt.close('all')

def visulise_seq(skeleton_list_seq, savepath, label):
    # skeleton = [j1[x,y,z], j2[x,y,z], ...]
    # skeleton_list = [s_frame1, s_frame2, ...]
    # skeleton_list_seq = [s_list_a, s_list_b]
    # savepath = $save_dir$\\00000.gif
    images = []
    savedir = os.path.dirname(savepath)
    temppath = os.path.join(savedir, 'temp')
    
    for i, (skela, skelb) in enumerate(zip(skeleton_list_seq[0], skeleton_list_seq[1])):
        image = draw_skeletons([skela, skelb], color_list = ['r', 'b'], frame_id= i, label = label, savepath = temppath)
        images.append(image)
    shutil.rmtree(temppath)     
    imageio.mimsave(savepath, images)

def visulise_seqs(skeletons_seq, savepath, label):
    # skeleton = [j1[x,y,z],j2[], ...]
    # skeleton_list = [s1, s2, ...]
    # skeletons_seq = [s_list_1, s_list_2, s_list_3, ...]
    images = []
    for i, skels in enumerate(skeletons_seq):
        image = draw_skeletons(skels, color_list = ['r', 'b'], label=label)
        images.append(image)     
    imageio.mimsave(savepath, images)






if __name__ == '__main__':
    datapath = '\\\\Seuvcl-7018\\g\\zc_data\\CVPR2024-Reaction\\NTU26\\annot\\train_0_9.pkl'
    # datapath = '\\\\Seuvcl-7018\\g\\zc_data\\CVPR2024-Reaction\\SBU\\annot\\train.pkl'
    import joblib
    x = joblib.load(datapath)
    for seq in x:
        for f_id, frame in enumerate(seq):
            # seq_a = frame['0']['normalized_joints_3d']
            # seq_b = frame['1']['normalized_joints_3d']
            seq_a = frame['0']['joints_3d']
            seq_b = frame['1']['joints_3d']
            # seq_a = np.array(frame['0']['original_joints_3d'])
            # seq_b = np.array(frame['1']['original_joints_3d'])

            # draw_skeletons([seq_a,seq_b],
            #                color_list=['r','b'],
            #                frame_id=f_id,
            #             #    savepath='.\\visual'
            #                )

            draw_skeletons_compare([[seq_a,seq_b],[seq_a,seq_b]],
                           color_list=['r','b'],
                           frame_id=f_id,
                        #    savepath='.\\visual'
                        )
            print(1)

