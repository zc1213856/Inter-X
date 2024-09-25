import numpy as np

def fit_plane_core(vs):
    '''
    vs: n*3
    z = a[0]*x + a[1]*y + a[2]
    normal: {a[0], a[1], -1}
    '''
    A11 = 0
    A12 = 0
    A13 = 0
    A21 = 0
    A22 = 0
    A23 = 0
    A31 = 0
    A32 = 0
    A33 = 0
    S0 = 0
    S1 = 0
    S2 = 0
    if isinstance(vs, list):
        vs = np.array(vs)
    for i in range(vs.shape[0]):
        A11 += (vs[i][0]*vs[i][0])
        A12 += (vs[i][0]*vs[i][1])
        A13 += (vs[i][0])
        A21 += (vs[i][0]*vs[i][1])
        A22 += (vs[i][1]*vs[i][1])
        A23 += (vs[i][1])
        A31 += (vs[i][0])
        A32 += (vs[i][1])
        A33 += (1)
        S0 += (vs[i][0]*vs[i][2])
        S1 += (vs[i][1]*vs[i][2])
        S2 += (vs[i][2])

    a = np.linalg.solve([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]],[S0,S1,S2])
    return a, np.array([a[0],a[1],-1])

def fit_plane(vs,flag='z'):
    '''
    flag define the plane function
    flag=z: z = a[0] * x + a[1] * y + a[2]
    flag=y: y = a[0] * x + a[1] * z + a[2]
    flag=x: x = a[0] * y + a[1] * z + a[2]
    '''
    if flag == 'z':
        dimtrans = [0,1,2]
    elif flag == 'y':
        dimtrans = [0,2,1]
    elif flag == 'x':
        dimtrans = [1,2,0]
    vs_new = vs[:,dimtrans]
    a,normal = fit_plane_core(vs_new)
    if flag == 'z':
        normal = np.array(
            [normal[0],normal[1],-1])
    elif flag == 'y':
        normal = np.array(
            [normal[0],-1,normal[1]])
    elif flag == 'x':
        normal = np.array(
            [-1,normal[0],normal[1]])
    return a,normal

def update_smpl_rotation():
    pass

def update_smpl_translation():
    pass

def update_camera_extrinsics():
    pass

def demo():
    from scipy.spatial.transform import Rotation as R
    import sys
    sys.path.append('./')
    from utils import smpl_utils
    from utils import rotate_utils
    import glob
    import os
    import pickle as pkl
    import torch
    pkl_dir = R'D:\yangyuan\Physics-Demo\pkl\douying002'
    import smplx
    model = smplx.create('./data/smplData/body_models','smpl',gender='neutral')
    '''1. 获取脚部的关节点'''
    smpl_foot_joints = []
    smpl_root_joints = []
    for pkl_path in glob.glob(os.path.join(pkl_dir,'*')):
        _,js,_ = smpl_utils.pkl2smpl(pkl_path,model,gender='neutral')
        smpl_foot_joints.append(js[10])
        smpl_foot_joints.append(js[11])
        smpl_root_joints.append(js[0])
    smpl_foot_joints = np.array(smpl_foot_joints)
    smpl_root_joints = np.array(smpl_root_joints)
    '''debug'''
    from utils import obj_utils
    mesh_data = obj_utils.MeshData()
    mesh_data.vert = smpl_foot_joints
    obj_utils.write_obj(
        'D:\yangyuan\Physics-Demo\debug\douying002.obj',mesh_data
    )
    '''debug'''
    '''2. 拟合平面法线'''
    a,normal = fit_plane(smpl_foot_joints,'y')

    '''debug'''
    xyzMax = np.max(smpl_foot_joints, axis=0)
    xyzMin = np.min(smpl_foot_joints, axis=0)
    xyz = []
    ratio = 100
    for i in range(ratio):
        for j in range(ratio):
            x = xyzMin[0] + (xyzMax[0]-xyzMin[0]) * i / ratio
            z = xyzMin[2] + (xyzMax[2]-xyzMin[2]) * j / ratio
            y = a[0] * x + a[1] * z + a[2]
            xyz.append([x,y,z])
    meshData = obj_utils.MeshData()
    meshData.vert = np.array(xyz)
    obj_utils.write_obj('D:\yangyuan\Physics-Demo\debug\douying002.obj', meshData) 
    '''debug'''

    smpl_foot_center = smpl_foot_joints.mean(0)
    smpl_root_center = smpl_root_joints.mean(0)
    up_vector = smpl_root_center-smpl_foot_center
    if (np.dot(up_vector,normal) < 0):
        normal *= -1.0
    '''3. 计算旋转矩阵和平移变换'''
    rot_matrix = rotate_utils.GetRotFromVecs(normal,np.array([0,0,1]))
    world_foot_joints = np.dot(rot_matrix,smpl_foot_center[:,None])
    translation = -world_foot_joints
    '''4. 更新smpl旋转平移参数'''
    idx = 0
    for pkl_path in glob.glob(os.path.join(pkl_dir,'*')):
        idx+=1
        with open(pkl_path,'rb') as file:
            smpl_parameters = pkl.load(file)
        output = model(
            betas = torch.tensor(smpl_parameters['person00']['betas'][None,:].astype(np.float32)),
            body_pose = torch.tensor(smpl_parameters['person00']['body_pose'][None,:].astype(np.float32)),
            global_orient = torch.tensor(np.array([[0.0,0.0,0.0]]).astype(np.float32)),
            transl = torch.tensor(np.array([[0.0,0.0,0.0]]).astype(np.float32)))
        js = output.joints.detach().cpu().numpy().squeeze()
        j0 = js[0]


        if 'pose' in smpl_parameters['person00']:
            smpl_parameters['person00']['pose'][:3] = (R.from_matrix(rot_matrix)*R.from_rotvec(smpl_parameters['person00']['pose'][:3])).as_rotvec()
        if 'global_orient' in smpl_parameters['person00']:
            smpl_parameters['person00']['global_orient'] = (R.from_matrix(rot_matrix)*R.from_rotvec(smpl_parameters['person00']['global_orient'])).as_rotvec()
        smpl_parameters['person00']['transl'] = R.from_matrix(rot_matrix).apply(j0 + smpl_parameters['person00']['transl']) - j0 + translation[:,0]
        temPath = os.path.join(
            R'D:\yangyuan\Physics-Demo\align_pkl\douying002',
            str(idx).zfill(6)+'.pkl'
        )
        os.makedirs(os.path.dirname(temPath),exist_ok=True)
        with open(temPath,'wb') as file:
            pkl.dump(smpl_parameters, file)    
        
        smpl_utils.pkl2smpl(
            temPath,model,gender='neutral',
            savePath=os.path.join(R'D:\yangyuan\Physics-Demo\align_mesh\douying002',str(idx).zfill(6)+'.obj')
        )
    '''5. 更新camera extrinsics参数'''
    origin_camera_parameters_intri,origin_camera_parameters_extri = rotate_utils.readVclCamparams(R'D:\yangyuan\Physics-Demo\camparams.txt')
    for camid,camex in enumerate(origin_camera_parameters_extri):
        camex_new = np.array(camex)
        camex_new[:3,:3] = np.dot(camex_new[:3,:3],np.linalg.inv(rot_matrix))
        camex_new[:3,3] = camex_new[:3,3] - np.dot(camex_new[:3,:3],translation)[:,0]
        origin_camera_parameters_extri[camid] = camex_new.tolist()
    rotate_utils.writeVclCamparams(
        R'D:\yangyuan\Physics-Demo\douying002.txt',
        origin_camera_parameters_intri,
        origin_camera_parameters_extri)

# demo()

import matplotlib.pyplot as plt
import numpy as np
import math

class OneEuroFilter:

    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        # Previous values.
        self.x_prev = x0
        self.dx_prev = float(dx0)
        self.t_prev = t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter_signal(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


if __name__ == '__main__':

    np.random.seed(1)

    # Parameters
    frames = 100
    start = 0
    end = 4 * np.pi
    scale = 0.5

    # The noisy signal
    t = np.linspace(start, end, frames)
    x = np.sin(t)
    x_noisy = x + np.random.normal(scale=scale, size=len(t))

    # The filtered signal
    min_cutoff = 0.15
    beta = 0.01
    x_hat = np.zeros_like(x_noisy)
    x_hat[0] = x_noisy[0]

    one_euro_filter = OneEuroFilter(t[0], x_noisy[0], min_cutoff=min_cutoff, beta=beta)
    for i in range(1, len(t)):
        x_hat[i] = one_euro_filter.filter_signal(t[i], x_noisy[i])

    plt.plot(t, x, label="sin", color='g')
    plt.plot(t, x_noisy, label='sin+random', color='b')
    plt.plot(t, x_hat, label='euro filter', color='r')
    plt.title("sin vs sin_noise vs euro_filter")
    plt.legend()
    plt.show()
    plt.savefig("./euroImg.png")
