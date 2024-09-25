import math
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R
import numpy as np

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

def rotation_6d_to_rotation_matrix(x):
    """Convert 6D rotation representation to 3x3 rotation matrix. Based on Zhou et al., "On the Continuity of
    Rotation Representations in Neural Networks", CVPR 2019
    >>> x_test = torch.rand((4, 8, 6))
    >>> y_test = rotation_6d_to_rotation_matrix(x_test)
    >>> assert y_test.shape == (4, 8, 3, 3)
    Args:
        x: (B,N,6) Batch of 6-D rotation representations
    Returns:
        (B,N,3,3) Batch of corresponding rotation matrices
    """
    x_shape = x.shape
    x = x.view(-1, 3, 2)

    a1 = x[..., 0]
    a2 = x[..., 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)

    rotmat = torch.stack((b1, b2, b3), dim=-1)
    return rotmat.view(*x_shape[:-1], 3, 3)

def rotation_6d_to_axis_angle(x: torch.Tensor) -> torch.Tensor:
    """Convert 6d rotation representation to axis angle (3D) representation.
    https://stackoverflow.com/questions/12463487/obtain-rotation-axis-from-rotation-matrix-and-translation-vector-in-opencv
    Args:
        x: 6d rotation tensor (..., 6)
    """
    assert x.shape[-1] == 6
    r = rotation_6d_to_rotation_matrix(x)

    angle = torch.arccos((r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2] - 1)/2)
    yz = (r[..., 2, 1] - r[..., 1, 2])**2
    xz = (r[..., 0, 2] - r[..., 2, 0])**2
    xy = (r[..., 1, 0] - r[..., 0, 1])**2
    norm = torch.sqrt(xy + xz + yz)

    ax = (r[..., 2, 1] - r[..., 1, 2]) / norm * angle
    ay = (r[..., 0, 2] - r[..., 2, 0]) / norm * angle
    az = (r[..., 1, 0] - r[..., 0, 1]) / norm * angle
    return torch.stack([ax, ay, az], dim=-1)

def matrix_to_rotation_6d(x: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to 6d representation.
    from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    """
    batch_dim = x.size()[:-2]
    return x[..., :2].clone().reshape(batch_dim + (6,))

def angle_axis_to_rotation_6d(x: torch.Tensor) -> torch.Tensor:
    """Convert rotation in axis-angle representation to 6d representation."""
    shape = x.shape[:-1]
    x_flat = torch.flatten(x, end_dim=-2)
    # y = kornia.geometry.angle_axis_to_rotation_matrix(x_flat)
    y = torch.tensor(R.from_rotvec(x.view(-1,3)).as_matrix())
    y6d = matrix_to_rotation_6d(y.view(x.shape[0],x.shape[1],3,3))
    return y6d.view(*shape, 6)

class Adjust:
    def __init__(self,pose,trans,frequence,smpl, t=0):
        pose_6d = angle_axis_to_rotation_6d(pose.view(1,-1,3))
        self.one_euro_filter_pose = OneEuroFilter(
            0, pose_6d.view(-1), 
            min_cutoff=0.004, beta=0.8)
        self.one_euro_filter_trans = OneEuroFilter(
            0, trans[0].detach().cpu().numpy(), 
            min_cutoff=0.15, beta=0.02)
        self.frequence = frequence
        self.smpl = smpl
        self.t = t

    def __call__(self,pose,trans):
        self.t += (1.0/self.frequence)

        pose_6d = angle_axis_to_rotation_6d(pose.view(1,-1,3))
        pose_hat = self.one_euro_filter_pose.filter_signal(self.t, pose_6d.view(-1))
        pose = rotation_6d_to_axis_angle(torch.tensor(pose_hat[None,:]).view(1,-1,6))
        pose = torch.tensor(pose,dtype=torch.float32).view(1,-1)
        # pose = rotation_6d_to_axis_angle(pose_hat[None,:].clone().detach().view(1,-1,6))
        # pose = pose.clone().detach().view(1,-1)

        trans_hat = self.one_euro_filter_trans.filter_signal(self.t, trans[0].detach().cpu().numpy())
        trans = torch.tensor(trans_hat[None,:])

        _,joints = self.smpl(
            betas = torch.tensor(np.zeros(10)[None,:].astype(np.float32)),
            thetas = pose,
            trans = trans)
        js = joints.detach().cpu().numpy().squeeze()
        foot_joint_l_z = js[10][2]
        foot_joint_r_z = js[11][2]
        if (foot_joint_l_z > foot_joint_r_z):
            joint_low = foot_joint_r_z
        else:
            joint_low = foot_joint_l_z
        if (joint_low<0):
            trans[0][2] -= (torch.tensor(joint_low)*2.0)
        
        return pose,trans