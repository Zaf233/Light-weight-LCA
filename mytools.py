import math
import os
import random
import time
# import apex
import mathutils
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn as nn

from utils import (rotate_back, rotate_forward)


def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]

    return pcl_uv, pcl_z


def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
    cam_intrinsic = cam_calib.numpy()
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    pcl_z = pcl_z.reshape(-1, 1)
    depth_img = np.zeros((img_shape[0], img_shape[1], 1))
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
    depth_img = torch.from_numpy(depth_img.astype(np.float32))
    depth_img = depth_img.cuda()
    depth_img = depth_img.permute(2, 0, 1)
    return depth_img, pcl_uv


def projection_encoding(pc_rotated, cam_calib, img_shape, depth_max):
    pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()  # [3, N]
    cam_intrinsic = cam_calib.numpy()
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
    threshold = max(img_shape[0], img_shape[1])
    mask = (pcl_uv[:, 0] > -2 * threshold) & (pcl_uv[:, 0] < 2 * threshold) & \
           (pcl_uv[:, 1] > -2 * threshold) & (pcl_uv[:, 1] < 2 * threshold) & \
           (pcl_z > 0)

    pc_xyz = pc_rotated.T  # [N, 3]
    pc_xyz = pc_xyz[mask] / depth_max
    pcl_uv = pcl_uv[mask] / threshold  # [N, 2]
    point_data = np.concatenate([pc_xyz, pcl_uv], axis=1)  # [N, 5]

    index = np.random.choice(point_data.shape[0], 4096, replace=True)
    sample_data = torch.from_numpy(point_data[index]).float()
    sample_data = sample_data.cuda()
    sample_data = sample_data.permute(1, 0)
    return sample_data


def reprojection(pc_input, cam_calib_input, transl_pred, rot_pred, img_shape, input_size, depth_max):
    new_lidar_depth_input = []

    for idx in range(len(pc_input)):
        pc = pc_input[idx]
        cam_calib = cam_calib_input[idx]
        R1 = mathutils.Quaternion(rot_pred[idx]).to_matrix()
        R1.resize_4x4()
        T1 = mathutils.Matrix.Translation(transl_pred[idx])
        RT1 = T1 * R1
        pc_rotated = rotate_forward(pc, RT1)  # Pc` = RT^(-1) * Pc
        pc_rotated = pc_rotated[:, pc_rotated[0, :] < depth_max].clone()

        depth_img, uv = lidar_project_depth(pc_rotated, cam_calib, img_shape)  # image_shape
        depth_img /= depth_max

        new_lidar_depth_input.append(depth_img)

    new_lidar_depth_input = torch.stack(new_lidar_depth_input)
    new_lidar_depth_input = F.max_pool2d(new_lidar_depth_input, kernel_size=5, stride=1, padding=2)

    new_lidar_depth_input = F.interpolate(new_lidar_depth_input, size=input_size, mode="bilinear", align_corners=True)
    return new_lidar_depth_input


def pose_fuse(t1, q1, t2, q2):
    '''
    (q, t) = (q1, t1)*(q2, t2)
    '''
    b, _ = q1.shape
    R1 = quaternion_to_matrix(q1)
    R2 = quaternion_to_matrix(q2)
    T1 = t1.view(b, 3, 1)
    T2 = t2.view(b, 3, 1)
    R = torch.bmm(R1, R2)
    T = torch.bmm(R1, T2) + T1
    q = matrix_to_quaternion(R)
    t = T.view(b, 3)
    return t, q


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    # pyre-ignore [16]: `torch.Tensor` has no attribute `new_tensor`.
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1)))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(*batch_dim, 4)


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def pose_inverse(t, q):
    b, _ = q.shape
    R = quaternion_to_matrix(q)
    T = t.view(b, 3, 1)

    temp = [r.inverse() for r in torch.unbind(R)]
    R_inverse = torch.stack(temp)
    T_inverse = -torch.bmm(R_inverse, T)
    return T_inverse, R_inverse


def pose_inverse_quant(t, q):
    b, _ = t.shape
    T, R = pose_inverse(t, q)
    t_ = T.view(b, 3)
    q_ = matrix_to_quaternion(R)
    return t_, q_