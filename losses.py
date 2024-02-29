# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

# Modified Author: Xudong Lv
# based on github.com/cattaneod/CMRNet/blob/master/losses.pyy

import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from quaternion_distances import quaternion_distance
from utils import quat2mat, rotate_back, rotate_forward, tvector2mat, quaternion_from_matrix


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
    '''
    (q, t) = (q1, t1)*(q2, t2)
    '''
    b, _ = q.shape
    R = quaternion_to_matrix(q)
    T = t.view(b, 3, 1)

    temp = [r.inverse() for r in torch.unbind(R)]
    R_inverse = torch.stack(temp)
    T_inverse = -torch.bmm(R_inverse, T)
    return T_inverse, R_inverse

class GeometricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sx = torch.nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.sq = torch.nn.Parameter(torch.Tensor([-3.0]), requires_grad=True)
        self.transl_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        total_loss = torch.exp(-self.sx) * loss_transl + self.sx
        total_loss += torch.exp(-self.sq) * loss_rot + self.sq
        return total_loss

class ProposedLoss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot):
        super(ProposedLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.losses = {}

    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = 0.
        if self.rescale_trans != 0.:
            loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean() * 100
        loss_rot = 0.
        if self.rescale_rot != 0.:
            loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        total_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot
        self.losses['total_loss'] = total_loss
        self.losses['transl_loss'] = loss_transl
        self.losses['rot_loss'] = loss_rot
        return self.losses

class L1Loss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot):
        super(L1Loss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = self.transl_loss(rot_err, target_rot).sum(1).mean()
        total_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot
        return total_loss

class DistancePoints3D(nn.Module):
    def __init__(self):
        super(DistancePoints3D, self).__init__()

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err):
        """
        Points Distance Error
        Args:
            point_cloud: list of B Point Clouds, each in the relative GT frame
            transl_err: network estimate of the translations
            rot_err: network estimate of the rotations

        Returns:
            The mean distance between 3D points
        """
        #start = time.time()
        total_loss = torch.tensor([0.0]).to(transl_err.device)
        for i in range(len(point_clouds)):
            point_cloud_gt = point_clouds[i].to(transl_err.device)
            point_cloud_out = point_clouds[i].clone()

            R_target = quat2mat(target_rot[i])
            T_target = tvector2mat(target_transl[i])
            RT_target = torch.mm(T_target, R_target)

            R_predicted = quat2mat(rot_err[i])
            T_predicted = tvector2mat(transl_err[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)

            RT_total = torch.mm(RT_target.inverse(), RT_predicted)

            point_cloud_out = rotate_forward(point_cloud_out, RT_total)

            error = (point_cloud_out - point_cloud_gt).norm(dim=0)
            topk, _ = torch.topk(error, 128)
            topk.clamp(100.)
            total_loss += topk.mean()

        #end = time.time()
        #print("3D Distance Time: ", end-start)

        return total_loss/target_transl.shape[0]

# The combination of L1 loss of translation part,
# quaternion angle loss of rotation part,
# distance loss of the pointclouds
class CombinedLoss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot, weight_point_cloud):
        super(CombinedLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none', beta=0.05)
        self.weight_point_cloud = weight_point_cloud
        self.loss = {}

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err):
        """
        The Combination of Pose Error and Points Distance Error
        Args:
            point_clouds: list of B Point Clouds, each in the relative GT frame
            target_transl: groundtruth of the translations
            target_rot: groundtruth of the rotations
            transl_err: network estimate of the translations
            rot_err: network estimate of the rotations

        Returns:
            The combination loss of Pose error and the mean distance between 3D points
        """
        loss_transl = 0.
        if self.rescale_trans != 0.:
            loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = 0.
        if self.rescale_rot != 0.:
            loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        pose_loss = self.rescale_trans * loss_transl + self.rescale_rot * loss_rot

        # start = time.time()
        point_clouds_loss = torch.tensor([0.0]).to(transl_err.device)
        for i in range(len(point_clouds)):
            point_cloud_gt = point_clouds[i].to(transl_err.device)
            point_cloud_out = point_clouds[i].clone()

            R_target = quat2mat(target_rot[i])
            T_target = tvector2mat(target_transl[i])
            RT_target = torch.mm(T_target, R_target)

            R_predicted = quat2mat(rot_err[i])
            T_predicted = tvector2mat(transl_err[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)

            RT_total = torch.mm(RT_target.inverse(), RT_predicted)

            point_cloud_out = rotate_forward(point_cloud_out, RT_total)

            error = (point_cloud_out - point_cloud_gt).norm(dim=0)
            error.clamp(100.)
            point_clouds_loss += error.mean()

        # end = time.time()
        # print("3D Distance Time: ", end-start)
        total_loss = (1 - self.weight_point_cloud) * pose_loss + \
                     self.weight_point_cloud * (point_clouds_loss / target_transl.shape[0])

        self.loss['total_loss'] = total_loss
        self.loss['transl_loss'] = loss_transl
        self.loss['rot_loss'] = loss_rot
        self.loss['point_clouds_loss'] = point_clouds_loss / target_transl.shape[0]
        return self.loss

class FlowLoss(nn.Module):
    def __init__(self):
        super(FlowLoss, self).__init__()

    def forward(self, pred, label):
        label = label.permute([0, 3, 1, 2])
        #B, C, H, W = label.size()
        #label_x = label[:, 0:1, :, :]/W
        #label_y = label[:, 1:2, :, :]/H
        #label_norm = torch.cat([label_x, label_y], dim=1)
        # get mask
        mask_valid = (label > 0)  # [B, 2, h, w]
        mask_valid = torch.min(mask_valid.float(), dim=1, keepdim=True)[0]  # [B, 1, h, w]
        # gen flow
        #x = torch.arange(0, W).view([1, 1, 1, W]).repeat([B, 1, H, 1])
        #y = torch.arange(0, H).view([1, 1, H, 1]).repeat([B, 1, 1, W])
        #grid = torch.cat([x, y], dim=1).to(label.device)
        #flow = label - grid
        # cal loss
        #new_disp = torch.abs(label - pred)  # [B, 2, H, W]
        #diff = torch.relu(new_disp - disp + margin)  # [B, 2, H, W]
        diff = torch.abs(pred - label)
        diff = torch.mean(diff, dim=1, keepdim=True)  # [B, 1, H, W]
        total_loss = torch.sum(mask_valid*diff)/(torch.sum(mask_valid) + 1e-6)
        return total_loss

class EulerLoss(nn.Module):
    def __init__(self):
        super(EulerLoss, self).__init__()
        self.smoothl1_loss = nn.SmoothL1Loss(reduction='none', beta=0.01)
        self.loss = {}

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err):
        b, _ = target_transl.shape

        target_t = target_transl.view([b, 3, 1])
        target_r = quaternion_to_matrix(target_rot)
        target_inv_t, target_inv_r = pose_inverse(target_transl, target_rot)

        pred_t = transl_err.view([b, 3, 1])
        pred_r = quaternion_to_matrix(rot_err)
        # loss r
        ones_matrix = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]).repeat([b, 1]).to(rot_err.device)
        r_mul = torch.bmm(target_inv_r, pred_r).view([b, 9])
        loss_r = self.smoothl1_loss(r_mul, ones_matrix).sum(1).mean()
        #loss_r = self.smoothl1_loss(pred_r, target_r).view([b, 9]).sum(1).mean()

        # loss t
        zeros_matrix = torch.tensor([[0.0, 0.0, 0.0]]).repeat([b, 1]).to(transl_err.device)
        t_mul = (torch.bmm(pred_r, target_inv_t) + pred_t).view([b, 3])
        loss_t = self.smoothl1_loss(t_mul, zeros_matrix).sum(1).mean()
        #loss_t = self.smoothl1_loss(pred_t, target_t).view([b, 3]).sum(1).mean()

        total_loss = loss_r + loss_t

        self.loss['total_loss'] = total_loss
        self.loss['transl_loss'] = loss_t
        self.loss['rot_loss'] = loss_r
        return self.loss
