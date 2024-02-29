"""
Original implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018
Jinwei Gu and Zhile Ren
Modified version (CMRNet) by Daniele Cattaneo
Modified version (LCCNet) by Xudong Lv
"""
import time
import math
import torch
import numpy as np
import mathutils
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mytools import pose_fuse
from utils import rotate_points, rotate_forward
from Vit import mySelfTransformer, mySelfTransformer_seq
from einops.layers.torch import Rearrange


def show_tensor(x):
    x = x[0, 0].detach().cpu().numpy().astype(np.float32)
    plt.imshow(x)
    plt.show()
    return


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class LayerNormBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(LayerNormBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.ln1 = nn.LayerNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.ln2 = nn.LayerNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = out.permute([0, 2, 3, 1])
        out = self.ln1(out)
        out = out.permute([0, 3, 1, 2])
        out = self.relu(out)

        out = self.conv2(out)
        out = out.permute([0, 2, 3, 1])
        out = self.ln2(out)
        out = out.permute([0, 3, 1, 2])

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class selfCalibNet(nn.Module):
    def __init__(self, in_channel=5):
        super(selfCalibNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channel, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.fc_transl = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        
        self.fc_rot = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, lidar):
        feat = torch.mean(self.net(lidar), dim=2, keepdim=False)
        transl = self.fc_transl(feat)
        rot = self.fc_rot(feat)
        rot = F.normalize(rot, p=2, dim=-1)
        return transl ,rot

class fuseBlock(nn.Module):
    def __init__(self, c0, c1, c2):
        super(fuseBlock, self).__init__()
        self.conv0_1 = nn.Sequential(
            nn.Conv2d(c0, c1, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv0_2 = nn.Sequential(
            nn.Conv2d(c0, c1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv1_0 = nn.Sequential(
            nn.Conv2d(c1, c0, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.conv2_0 = nn.Sequential(
            nn.Conv2d(c2, c0, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(c2, c1, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.conv0 = nn.Sequential(
            nn.Conv2d(c0, c0, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x0, x1, x2):
        '''
        x0: [128, 256]
        x1: [64, 128]
        x2: [32, 64]
        '''
        # x0
        x1_0 = self.conv1_0(F.interpolate(x1, size=[128, 256], mode='bilinear', align_corners=True))
        x2_0 = self.conv2_0(F.interpolate(x2, size=[128, 256], mode='bilinear', align_corners=True))
        x0_ = self.conv0(x0 + x1_0 + x2_0)
        # x1
        x0_1 = self.conv0_1(x0)
        x2_1 = self.conv2_1(F.interpolate(x2, size=[64, 128], mode='bilinear', align_corners=True))
        x1_ = self.conv1(x0_1 + x1 + x2_1)
        # x2
        x0_2 = self.conv0_2(x0)
        x1_2 = self.conv1_2(x1)
        x2_ = self.conv2(x0_2 + x1_2 + x2)
        return x0_, x1_, x2_

class HRExtractor(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HRExtractor, self).__init__()
        block = LayerNormBasicBlock
        # rgb conv
        self.inplanes = in_channel
        self.layer1 = self._make_layer(block, 32, 2, stride=2)
        self.layer2 = self._make_layer(block, 64, 2, stride=2)
        self.layer3 = self._make_layer(block, 128, 2, stride=2)

        self.conv1_0 = nn.Sequential(
            nn.Conv2d(32, in_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.conv2_0 = nn.Sequential(
            nn.Conv2d(64, in_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.conv3_0 = nn.Sequential(
            nn.Conv2d(128, in_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        b, c, h, w = x.shape
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)

        f1_0 = F.interpolate(self.conv1_0(f1), size=[h, w], mode='bilinear', align_corners=True)
        f2_0 = F.interpolate(self.conv2_0(f2), size=[h, w], mode='bilinear', align_corners=True)
        f3_0 = F.interpolate(self.conv3_0(f3), size=[h, w], mode='bilinear', align_corners=True)
        out = self.conv0(x + f1_0 + f2_0 + f3_0)
        return out

class iterCalibNet(nn.Module):
    def __init__(self):
        super(iterCalibNet, self).__init__()
        # resnet with leakyRELU
        #block = BasicBlock
        block = LayerNormBasicBlock
        # rgb conv
        self.conv_rgb = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # depth conv
        self.conv_depth = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # fuse conv
        self.inplanes = 32
        self.layer0 = self._make_layer(block, 32, 2, stride=2)
        self.layer1 = self._make_layer(block, 64, 2, stride=2)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)

        # decode
        self.transformer_fuse = mySelfTransformer(image_size=(8, 16), channels=512, patch_size=(1, 1), dim=16, depth=3, heads=4, 
        mlp_dim=64)

        self.mlp_transl = nn.Sequential(
            nn.LayerNorm(128*16),
            nn.Linear(128*16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.mlp_rot = nn.Sequential(
            nn.LayerNorm(128*16),
            nn.Linear(128*16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, rgb, depth):
        b, _, h, w = rgb.shape
        # extract rgb feature
        rgb0 = self.conv_rgb(rgb)
        depth0 = self.conv_depth(depth)
        # extract fuse feature
        fuse0 = torch.cat([rgb0, depth0], dim=1)
        fuse1 = self.layer0(fuse0)  # [b, 16, 128, 256]
        fuse2 = self.layer1(fuse1)  # [b, 32, 64, 128]
        fuse3 = self.layer2(fuse2)  # [b, 64, 32, 64]
        fuse4 = self.layer3(fuse3)  # [b, 128, 16, 32]
        fuse5 = self.layer4(fuse4)  # [b, 256, 8, 16]
        #encoder
        #f = self.transformer_fuse(rgb5, depth5)
        #f = f.view([b, 128*8*2])
        #f = self.transformer_fuse(torch.cat([rgb5, depth5], dim=1))
        f = self.transformer_fuse(fuse5)
        f = f.view([b, 128*16])

        transl = self.mlp_transl(f)
        rot = self.mlp_rot(f)
        rot = F.normalize(rot, dim=1)
        return transl, rot

class refineNet(nn.Module):
    def __init__(self):
        super(refineNet, self).__init__()
        # rgb conv
        self.conv_rgb = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # depth conv
        self.conv_depth = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # fuse feature
        block = BasicBlock
        self.inplanes = 32
        self.layer0 = self._make_layer(block, 32, 2, stride=2)
        self.layer1 = self._make_layer(block, 64, 2, stride=2)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)
        # transformer
        self.transformer = mySelfTransformer(image_size=(8, 16), channels=512, patch_size=(1, 1), dim=16, depth=4, heads=8, 
        mlp_dim=64)
        self.mlp_transl = nn.Sequential(
            nn.LayerNorm(128*16),
            nn.Linear(128*16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.mlp_rot = nn.Sequential(
            nn.LayerNorm(128*16),
            nn.Linear(128*16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, rgb, depth):
        b, _, h, w = rgb.shape
        rgb0 = self.conv_rgb(rgb)
        depth0 = self.conv_depth(depth)
        x = torch.cat([rgb0, depth0], dim=1)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.transformer(x).view([b, 128*16])
        transl = self.mlp_transl(x)
        rot = self.mlp_rot(x)
        rot = F.normalize(rot, dim=1)
        return transl, rot

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        # rgb conv
        self.conv_rgb = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # depth conv
        self.conv_depth = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # fuse feature
        block = BasicBlock
        self.inplanes = 32
        self.layer0 = self._make_layer(block, 32, 2, stride=2)
        self.layer1 = self._make_layer(block, 64, 2, stride=2)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, rgb, depth):
        rgb0 = self.conv_rgb(rgb)
        depth0 = self.conv_depth(depth)
        x = torch.cat([rgb0, depth0], dim=1)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        # transformer
        self.transformer = mySelfTransformer(image_size=(8, 16), channels=512, patch_size=(1, 1), dim=16, depth=4, heads=8, 
        mlp_dim=64)
        self.mlp_transl = nn.Sequential(
            nn.LayerNorm(128*16),
            nn.Linear(128*16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.mlp_rot = nn.Sequential(
            nn.LayerNorm(128*16),
            nn.Linear(128*16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.transformer(x).view([b, 128*16])
        transl = self.mlp_transl(x)
        rot = self.mlp_rot(x)
        rot = F.normalize(rot, dim=1)
        return transl, rot

class LCANet(nn.Module):
    def __init__(self, img_shape=[384, 1280], depth_max=80.0, sample_num=4096, repeat=3):
        super(LCANet, self).__init__()
        self.img_shape = img_shape
        self.depth_max = depth_max
        self.sample_num = sample_num
        self.repeat = repeat
        # selfNet
        self.selfnet = selfCalibNet()
        # iterNet
        #self.iternet = iterCalibNet()
        self.iternet = refineNet()

    def get_2D_lidar_projection(self, pcl, cam_intrinsic):
        pcl_xyz = cam_intrinsic @ pcl.T
        pcl_xyz = pcl_xyz.T
        pcl_z = pcl_xyz[:, 2]
        pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
        pcl_uv = pcl_xyz[:, :2]
        return pcl_uv, pcl_z
    
    def record_lidar_proj(self, pc_rotated, calib):
        pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()  # [3, N]
        cam_intrinsic = calib.detach().cpu().numpy()
        pcl_uv, pcl_z = self.get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
        pcl_xyz = pc_rotated.T

        threshold = 2*max(self.img_shape[0], self.img_shape[1])
        mask = (pcl_uv[:, 0] > -threshold) & (pcl_uv[:, 0] < threshold) & (pcl_uv[:, 1] > -threshold) & (pcl_uv[:, 1] < threshold) & (pcl_z > 0)
        pcl_uv = pcl_uv[mask]
        pcl_xyz = pcl_xyz[mask]

        pcl_uv = pcl_uv[0:self.sample_num]/threshold
        pcl_xyz = pcl_xyz[0:self.sample_num]/self.depth_max
        sample_data = np.concatenate([pcl_uv, pcl_xyz], axis=1)
        sample_data = torch.from_numpy(sample_data).float()
        sample_data = sample_data.permute(1, 0)
        return sample_data

    def lidar_project_depth(self, pc_rotated, cam_calib):
        pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
        cam_intrinsic = cam_calib.detach().cpu().numpy()
        pcl_uv, pcl_z = self.get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
        mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < self.img_shape[1]) & (pcl_uv[:, 1] > 0) & (pcl_uv[:, 1] < self.img_shape[0]) & (pcl_z > 0)
        # gen projection depth map
        pcl_uv = pcl_uv[mask]
        pcl_z = pcl_z[mask]
        pcl_uv = pcl_uv.astype(np.uint32)
        pcl_z = pcl_z.reshape(-1, 1)
        depth_img = np.zeros((self.img_shape[0], self.img_shape[1], 1))
        depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
        depth_img = torch.from_numpy(depth_img.astype(np.float32))
        depth_img = depth_img/self.depth_max
        depth_img = depth_img.permute(2, 0, 1)
        return depth_img

    def trans_pc(self, pc, t, q):
        R = mathutils.Quaternion(q).to_matrix()
        R.resize_4x4()
        T = mathutils.Matrix.Translation(t)
        RT = T * R
        pc_rotated = rotate_forward(pc, RT) # Pc = RT^(-1) * Pc
        return pc_rotated

    def reprojection(self, pc, t, q, cam_calib):
        pc_rotated = self.trans_pc(pc, t, q)
        depth = self.lidar_project_depth(pc_rotated, cam_calib)
        return depth

    def forward(self, rgb, pc, calib):
        pre_transls = []
        pre_rots = []
        #................................ selfNet ......................................#
        pc_proj = [self.record_lidar_proj(p, c) for (p, c) in zip(pc, calib)]
        pc_proj = torch.stack(pc_proj).to(rgb.device)
        transl_err, rot_err = self.selfnet(pc_proj)
        pre_transls.append(transl_err)
        pre_rots.append(rot_err)
        #................................ iterNet ......................................#
        for i in range(self.repeat):
            depth = [self.reprojection(p, t, q, c) for (p, t, q, c) in zip(pc, pre_transls[i], pre_rots[i], calib)]
            depth = torch.stack(depth)
            depth = F.max_pool2d(depth, kernel_size=5, stride=1, padding=2)
            depth = F.interpolate(depth, size=(rgb.shape[2], rgb.shape[3]), mode="bilinear").to(rgb.device)

            transl_err, rot_err = self.iternet(rgb, depth)
            pre_t, pre_q = pre_transls[i].clone().detach(), pre_rots[i].clone().detach()
            cur_t, cur_q = pose_fuse(pre_t, pre_q, transl_err, rot_err)
            pre_transls.append(cur_t)
            pre_rots.append(cur_q)

        return pre_transls, pre_rots

if __name__ == '__main__':
    img = torch.rand([4, 3, 256, 512])
    depth = torch.rand([4, 1, 256, 512])
    lidar = torch.rand([4, 4, 4041])
    lidar[:, 3, :] = 1
    calib = torch.rand([4, 3, 3])

    lca = LCANet()
    print('Number of LCANet parameters: {}'.format(sum([p.data.nelement() for p in lca.parameters()])))
    selfCalib = selfCalibNet()
    iterCalib = iterCalibNet()
    refineCalib = refineNet()

    #transl, rot = selfCalib(lidar)
    #transl, rot = iterCalib(img, depth)
    #transl, rot = refineCalib(img, depth)
    #print('trans: {}'.format(transl.shape))
    #print('rot: {}'.format(rot.shape))
    #print('att: {}'.format(att.shape))
    '''
    
    time_count1 = 0.0
    time_count2 = 0.0

    img = img.cuda()
    lidar = lidar.cuda()
    gen = gen.cuda()
    lcc = lcc.cuda()
    
    for i in range(1000):
        t0 = time.time()
        fake_depth = gen(img)
        t1 = time.time()
        trans_, rot, _, _ = lcc(fake_depth, lidar)
        t2 = time.time()
        time_count1 += (t1-t0)
        time_count2 += (t2-t1)
    print('Speed1: {}ms/frame, Speed2: {}ms/frame'.format(time_count1, time_count2))
    '''

