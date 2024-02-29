# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

# Modified Author: Xudong Lv
# based on github.com/cattaneod/CMRNet/blob/master/main_visibility_CALIB.py

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
import torch.utils.data
import torch.nn as nn

from DatasetLidarCamera import DatasetLidarCameraKittiOdometry
from losses import DistancePoints3D, GeometricLoss, L1Loss, ProposedLoss, CombinedLoss
from LCCNet import LCCNet
from generator import Generator

from quaternion_distances import quaternion_distance

from utils import (merge_inputs, overlay_imgs, quat2mat,
                   rotate_back, rotate_forward,
                   tvector2mat)
import matplotlib.pyplot as plt
import skimage
import cv2

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

os.environ['CUDA_VISIBLE_DEVICES'] = '6'


EPOCH = 1


class MaskL1Loss(nn.Module):
    def __init__(self):
        super(MaskL1Loss, self).__init__()

    def forward(self, target, pred):
        mask_valid = (target > 0.0)
        diff = torch.abs(pred - target)
        loss = torch.mean(diff[mask_valid])
        return loss


def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH*100
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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


def show_tensor(x, title):
    x = x.permute([1, 2, 0])
    plt.imshow(x.cpu().detach().numpy(), cmap='gray')
    plt.axis('off')
    #plt.title(title)
    plt.show()


def show_att(att):
    att_data = att.permute([0, 2, 3, 1])
    data = att_data[0].detach().cpu().numpy()
    plt.imshow(data)
    plt.axis('off')
    # plt.title(title)
    plt.show()

def show_att_rgb(att, rgb, refl_img, idx):
    b, c, h, w = rgb.shape
    data_att = att.view([1, 1, 8, 16])
    data_att = F.interpolate(data_att, size=[h, w], mode='bilinear')
    data_rgb = convert_rgb(rgb)
    # gen heatmap
    data_att = data_att.permute([0, 2, 3, 1])
    att_numpy = data_att[0].detach().cpu().numpy()
    att_numpy = att_numpy/att_numpy.max()
    heatmap = np.uint8(255 * att_numpy)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
    # show heatmap and rgb image
    data_rgb = data_rgb.permute([0, 2, 3, 1])
    rgb_numpy = data_rgb[0].detach().cpu().numpy()
    rgb_img = np.uint8(255 * rgb_numpy)
    #plt.imshow(rgb_img)
    #plt.axis('off')
    #plt.show()
    #dest = cv2.addWeighted(heatmap, 0.2, rgb_img, 0.8, 0)
    #plt.imshow(dest)
    #plt.axis('off')
    # plt.title(title)
    #plt.show()
    # show heatmap and depth
    data_depth = refl_img.repeat(1, 3, 1, 1).permute([0, 2, 3, 1])
    depth_numpy = data_depth[0].detach().cpu().numpy()
    depth_img = np.uint8(255 * depth_numpy)
    #plt.imshow(depth_img)
    #plt.axis('off')
    #plt.show()
    dest_ = cv2.addWeighted(heatmap, 0.3, depth_img, 1.0, 0)
    #cv2.imwrite('sample/RGB_{}.png'.format(idx), rgb_img[...,::-1])
    #cv2.imwrite('sample/Depth_{}.png'.format(idx), depth_img[...,::-1])
    #cv2.imwrite('sample/Att_{}.png'.format(idx), dest_[...,::-1])
    plt.imshow(dest_)
    plt.axis('off')
    plt.show()

def convert_rgb(rgb):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    rgb = rgb.clone().cpu().detach().permute(0, 2, 3, 1).numpy()
    rgb = rgb*std+mean
    rgb = torch.from_numpy(rgb).permute(0, 3, 1, 2)
    #rgb = F.interpolate(rgb, size=(256, 512), mode="bilinear", align_corners=True)
    return rgb

def show_rgb(rgb):
    plt.imshow(rgb.permute(1, 2, 0).numpy())
    plt.axis('off')
    #plt.title('RGB')
    plt.show()


# CNN test
def val(gen, disc, rgb, refl_img, target_transl, target_rot, loss_fn, point_clouds, loss):
    gen.eval()
    disc.eval()
    # Run model
    with torch.no_grad():
        fake_depth = gen(rgb)
        transl_err, rot_err, att, _, _ = disc(fake_depth, refl_img)
        transl_err_ = torch.sum(transl_err * att, dim=1, keepdim=False)
        rot_err_ = torch.sum(rot_err * att, dim=1, keepdim=False)
        rot_err_ = F.normalize(rot_err_, dim=-1)

    if loss == 'points_distance' or loss == 'combined':
        losses = loss_fn(point_clouds, target_transl, target_rot, transl_err_, rot_err_)
    else:
        losses = loss_fn(target_transl, target_rot, transl_err_, rot_err_)

    total_trasl_error = torch.tensor(0.0).to(target_transl.device)
    total_rot_error = quaternion_distance(target_rot, rot_err_, target_rot.device)
    total_rot_error = total_rot_error * 180. / math.pi
    for j in range(rgb.shape[0]):
        total_trasl_error += torch.norm(target_transl[j] - transl_err_[j]) * 100.

    return losses, total_trasl_error.item(), total_rot_error.sum().item(), rot_err_, transl_err_, fake_depth, att


def cal_ssim(im1, im2):
    im1 = im1.cpu().detach().numpy()
    im2 = im2.cpu().detach().numpy()
    ssim = skimage.measure.compare_ssim(im1, im2, data_range=1)
    return ssim

if __name__ == '__main__':
    img_shape = (384, 1280)
    input_size = (256, 512)
    _config = {'checkpoints': 'checkpoints/',
               'dataset': 'kitti/odom0',
               'data_folder': '/data1/zhuangfan/kitti/odometry/dataset',
               'use_reflectance': False,
               'val_sequence': 0,
               'loss': 'combined',
               'max_t': 1.5, 'max_r': 20.0,
               'batch_size': 1, 'num_worker': 4,
               'network': 'Res_f1',
               'weights_gen': 'checkpoints/kitti/odom0/val_seq_00/models_gen/checkpoint_r20.00_t1.50_e118_3.898.tar',
               'weights_disc': 'checkpoints/kitti/odom0/val_seq_00/models_disc/checkpoint_r20.00_t1.50_e118_3.898.tar',
               'rescale_rot': 1.0, 'rescale_transl': 2.0,
               'precision': "O0",
               'norm': 'bn', 'dropout': 0.0,
               'max_depth': 80.,
               'weight_point_cloud': 0.5
               }
    _config["checkpoints"] = os.path.join(_config["checkpoints"], _config['dataset'])
    if _config['val_sequence'] is None:
        raise TypeError('val_sequences cannot be None')
    else:
        _config['val_sequence'] = f"{_config['val_sequence']:02d}"
        print("Val Sequence: ", _config['val_sequence'])
        dataset_class = DatasetLidarCameraKittiOdometry
    dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                split='val', use_reflectance=_config['use_reflectance'], guass_kernel=0,
                                val_sequence=_config['val_sequence'])

    seed = 0
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    def init_fn(x): return _init_fn(x, seed)

    val_dataset_size = len(dataset_val)
    print('Number of the val dataset: {}'.format(val_dataset_size))

    # Training and validation set creation
    num_worker = _config['num_worker']
    batch_size = _config['batch_size']
    ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=True)
    print(len(ValImgLoader))
    # loss function choice
    if _config['loss'] == 'simple':
        loss_fn = ProposedLoss(_config['rescale_transl'], _config['rescale_rot'])
    elif _config['loss'] == 'geometric':
        loss_fn = GeometricLoss()
        loss_fn = loss_fn.cuda()
    elif _config['loss'] == 'points_distance':
        loss_fn = DistancePoints3D()
    elif _config['loss'] == 'L1':
        loss_fn = L1Loss(_config['rescale_transl'], _config['rescale_rot'])
    elif _config['loss'] == 'combined':
        loss_fn = CombinedLoss(_config['rescale_transl'], _config['rescale_rot'], _config['weight_point_cloud'])
    else:
        raise ValueError("Unknown Loss Function")
    # lcc network choice and settings
    if _config['network'].startswith('Res'):
        feat = 1
        md = 4
        split = _config['network'].split('_')
        for item in split[1:]:
            if item.startswith('f'):
                feat = int(item[-1])
            elif item.startswith('md'):
                md = int(item[2:])
        assert 0 < feat < 7, "Feature Number from PWC have to be between 1 and 6"
        assert 0 < md, "md must be positive"
        disc = LCCNet(input_size, use_feat_from=feat, md=md, use_reflectance=_config['use_reflectance'],
                      dropout=_config['dropout'], Action_Func='leakyrelu', attention=False, res_num=18)
    else:
        raise TypeError("Network unknown")
    if _config['weights_disc'] is not None:
        print(f"Loading weights from {_config['weights_disc']}")
        checkpoint = torch.load(_config['weights_disc'], map_location='cpu')
        saved_state_dict = checkpoint['state_dict']
        disc.load_state_dict(saved_state_dict)
    disc = nn.DataParallel(disc)
    disc = disc.cuda()
    print('Number of lcc parameters: {}'.format(sum([p.data.nelement() for p in disc.parameters()])))
    # gen network choice and settings
    gen = Generator(img_channels=3)
    if _config['weights_gen'] is not None:
        print(f"Loading weights from {_config['weights_gen']}")
        checkpoint = torch.load(_config['weights_gen'], map_location='cpu')
        saved_state_dict = checkpoint['state_dict']
        gen.load_state_dict(saved_state_dict)
    gen = nn.DataParallel(gen)
    gen = gen.cuda()
    print('Number of gen parameters: {}'.format(sum([p.data.nelement() for p in gen.parameters()])))
    ssim_fake_lidar_list = []
    ssim_rgb_list = []
    # start to visual
    for batch_idx, sample in enumerate(ValImgLoader):
        #print(f'batch {batch_idx+1}/{len(TrainImgLoader)}', end='\r')
        start_time = time.time()
        rgb_input = []
        lidar_depth_input = []
        lidar_depth_gt = []
        shape_pad_input = []
        real_shape_input = []
        pc_rotated_input = []

        # gt pose
        sample['tr_error'] = sample['tr_error'].cuda()
        sample['rot_error'] = sample['rot_error'].cuda()

        for idx in range(len(sample['rgb'])):
            # ProjectPointCloud in RT-pose
            real_shape = [sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2], sample['rgb'][idx].shape[0]]

            sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda()
            pc_lidar = sample['point_cloud'][idx].clone()

            if _config['max_depth'] < 80.:
                pc_lidar = pc_lidar[:, pc_lidar[0, :] < _config['max_depth']].clone()

            depth_gt, uv = lidar_project_depth(pc_lidar, sample['calib'][idx], real_shape) # image_shape
            depth_gt /= _config['max_depth']

            reflectance = None
            if _config['use_reflectance']:
                reflectance = sample['reflectance'][idx].cuda()

            R = mathutils.Quaternion(sample['rot_error'][idx]).to_matrix()
            R.resize_4x4()
            T = mathutils.Matrix.Translation(sample['tr_error'][idx])
            RT = T * R

            pc_rotated = rotate_back(sample['point_cloud'][idx], RT) # Pc` = RT * Pc

            if _config['max_depth'] < 80.:
                pc_rotated = pc_rotated[:, pc_rotated[0, :] < _config['max_depth']].clone()

            depth_img, uv = lidar_project_depth(pc_rotated, sample['calib'][idx], real_shape) # image_shape
            depth_img /= _config['max_depth']

            if _config['use_reflectance']:
                refl_img = None

            # PAD ONLY ON RIGHT AND BOTTOM SIDE
            rgb = sample['rgb'][idx].cuda()
            shape_pad = [0, 0, 0, 0]

            shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
            shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1

            rgb = F.pad(rgb, shape_pad)
            depth_img = F.pad(depth_img, shape_pad)
            depth_gt = F.pad(depth_gt, shape_pad)

            rgb_input.append(rgb)
            lidar_depth_input.append(depth_img)
            lidar_depth_gt.append(depth_gt)
            real_shape_input.append(real_shape)
            shape_pad_input.append(shape_pad)
            pc_rotated_input.append(pc_rotated)

        rgb_input = torch.stack(rgb_input)
        lidar_depth_input = torch.stack(lidar_depth_input)
        lidar_depth_gt = torch.stack(lidar_depth_gt)
        rgb_show = rgb_input.clone()
        lidar_show = lidar_depth_input.clone()
        lidar_gt_show = lidar_depth_gt.clone()

        lidar_depth_input = F.max_pool2d(lidar_depth_input, kernel_size=5, stride=1, padding=2)
        lidar_depth_gt = F.max_pool2d(lidar_depth_gt, kernel_size=5, stride=1, padding=2)

        rgb_input = F.interpolate(rgb_input, size=input_size, mode="bilinear", align_corners=True)
        lidar_depth_input = F.interpolate(lidar_depth_input, size=input_size, mode="bilinear", align_corners=True)
        lidar_depth_gt = F.interpolate(lidar_depth_gt, size=input_size, mode="bilinear", align_corners=True)
        loss, trasl_e, rot_e, R_predicted, T_predicted, fake_lidar, att = val(gen, disc, rgb_input, lidar_depth_input,
                                                                         sample['tr_error'], sample['rot_error'], loss_fn,
                                                                         sample['point_cloud'], _config['loss'])
        show_att_rgb(att, rgb_input, lidar_depth_input, batch_idx)
        # show
        show_idx = 0
        # output image: The overlay image of the input rgb image
        # and the projected lidar pointcloud depth image
        rotated_point_cloud = pc_rotated_input[show_idx]
        R_predicted = quat2mat(R_predicted[show_idx])
        T_predicted = tvector2mat(T_predicted[show_idx])
        RT_predicted = torch.mm(T_predicted, R_predicted)
        rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted)

        depth_pred, uv = lidar_project_depth(rotated_point_cloud, sample['calib'][show_idx],
                                             real_shape_input[show_idx]) # or image_shape
        depth_pred /= _config['max_depth']
        depth_pred = F.pad(depth_pred, shape_pad_input[show_idx])

        pred_show = overlay_imgs(rgb_show[show_idx], depth_pred.unsqueeze(0))
        input_show = overlay_imgs(rgb_show[show_idx], lidar_show[show_idx].unsqueeze(0))
        gt_show = overlay_imgs(rgb_show[show_idx], lidar_gt_show[show_idx].unsqueeze(0))

        pred_show = torch.from_numpy(pred_show)
        pred_show = pred_show.permute(2, 0, 1)
        input_show = torch.from_numpy(input_show)
        input_show = input_show.permute(2, 0, 1)
        gt_show = torch.from_numpy(gt_show)
        gt_show = gt_show.permute(2, 0, 1)

        #rgb = convert_rgb(rgb_show)
        #show_rgb(rgb[0])
        #show_tensor(gt_show, 'GT')
        #show_tensor(input_show, 'Input')
        #show_tensor(pred_show, 'Output')
        #show_tensor(lidar_depth_gt[0], 'Lidar GT')
        #show_tensor(lidar_depth_input[0].cpu(), 'Mis-Calib')
        #show_tensor(fake_lidar[0], 'Gen Result')
        '''
        # cal ssim between GT and fake LiDAR
        ssim_fake_lidar = cal_ssim(lidar_depth_gt[0, 0], fake_lidar[0, 0])
        # cal ssim between GT and RGB
        rgb[:, :, 0:152, :] = 0.0
        ssim_r = cal_ssim(lidar_depth_gt[0, 0], rgb[0, 0, :, :])
        ssim_g = cal_ssim(lidar_depth_gt[0, 0], rgb[0, 1, :, :])
        ssim_b = cal_ssim(lidar_depth_gt[0, 0], rgb[0, 2, :, :])
        ssim_rgb = (ssim_r + ssim_g + ssim_b)/3
        # hist
        ssim_fake_lidar_list.append(ssim_fake_lidar)
        ssim_rgb_list.append(ssim_rgb)
        if batch_idx%100 == 0:
            print('Batch_idx: {}, Fake LiDAR: {}, RGB: {}'.format(batch_idx, ssim_fake_lidar, ssim_rgb))
    ssim_fake_lidar_list = np.array(ssim_fake_lidar_list)
    ssim_rgb_list = np.array(ssim_rgb_list)
    np.save('ssim_fake_lidar', ssim_fake_lidar_list)
    np.save('ssim-rgb', ssim_rgb_list)
    plt.hist(ssim_fake_lidar_list, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.title('fake-lidar-ssim')
    #plt.show()
    f = plt.gcf()
    f.savefig('ssim_fake_lidar.png')
    plt.hist(ssim_rgb_list, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.title('rgb-ssim')
    #plt.show()
    f = plt.gcf()
    f.savefig('ssim_rgb.png')
    '''
