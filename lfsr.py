# Copyright 2024 The KaiJIN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PSW++: Improved Position-Sensitive Windowing Strategy for Light Field Image Super-Resolution
"""
import os
import glob
import random
import importlib
import tqdm
import copy
import argparse
import functools

import cv2
import h5py
import imageio
import mat73
import numpy as np
import scipy.io
from skimage import metrics

import torch
from torch.nn import functional as F
from torch import nn
import einops


PROTOCALS_4X = {
    'LFSR.ALL2': {
        'train': [
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/EPFL/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/HCI_new/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/HCI_old/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/INRIA_Lytro/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/Stanford_Gantry/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_13579/EPFL/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_13579/HCI_new/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_13579/HCI_old/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_13579/INRIA_Lytro/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_13579/Stanford_Gantry/',
        ],
        'val': [
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/EPFL/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/HCI_new/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/HCI_old/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/INRIA_Lytro/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/Stanford_Gantry/',
        ]
    },
    'LFSR.ALL': {
        'train': [
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/EPFL/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/HCI_new/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/HCI_old/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/INRIA_Lytro/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/Stanford_Gantry/',
        ],
        'val': [
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/EPFL/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/HCI_new/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/HCI_old/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/INRIA_Lytro/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/Stanford_Gantry/',
        ]
    },
    'LFSR.EPFL': {
        'train': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32/EPFL/'],
        'val': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x/EPFL/']
    },
    'LFSR.HCInew': {
        'train': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32/HCI_new/'],
        'val': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x/HCI_new/']
    },
    'LFSR.HCIold': {
        'train': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32/HCI_old/'],
        'val': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x/HCI_old/']
    },
    'LFSR.INRIA': {
        'train': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32/INRIA_Lytro/'],
        'val': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x/INRIA_Lytro/']
    },
    'LFSR.STFgantry': {
        'train': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32/Stanford_Gantry/'],
        'val': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x/Stanford_Gantry/']
    },
    'LFSR.NTIRE.VAL': {
        'test': [
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_inference/SR_5x5_4x/NTIRE_Val_Real/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_inference/SR_5x5_4x/NTIRE_Val_Synth/'
        ]
    },
    'LFSR.NTIRE.TEST': {
        'test': [
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_inference/SR_5x5_4x/NTIRE_Test_Real/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_inference/SR_5x5_4x/NTIRE_Test_Synth/'
        ]
    }
}



class LFSRDataset(torch.utils.data.Dataset):

  """Light-Field Dataset

  Data Format:
    1) h5 format:
    2) npy format:
    3) png/bmp format:

  """

  def __init__(self, phase, paths, transform, scale, transpose=True, **kwargs):
    """LFSR dataset with h5 file
    """
    self.targets = []
    self.phase = phase
    self.angular = 5
    self.scale = scale
    self.patch_size = 32
    self.transpose = transpose

    if self.phase == 'train':
      # if use .mat data, should repeat to 300
      repeat = 1
    else:
      repeat = 1

    # collect files
    files = []
    for path in paths:
      files.extend(sorted(glob.glob(f'{path}/**/*.h5', recursive=True)))
      files.extend(sorted(glob.glob(f'{path}/**/*.mat', recursive=True)))

    # preprocess
    self.cache = {}
    for path in tqdm.tqdm(files):
      if path.endswith('.mat'):
        self.cache[path] = self.load_mat(path)

    self.targets = files * repeat
    self.transform = transform
    self.count = 0

  @staticmethod
  def rgb2ycbcr(x):
    """rgb to ycbcr

    Args:
        x (np.ndarray): [H, W, 3] (0, 255) range

    Returns:
        y (np.ndarray): [H, W, 3] (16, 235) range
    """
    x = x / 255.0
    y = np.zeros(x.shape, dtype='double')
    y[:, :, 0] = 65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] + 24.966 * x[:, :, 2] + 16.0
    y[:, :, 1] = -37.797 * x[:, :, 0] - 74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:, :, 2] = 112.000 * x[:, :, 0] - 93.786 * x[:, :, 1] - 18.214 * x[:, :, 2] + 128.0
    # y = y / 255.0
    return y

  @staticmethod
  def ycbcr2rgb(x):
    """ycbcr to rgb

    Args:
        y (np.ndarray): [H, W, 3] (16, 235) range

    Returns:
        x (np.ndarray): [H, W, 3] (0, 255) range
    """
    x = x / 255.0
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255

    y = np.zeros(x.shape, dtype='double')
    y[:, :, 0] = mat_inv[0, 0] * x[:, :, 0] + mat_inv[0, 1] * x[:, :, 1] + mat_inv[0, 2] * x[:, :, 2] - offset[0]
    y[:, :, 1] = mat_inv[1, 0] * x[:, :, 0] + mat_inv[1, 1] * x[:, :, 1] + mat_inv[1, 2] * x[:, :, 2] - offset[1]
    y[:, :, 2] = mat_inv[2, 0] * x[:, :, 0] + mat_inv[2, 1] * x[:, :, 1] + mat_inv[2, 2] * x[:, :, 2] - offset[2]
    return y * 255.0

  @staticmethod
  def load_mat(path):
    try:
      LF = np.array(scipy.io.loadmat(path)['LF'])
    except BaseException:
      LF = np.array(mat73.loadmat(path)['LF'])
    return LF

  @staticmethod
  def parse_mat(mat, patch_size, scale):
    LF = np.copy(mat)  # self.cache[path])

    U, V, H, W, _ = LF.shape

    # augmentation for random select 5x5 patches
    if U == V == 9:
      # select 5 row
      select_row = [4, ]
      select_row.extend(random.sample([0, 1, 2, 3], 2))
      select_row.extend(random.sample([5, 6, 7, 8], 2))
      select_row = sorted(select_row)
      # select 5 col
      select_col = [4, ]
      select_col.extend(random.sample([0, 1, 2, 3], 2))
      select_col.extend(random.sample([5, 6, 7, 8], 2))
      select_col = sorted(select_col)
      LF = LF[select_row][:, select_col]

    # random crop a patch
    U, V, H, W, _ = LF.shape
    patch_size = patch_size * scale
    rnd_h = random.randint(0, max(0, H - patch_size))
    rnd_w = random.randint(0, max(0, W - patch_size))
    LF = LF[:, :, rnd_h: rnd_h + patch_size, rnd_w: rnd_w + patch_size, 0:3]

    # convert to sai views
    LF = np.transpose(LF, (0, 2, 1, 3, 4)).reshape([U * patch_size, V * patch_size, 3])

    # convert to yuv and select y only
    Sr = LFSRDataset.rgb2ycbcr(LF * 255.0) / 255.0
    Sr_uv = Sr[..., 1:3]
    Hr = Sr[..., 0:1]
    Lr = imresize.imresize(Hr, scalar_scale=1.0 / scale)

    return Lr, Hr, Sr_uv

  @staticmethod
  def parse_h5(path, transpose=True):
    with h5py.File(path, 'r') as hf:
      Hr = np.expand_dims(np.array(hf.get('Hr_SAI_y')), axis=2)
      Lr = np.expand_dims(np.array(hf.get('Lr_SAI_y')), axis=2)
      # [2, 3000, 2000] -> [3000, 2000, 2]
      Sr_uv = np.array(hf.get('Sr_SAI_cbcr'))  # .transpose(1, 2, 0)

      if transpose:
        Hr = np.transpose(Hr, (1, 0, 2))
        Lr = np.transpose(Lr, (1, 0, 2))
        if Sr_uv.ndim > 0:
          Sr_uv = np.transpose(Sr_uv, (0, 2, 1))

      return Lr, Hr, Sr_uv

  def __len__(self):
    return len(self.targets)

  def __getitem__(self, idx):
    """fetch elements
    """
    filepath = self.targets[idx]

    if filepath.endswith('.h5'):
      lr, hr, uv = self.parse_h5(filepath, transpose=self.transpose)
    elif filepath.endswith('.mat'):
      mat = self.cache[filepath]
      lr, hr, uv = self.parse_mat(mat, patch_size=self.patch_size, scale=self.scale)
    else:
      raise NotImplementedError(filepath)

    if self.phase != 'test':
      return self.transform(lr, hr, uv)
    else:
      return self.transform(lr, uv, filepath)


#!<-----------------------------------------------------------------------------
#!< PREPROCESS and POSTPROCESS
#!<-----------------------------------------------------------------------------


class LF_divide_integrate(object):

  def __init__(self, scale, patch_size, stride):
    self.scale = scale
    self.patch_size = patch_size
    self.stride = stride
    self.bdr = (patch_size - stride) // 2
    self.pad = torch.nn.ReflectionPad2d(padding=(self.bdr, self.bdr + stride - 1, self.bdr, self.bdr + stride - 1))

  def ImageExtend(self, Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]

    return Im_out

  def LFdivide(self, LF):
    assert LF.size(0) == 1, 'The batch_size of LF for test requires to be one!'
    LF = LF.squeeze(0)
    [c, u, v, h, w] = LF.size()
    self.img_h = h
    self.img_w = w

    LF = einops.rearrange(LF, 'c u v h w -> (c u v) 1 h w')
    self.numU = (h + self.bdr * 2 - 1) // self.stride
    self.numV = (w + self.bdr * 2 - 1) // self.stride

    # LF_pad = self.pad(LF)
    LF_pad = self.ImageExtend(LF, [self.bdr, self.bdr + self.stride - 1, self.bdr, self.bdr + self.stride - 1])
    LF_divided = F.unfold(LF_pad, kernel_size=self.patch_size, stride=self.stride)
    LF_divided = einops.rearrange(LF_divided, '(c u v) (h w) (numU numV) -> (numU numV) c u v h w', u=u, v=v, h=self.patch_size, w=self.patch_size, numU=self.numU, numV=self.numV)  # nopep8
    return LF_divided

  def LFintegrate(self, LF_divided):
    LF_divided = LF_divided[:, :, :, :, self.bdr * self.scale:(self.bdr + self.stride) * self.scale, self.bdr * self.scale:(self.bdr + self.stride) * self.scale]  # nopep8
    LF = einops.rearrange(LF_divided, '(numU numV) c u v h w -> c u v (numU h) (numV w)', numU=self.numU, numV=self.numV)  # nopep8
    return LF[..., : self.img_h * self.scale, : self.img_w * self.scale]

#!<---------------------------------------------------------------------------
#!< PSW
#!<---------------------------------------------------------------------------


class LF_divide_integrate_psw(object):

  def __init__(self, scale, patch_size, stride):
    self.scale = scale
    self.patch_size = patch_size
    self.stride = stride
    self.bdr = (patch_size - stride) // 2
    self.pad = torch.nn.ReflectionPad2d(padding=(self.bdr, self.bdr + stride - 1, self.bdr, self.bdr + stride - 1))

  def LFdivide(self, LF):
    assert LF.size(0) == 1, 'The batch_size of LF for test requires to be one!'
    LF = LF.squeeze(0)
    [c, u0, v0, h0, w0] = LF.size()
    stride = self.stride
    patch_size = self.patch_size

    self.sai_h = h0
    self.sai_w = w0

    sub_lf = []
    numU = 0
    for y in range(0, h0, stride):
      numV = 0
      for x in range(0, w0, stride):
        if y + patch_size > h0 and x + patch_size <= w0:
          sub_lf.append(LF[..., h0 - patch_size:, x: x + patch_size])
        elif y + patch_size <= h0 and x + patch_size > w0:
          sub_lf.append(LF[..., y: y + patch_size, w0 - patch_size:])
        elif y + patch_size > h0 and x + patch_size > w0:
          sub_lf.append(LF[..., h0 - patch_size:, w0 - patch_size:])
        else:
          sub_lf.append(LF[..., y: y + patch_size, x: x + patch_size])
        numV += 1
      numU += 1

    LF_divided = torch.stack(sub_lf, dim=0)
    return LF_divided

  def LFintegrate(self, LF_divided):
    # each SAI size
    stride = self.stride * self.scale
    patch_size = self.patch_size * self.scale
    bdr = self.stride // 2

    # rearrange to SAI views
    # print(LF_divided.shape)
    _, c, u, v, h, w = LF_divided.size()
    h1 = self.sai_h * self.scale
    w1 = self.sai_w * self.scale

    # allocate space
    out = torch.zeros(c, u, v, h1, w1).to(LF_divided.device)
    mask = torch.zeros(c, u, v, h1, w1).to(LF_divided.device)

    # colllect outter for patch_size
    idx = 0
    for y in range(0, h1, stride):
      for x in range(0, w1, stride):
        if y + patch_size > h1 and x + patch_size <= w1:
          out[..., h1 - patch_size:, x: x + patch_size] += LF_divided[idx]
          mask[..., h1 - patch_size:, x: x + patch_size] += 1
        elif y + patch_size <= h1 and x + patch_size > w1:
          out[..., y: y + patch_size, w1 - patch_size:] += LF_divided[idx]
          mask[..., y: y + patch_size, w1 - patch_size:] += 1
        elif y + patch_size > h1 and x + patch_size > w1:
          out[..., h1 - patch_size:, w1 - patch_size:] += LF_divided[idx]
          mask[..., h1 - patch_size:, w1 - patch_size:] += 1
        else:
          out[..., y: y + patch_size, x: x + patch_size] += LF_divided[idx]
          mask[..., y: y + patch_size, x: x + patch_size] += 1
        idx += 1
    final = out / mask
    return final

#!<---------------------------------------------------------------------------
#!< PSW++
#!<---------------------------------------------------------------------------


class LF_divide_integrate_pswpp(object):
  def __init__(self, scale, patch_size, stride):
    self.scale = scale
    self.patch_size = patch_size
    self.stride = stride
    self.bdr = (patch_size - stride) // 2
    self.pad = torch.nn.ReflectionPad2d(padding=(self.bdr, self.bdr + stride - 1, self.bdr, self.bdr + stride - 1))

  def LFdivide(self, LF):
    assert LF.size(0) == 1, 'The batch_size of LF for test requires to be one!'
    LF = LF.squeeze(0)
    [c, u0, v0, h0, w0] = LF.size()
    stride = self.stride
    patch_size = self.patch_size

    self.sai_h = h0
    self.sai_w = w0

    sub_lf = []
    numU = 0
    for y in range(0, h0, stride):
      numV = 0
      for x in range(0, w0, stride):
        if y + patch_size > h0 and x + patch_size <= w0:
          sub_lf.append(LF[..., h0 - patch_size:, x: x + patch_size])
        elif y + patch_size <= h0 and x + patch_size > w0:
          sub_lf.append(LF[..., y: y + patch_size, w0 - patch_size:])
        elif y + patch_size > h0 and x + patch_size > w0:
          sub_lf.append(LF[..., h0 - patch_size:, w0 - patch_size:])
        else:
          sub_lf.append(LF[..., y: y + patch_size, x: x + patch_size])
        numV += 1
      numU += 1

    LF_divided = torch.stack(sub_lf, dim=0)
    return LF_divided

  def LFintegrate(self, LF_divided):
    # each SAI size
    stride = self.stride * self.scale
    patch_size = self.patch_size * self.scale
    bdr = self.stride // 2

    # rearrange to SAI views
    _, c, u, v, h, w = LF_divided.size()
    h1 = self.sai_h * self.scale
    w1 = self.sai_w * self.scale

    # allocate space
    out = torch.zeros(c, u, v, h1, w1).to(LF_divided.device)
    mask = torch.zeros(c, u, v, h1, w1).to(LF_divided.device)

    # colllect outter for patch_size
    idx = 0
    for y in range(0, h1, stride):
      for x in range(0, w1, stride):
        if y + patch_size > h1 and x + patch_size <= w1:
          out[..., h1 - patch_size:, x: x + patch_size] += LF_divided[idx]
          mask[..., h1 - patch_size:, x: x + patch_size] += 1
        elif y + patch_size <= h1 and x + patch_size > w1:
          out[..., y: y + patch_size, w1 - patch_size:] += LF_divided[idx]
          mask[..., y: y + patch_size, w1 - patch_size:] += 1
        elif y + patch_size > h1 and x + patch_size > w1:
          out[..., h1 - patch_size:, w1 - patch_size:] += LF_divided[idx]
          mask[..., h1 - patch_size:, w1 - patch_size:] += 1
        else:
          out[..., y: y + patch_size, x: x + patch_size] += LF_divided[idx]
          mask[..., y: y + patch_size, x: x + patch_size] += 1
        idx += 1
    # final = out / mask

    # collect inner for patch_size
    idx = 0
    out_in = torch.zeros(c, u, v, h1, w1).to(LF_divided.device)
    mask_in = torch.zeros(c, u, v, h1, w1).to(LF_divided.device)
    for y in range(0, h1, stride):
      for x in range(0, w1, stride):
        if y + patch_size > h1 and x + patch_size <= w1:
          pass
        elif y + patch_size <= h1 and x + patch_size > w1:
          pass
        elif y + patch_size > h1 and x + patch_size > w1:
          pass
        else:
          out_in[..., y + bdr: y + bdr + stride, x + bdr: x + bdr + stride] += LF_divided[idx][..., bdr: bdr + stride, bdr: bdr + stride]  # nopep8
          mask_in[..., y + bdr: y + bdr + stride, x + bdr: x + bdr + stride] += 1
        idx += 1

    # inner to zero
    mask[mask_in != 0] = 0
    out[mask_in != 0] = 0
    final = (out + out_in) / (mask + mask_in)

    return final

#!<-----------------------------------------------------------------------------
#!< TRAINING TRANSFORM
#!<-----------------------------------------------------------------------------

def flip(img, mode):
  if mode == 0:
    return img
  elif mode == 1:
    return np.flipud(np.rot90(img))
  elif mode == 2:
    return np.flipud(img)
  elif mode == 3:
    return np.rot90(img, k=3)
  elif mode == 4:
    return np.flipud(np.rot90(img, k=2))
  elif mode == 5:
    return np.rot90(img)
  elif mode == 6:
    return np.rot90(img, k=2)
  elif mode == 7:
    return np.flipud(np.rot90(img, k=3))

def to_tensor(inputs: np.ndarray, scale=None, mean=None, std=None, **kwargs) -> np.ndarray:
  # mean = torch.tensor(mean) if mean is not None else None
  # std = torch.tensor(std) if std is not None else None

  if inputs.ndim == 3:
    m = torch.from_numpy(np.ascontiguousarray(inputs.transpose((2, 0, 1))))
  elif inputs.ndim == 2:
    m = torch.from_numpy(inputs)[None]
  elif inputs.ndim == 4:
    m = torch.from_numpy(np.ascontiguousarray(inputs.transpose((0, 3, 1, 2))))
  else:
    raise NotImplementedError(inputs.ndim)

  m = m.type(torch.FloatTensor)
  if scale is not None:
    m = m.float().div(scale)
  if mean is not None:
    m.sub_(torch.tensor(mean)[:, None, None])
  if std is not None:
    m.div_(torch.tensor(std)[:, None, None])
  return m

def transform_test_lfsr(lr_img, uv_img, path, scale=4):
  """input ycbcr [0, 1.0] float
  """
  lr_img = to_tensor(lr_img, scale=1.0)
  return lr_img, uv_img, path


def transform_val_lfsr(lr_img, hr_img, uv_img, scale=4):
  """input ycbcr [0, 1.0] float
  """
  lr_img = to_tensor(lr_img, scale=1.0)
  hr_img = to_tensor(hr_img, scale=1.0)
  return lr_img, hr_img


def transform_train_lfsr(lr_img, hr_img, uv_img, patch_size=32, scale=4, angular=5):
  """input ycbcr [0, 1.0] float
  """
  # randomly crop the HR patch
  H, W, C = lr_img.shape

  # augmentation - flip and/or rotate
  mode = random.randint(0, 7)
  lr_img = flip(lr_img, mode=mode)
  hr_img = flip(hr_img, mode=mode)

  # convert to tensor
  lr_img = to_tensor(lr_img, scale=1.0)
  hr_img = to_tensor(hr_img, scale=1.0)
  return lr_img, hr_img

#!<-----------------------------------------------------------------------------
#!< TRAINING PIPELINE
#!<-----------------------------------------------------------------------------

class ModelEMA():

  def __init__(self, model: nn.Module, decay=0.997, **kwargs):
    super(ModelEMA, self).__init__()
    self.decay = decay
    self.model = model
    self.ema_model = copy.deepcopy(model.eval())

  def step(self):
    if self.decay <= 0:
      return
    params = dict(self.model.named_parameters())
    ema_params = dict(self.ema_model.named_parameters())
    for i, k in enumerate(params):
      ema_params[k].data.mul_(self.decay).add_(params[k].data, alpha=1 - self.decay)

  def model(self):
    return self.ema_model

  def state_dict(self):
    return self.ema_model.state_dict()

  def forward(self, *args, **kwargs):
    return self.ema_model(*args, **kwargs)

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

class LFSR():

  """Light-Field Super-Resolution Training Pipeline
  """

  def __init__(self, config):
    self.Config = config

    # scalar
    self.Epoch = 0
    self.Step = 0

    # models
    self.Model = self.build_model()

    # optim
    self.Optim = self.build_optim(self.Model)

    # load model and possible optimizer
    if self.Config.model_path is not None:
      content = torch.load(self.Config.model_path)['state_dict']
      ckpt = {}
      for k, v in content.items():
        ckpt[k.replace('module.', '')] = v
      self.Model.load_state_dict(ckpt)

    # extend to distributed
    if self.Config.task == 'train':
      self.Model = torch.nn.DataParallel(self.Model)

    # setting ema: note that loadding paramaters first
    self.ModelEMA = ModelEMA(self.Model, decay=self.Config.ema_decay)

    # LFSR specific pre/post process:
    if self.Config.processor == 'vanilla':
      processor = LF_divide_integrate
    elif self.Config.processor == 'psw':
      processor = LF_divide_integrate_psw
    elif self.Config.processor == 'psw++':
      processor = LF_divide_integrate_pswpp
    self.Processor = processor(self.Config.scale, self.Config.patch_size, self.Config.stride)

  #!<---------------------------------------------------------------------------
  #!< PREPARE
  #!<---------------------------------------------------------------------------

  def __call__(self):
    return {
        'train': self.train,
        'val': self.val,
        'val_all': self.val_all,
        'test': self.test,
    }[self.Config.task]()

  def dump(self):
    cfg = self.Config

    # save main model
    path = f'{cfg.root}/model.epoch-{self.Epoch}.step-{self.Step}.pth'
    torch.save({
        'state_dict': self.Model.state_dict(),
        'global_step': self.Step,
        'global_epoch': self.Epoch,
        'optimizer': self.Optim.state_dict(),
    }, path)
    print(f'Model has saved in {path}')

    # save ema model
    path = f'{cfg.root}/model.epoch-{self.Epoch}.step-{self.Step}.ema.pth'
    torch.save({'state_dict': self.ModelEMA.state_dict()}, path)
    print(f'EMA Model has saved in {path}')

  def build_model(self):
    cfg = self.Config
    device = self.Config.device
    angular, scale = cfg.angular, cfg.scale

    module = importlib.import_module(f'{cfg.model}')
    model = getattr(module, cfg.model)(angRes_in=angular, scale_factor=scale, use_flashatten=True)

    model.to(device)
    return model

  def build_optim(self, model):
    cfg = self.Config
    if cfg.task == 'train':
      optim = torch.optim.Adam(model.parameters(), lr=cfg.train_lr, betas=(0.99, 0.999), eps=1e-08, weight_decay=0.0)
    else:
      optim = None
    return optim

  def build_dataset(self, phase):
    cfg = self.Config

    if cfg.scale == 2:
      paths = PROTOCALS_2X[cfg.dataset][phase]
    elif cfg.scale == 4:
      paths = PROTOCALS_4X[cfg.dataset][phase]

    transform_train = functools.partial(
        transform_train_lfsr,
        patch_size=cfg.patch_size,
        scale=cfg.scale,
        angular=cfg.angular)

    if phase == 'train':
      dataset = LFSRDataset(phase=phase, paths=paths, transform=transform_train, scale=cfg.scale)
    elif phase == 'val':
      dataset = LFSRDataset(phase=phase, paths=paths, transform=transform_val_lfsr, scale=cfg.scale)
    elif phase == 'test':
      dataset = LFSRDataset(phase=phase, paths=paths, transform=transform_test_lfsr, scale=cfg.scale)
    else:
      raise NotImplementedError(phase)

    return dataset

  def build_dataloader(self, phase, dataset):
    """build dataloader
    """
    cfg = self.Config
    if phase == 'train':
      return torch.utils.data.DataLoader(
          dataset=dataset,
          batch_size=cfg.train_batchsize,
          num_workers=4,
          shuffle=True,
          pin_memory=True,
          drop_last=True,
          prefetch_factor=2,
          persistent_workers=True)

    else:
      return torch.utils.data.DataLoader(
          dataset=dataset,
          batch_size=1,
          shuffle=False,
          num_workers=16,
          pin_memory=False,
          drop_last=False)

  #!<---------------------------------------------------------------------------
  #!< TRAINING
  #!<---------------------------------------------------------------------------

  def optimize(self, lr_images, hr_images):
    cfg = self.Config
    device = cfg.device

    hr_preds = self.Model(lr_images)
    losses = {'loss_l1': self.loss_l1(hr_preds, hr_images)}

    return losses

  def lr_scheduler(self):
    cfg = self.Config

    if cfg.train_lr_scheduler == 1:
      scheduler = torch.optim.lr_scheduler.StepLR(self.Optim, step_size=15, gamma=0.5)
    elif cfg.train_lr_scheduler == 2:
      scheduler = torch.optim.lr_scheduler.StepLR(self.Optim, step_size=25, gamma=0.5)
    elif cfg.train_lr_scheduler == 3:
      scheduler = torch.optim.lr_scheduler.StepLR(self.Optim, step_size=80, gamma=0.5)

    return scheduler

  def train(self):
    cfg = self.Config
    device = self.Config.device
    init_step = self.Step

    # build train dataset
    train_set = self.build_dataset('train')
    train_loader = self.build_dataloader('train', train_set)
    total_step = len(train_loader) * cfg.train_epoch

    # print trainable parameters
    lr_scheduler = self.lr_scheduler()

    # loss collections
    self.loss_l1 = nn.L1Loss()

    # training loop
    while self.Epoch < cfg.train_epoch:
      self.Epoch += 1
      self.Model.train()

      for lr_images, hr_images in train_loader:
        self.Step += 1

        # moving to device
        lr_images = lr_images.float().to(device)
        hr_images = hr_images.float().to(device)

        # compute loss
        losses = self.optimize(lr_images, hr_images)

        # accumulate gradient
        loss = sum(loss for loss in losses.values())
        self.Optim.zero_grad()
        loss.backward()
        self.Optim.step()
        self.ModelEMA.step()

        # display
        if self.Step % cfg.log == 0:
          print(self.Epoch, self.Step, loss.item())

        # end of step

      if self.Epoch % cfg.log_save == 0 and cfg.master:
        self.dump()

      if self.Epoch % cfg.log_val == 0 and cfg.master:
        self.val_all()

      lr_scheduler.step()

      # end of epoch

  #!<---------------------------------------------------------------------------
  #!< VAL or TEST
  #!<---------------------------------------------------------------------------

  def inference_patch(self, lr_images):
    cfg = self.Config
    if cfg.ema_decay > 0:
      model = self.ModelEMA
    else:
      model = self.Model
    return model(lr_images)

  def inference_tta(self, lr_images):
    hr_preds = []
    hr_preds.append(self.inference_patch(lr_images))
    hr_preds.append(self.inference_patch(lr_images.rot90(1, [2, 3]).flip([2])).flip([2]).rot90(3, [2, 3]))
    hr_preds.append(self.inference_patch(lr_images.flip([2])).flip([2]))
    hr_preds.append(self.inference_patch(lr_images.rot90(3, [2, 3])).rot90(1, [2, 3]))
    hr_preds.append(self.inference_patch(lr_images.rot90(2, [2, 3]).flip([2])).flip([2]).rot90(2, [2, 3]))
    hr_preds.append(self.inference_patch(lr_images.rot90(1, [2, 3])).rot90(3, [2, 3]))
    hr_preds.append(self.inference_patch(lr_images.rot90(2, [2, 3])).rot90(2, [2, 3]))
    hr_preds.append(self.inference_patch(lr_images.rot90(3, [2, 3]).flip([2])).flip([2]).rot90(1, [2, 3]))
    return torch.stack(hr_preds, dim=0).mean(dim=0)

  def inference(self, lr_images):
    assert lr_images.size(0) == 1, f"require input batchsize should be 1."

    cfg = self.Config
    device = cfg.device
    scale = cfg.scale
    angular = cfg.angular
    img_h, img_w = lr_images.shape[-2:]

    # expand to aperture mode
    sub_lf = einops.rearrange(lr_images, 'b c (u h) (v w) -> b c u v h w', u=angular, v=angular)

    # crop to patches: [70, 1, 5, 5, 32, 32] -> [70, 1, 5 * 32, 5 * 32]
    sub_lf = self.Processor.LFdivide(sub_lf)
    # print(sub_lf.shape)
    sub_lf = einops.rearrange(sub_lf, 'n c u v h w -> n c (u h) (v w)', u=angular, v=angular)
    sub_lf_out = torch.zeros_like(sub_lf).repeat(1, 1, scale, scale)

    # # loop for every pathces
    for i in range(sub_lf.size(0)):
      if cfg.tta:
        sub_lf_out[i: i + 1] = self.inference_tta(sub_lf[i: i + 1])
      else:
        sub_lf_out[i: i + 1] = self.inference_patch(sub_lf[i: i + 1])

    # # intergrate into one image
    sub_lf_out = einops.rearrange(sub_lf_out, 'n c (u h) (v w) -> n c u v h w', u=angular, v=angular)
    sub_lf_out = self.Processor.LFintegrate(sub_lf_out)
    sub_lf_out = einops.rearrange(sub_lf_out, 'c u v h w -> 1 c (u h) (v w)', u=angular, v=angular)

    return sub_lf_out

  def compute_psnr_ssim(self, sr, gt):
    """compute psnr and ssim index in terms of matlab version

    Args:
        sr (torch.Tensor): [N, 1, H, W] in [0, 1] float
        gt (torch.Tensor): [N, 1, H, W] in [0, 1] float

    Returns:
        dict: {'psnr': v, 'ssim': v}
    """
    cfg = self.Config
    angular = cfg.angular
    assert sr.size(0) == gt.size(0) == 1, "Current only support batchsize to 1."

    # following basic lfsr mode: using float32 (0, 1) to compute
    sr = einops.rearrange(sr, '1 1 (u h) (v w) -> u v h w', u=angular, v=angular).cpu().numpy()
    gt = einops.rearrange(gt, '1 1 (u h) (v w) -> u v h w', u=angular, v=angular).cpu().numpy()

    psnr = np.zeros([angular, angular], dtype=np.float32)
    ssim = np.zeros([angular, angular], dtype=np.float32)

    for y, x in zip(range(angular), range(angular)):
      psnr[y, x] = metrics.peak_signal_noise_ratio(gt[y, x], sr[y, x])
      ssim[y, x] = metrics.structural_similarity(gt[y, x], sr[y, x], gaussian_weights=True)

    psnr = psnr.sum() / np.sum(psnr > 0)
    ssim = ssim.sum() / np.sum(ssim > 0)

    return {'psnr': psnr, 'ssim': ssim}

  @torch.no_grad()
  def val(self, loader=None, **kwargs):
    cfg = self.Config
    device = self.Config.device

    # reset
    self.Model.eval()

    # dataset
    if loader is None:
      dataset = self.build_dataset('val')
      loader = self.build_dataloader('val', dataset)

    # create foler for every epoch
    root = os.makedirs(f'{cfg.root}/val/epoch_{self.Epoch}_step_{self.Step}/', exist_ok=True)
    reports = {'psnr': [], 'ssim': []}

    # loop infernece
    step = 0
    total_step = int(len(loader))

    for lr_images, hr_images in loader:
      step += 1

      # moving to device [B, C, 5 * H, 5 * W]
      lr_images = lr_images.float().to(device)
      hr_images = hr_images.float().to(device)

      # inference
      hr_preds = self.inference(lr_images)
      # cv2.imwrite(f'img_{step}.png', torch.cat([hr_preds, hr_images, (hr_preds - hr_images).abs()], dim=3)[0][0].mul(255).round().byte().cpu().numpy())

      # compute psnr/ssim per images
      metric = self.compute_psnr_ssim(hr_preds, hr_images)
      print('{}/{}, psnr: {}, ssim: {}'.format(step, total_step, metric['psnr'], metric['ssim']))
      reports['psnr'].append(metric['psnr'])
      reports['ssim'].append(metric['ssim'])

    reports['psnr'] = np.mean(reports['psnr'])
    reports['ssim'] = np.mean(reports['ssim'])
    return reports


  @torch.no_grad()
  def val_all(self):
    """validation for all lfsr dataset
    """
    total_psnr, total_ssim = [], []
    real_psnr, synth_psnr = [], []

    for dataset in ['LFSR.EPFL', 'LFSR.HCInew', 'LFSR.HCIold', 'LFSR.INRIA', 'LFSR.STFgantry']:
      self.Config.dataset = dataset
      reports = self.val()
      total_psnr.append(float(reports['psnr']))
      total_ssim.append(float(reports['ssim']))

      if dataset in ['LFSR.EPFL', 'LFSR.INRIA', 'LFSR.STFgantry']:
        real_psnr.append(total_psnr[-1])
      else:
        synth_psnr.append(total_psnr[-1])

    print('Epoch:{}, Iter:{}, mean_psnr: {:.4f}, mean_ssim: {:.4f}'.format(self.Epoch, self.Step, np.mean(total_psnr), np.mean(total_ssim))) # nopep8


  @torch.no_grad()
  def test(self, **kwargs):
    """challenge submission
    """
    cfg = self.Config
    device = cfg.device
    angular = cfg.angular

    # reset
    self.Model.eval()

    # dataset
    dataset = self.build_dataset('test')
    loader = self.build_dataloader('test', dataset)

    # create folder for every epoch
    root = f'{cfg.root}/test/epoch_{self.Epoch}_step_{self.Step}/'
    os.makedirs(root, exist_ok=True)
    cache = {
        'path': cfg.model_path,
        'name': cfg.name,
        'pred': [],
        'path': [],
    }

    # start
    step, total = 0, int(len(loader))
    for y_lr, uv_sr, filepath in tqdm.tqdm(loader):

      # count
      step += 1
      y_lr = y_lr.float().to(device)
      uv_sr = uv_sr.float().to(device)

      # inference
      y_pred = self.inference(y_lr)
      yuv_sr = torch.cat([y_pred, uv_sr], dim=1)  # [1, 3, h, w] in (0, 1)

      # cache
      cache['pred'].append(yuv_sr)
      cache['path'].append(filepath)

      # construct dst folder
      folder, name = filepath[0].split('/')[-2:]
      folder = 'Real' if 'Real' in folder else 'Synth'
      dstdir = f'{root}/{folder}/{name[:-3]}/'
      os.makedirs(dstdir, exist_ok=True)
      print(dstdir)

      # save sub-image to folder
      yuv_sr = einops.rearrange(yuv_sr, '1 c (u h) (v w) -> u v h w c', u=angular, v=angular)
      yuv_sr = yuv_sr.cpu().mul(255.0).numpy()
      for i in range(angular):
        for j in range(angular):
          out = np.clip(dataset.ycbcr2rgb(yuv_sr[i, j]).round(), 0, 255).astype('uint8')
          imageio.imwrite(f'{dstdir}/View_{i}_{j}.bmp', out)

    # save
    torch.save(cache, os.path.join(root, 'test.pth'))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # ---------------------------------------------
  #  USED BY CONTEXT
  # ---------------------------------------------
  parser.add_argument('--name', type=str, default='LFSR')
  parser.add_argument('--root', type=str, default=None, help="None for creating, otherwise specific root.")
  parser.add_argument('--device', type=str, default='cuda:0')
  parser.add_argument('--output_dir', type=str, default='_outputs', help="default output folder.")

  # ---------------------------------------------
  #  USED BY COMMON
  # ---------------------------------------------
  parser.add_argument('--task', type=str, default=None, choices=['train', 'val', 'val_all', 'test'])
  parser.add_argument('--dataset', type=str, default=None)

  # ---------------------------------------------
  #  USED BY LOGGER
  # ---------------------------------------------
  parser.add_argument('--log', type=int, default=10, help="display interval step.")
  parser.add_argument('--log-val', type=int, default=1, help="running validation in terms of step.")
  parser.add_argument('--log-save', type=int, default=1, help="saveing checkpoint with interval.")

  # ---------------------------------------------
  #  USED BY MODEL-SPECIFIC
  # ---------------------------------------------
  parser.add_argument('--model', type=str, default=None, help="")
  parser.add_argument('--model-path', type=str, default=None, help="loadding pretrain/last-checkpoint model.")
  parser.add_argument('--model-ema-path', type=str, default=None, help="loadding pretrain/last-checkpoint model.")
  parser.add_argument('--model-source', type=str, default=None)

  # ---------------------------------------------
  #  USED BY TRAIN-SPECIFIC
  # ---------------------------------------------
  parser.add_argument('--train-batchsize', type=int, default=32, help="total batch size across devices.")
  parser.add_argument('--train-epoch', type=int, default=100, help="total training epochs.")
  parser.add_argument('--train-lr', type=float, default=2e-4, help="training learning rate.")
  parser.add_argument('--train-lr-scheduler', type=int, default=1, help='scheduler mode.')
  parser.add_argument('--input-colorspace', type=str, default='Y', choices=['Y', 'YUV', 'RGB'])
  parser.add_argument('--ema-decay', type=float, default=0.0, help="using EMA techniques.")

  # ---------------------------------------------
  #  USED BY INPUT-SPECIFIC
  # ---------------------------------------------
  parser.add_argument('--scale', type=int, default=4, help="upsample scale.")
  parser.add_argument('--mode', type=str, default='batch', choices=['single', 'batch'])
  parser.add_argument('--angular', type=int, default=5, choices=[5, 9])
  parser.add_argument('--processor', type=str, default='vanilla', choices=['vanilla', 'psw', 'psw++'])

  # only for test tta
  parser.add_argument('--tta', action='store_true', help="test time augmentation.")
  parser.add_argument('--patch-size', type=int, default=32)
  parser.add_argument('--stride', type=int, default=16)

  import torch.backends.cudnn as cudnn
  cudnn.benchmark = True

  config, _ = parser.parse_known_args()
  config.root = "%s/%s" % (config.output_dir, config.name)
  os.makedirs(config.root, exist_ok=True)

  LFSR(config)()
