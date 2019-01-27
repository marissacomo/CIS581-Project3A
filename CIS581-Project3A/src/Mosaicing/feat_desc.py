'''
  File name: feat_desc.py
  Author:
  Date created:
'''

'''
  File clarification:
    Extracting Feature Descriptor for each feature point. You should use the subsampled image around each point feature, 
    just extract axis-aligned 8x8 patches. Note that it’s extremely important to sample these patches from the larger 40x40 
    window to have a nice big blurred descriptor. 
    - Input img: H × W matrix representing the gray scale input image.
    - Input x: N × 1 vector representing the column coordinates of corners.
    - Input y: N × 1 vector representing the row coordinates of corners.
    - Outpuy descs: 64 × N matrix, with column i being the 64 dimensional descriptor (8 × 8 grid linearized) computed at location (xi , yi) in img.
'''

import numpy as np
import cv2


def normalize_list(listObj):
  norm = np.linalg.norm(listObj)

  if (norm == 0):
    print('Divide by zero: ', listObj)

  normalized_list = listObj / norm
  return normalized_list

def feat_desc(img, x, y):

  spacing = 5
  samples = 8

  half = int(samples / 2)

  x_offset, y_offset = np.meshgrid(np.arange(-spacing * half, spacing *
                                             half, spacing),
                                   np.arange(-spacing * half, spacing *
                                             half, spacing))

  x_offset[:,half:] += spacing
  y_offset[half:, :] += spacing

  # print(x_offset, y_offset)

  descs = np.zeros((x.shape[0], samples * samples))

  x_offset = x_offset.flatten()
  y_offset = y_offset.flatten()

  sample_offset_x = np.tile(x_offset, (x.shape[0], 1))
  sample_offset_y = np.tile(y_offset, (y.shape[0], 1))

  sample_offset_x = sample_offset_x + np.tile(np.array([x]).transpose(), (1, x_offset.shape[0]))
  sample_offset_y = sample_offset_y + np.tile(np.array([y]).transpose(), (1, y_offset.shape[0]))

  nc_img = img.shape[1]
  nr_img = img.shape[0]
  sample_offset_x = np.clip(sample_offset_x, 0, nr_img - 1).astype(np.int32)
  sample_offset_y = np.clip(sample_offset_y, 0, nc_img - 1).astype(np.int32)

  descs = img[sample_offset_x, sample_offset_y].astype(np.float64)

  for i in range(0, descs.shape[0]):
    descs[i] = descs[i] - np.mean(descs[i])
    descs[i] = normalize_list(descs[i])

  # print('Feature Descs', descs)

  return descs
