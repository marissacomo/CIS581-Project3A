'''
  File name: feat_match.py
  Author:
  Date created:
'''

'''
  File clarification:
    Matching feature descriptors between two images. You can use k-d tree to find the k nearest neighbour. 
    Remember to filter the correspondences using the ratio of the best and second-best match SSD. You can set the threshold to 0.6.
    - Input descs1: 64 × N1 matrix representing the corner descriptors of first image.
    - Input descs2: 64 × N2 matrix representing the corner descriptors of second image.
    - Outpuy match: N1 × 1 vector where match i points to the index of the descriptor in descs2 that matches with the
                    feature i in descriptor descs1. If no match is found, you should put match i = −1.
'''

import numpy as np
import cv2

def feat_match(descs1, descs2):

  a2 = np.zeros((descs1.shape[0] * descs1.shape[0], descs1.shape[1]))
  b2 = np.zeros((descs1.shape[0] * descs1.shape[0], descs1.shape[1]))
  d = np.zeros((descs1.shape[0] * descs1.shape[0]))

  result = np.zeros((descs1.shape[0]))
  result.fill(-1)

  for i in range(0, descs1.shape[0]):
    a2[(i * descs1.shape[0]):((i + 1) * descs1.shape[0]), :] = np.tile(descs1[i], (descs1.shape[0], 1))
    b2[(i * descs1.shape[0]):((i + 1) * descs1.shape[0]), :] = np.copy(descs2)

  d = np.subtract(a2, b2)
  d = np.multiply(d, d)
  d = np.sum(d, axis=1)
  d = np.sqrt(d)

  index_grid = np.meshgrid(np.arange(descs1.shape[0]))[0]

  for i in range(0, descs1.shape[0]):
    d_part = d[(i * descs1.shape[0]):((i + 1) * descs1.shape[0])]
    indices = d_part.argsort()

    sorted_d = d_part[indices]

    ratio = sorted_d[0] / sorted_d[1]

    if ratio < 0.75:
      result[i] = indices[0]

  return result