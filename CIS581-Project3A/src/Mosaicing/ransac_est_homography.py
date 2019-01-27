'''
  File name: ransac_est_homography.py
  Author:
  Date created:
'''

'''
  File clarification:
    Use a robust method (RANSAC) to compute a homography. Use 4-point RANSAC as 
    described in class to compute a robust homography estimate:
    - Input x1, y1, x2, y2: N × 1 vectors representing the correspondences feature coordinates in the first and second image. 
                            It means the point (x1_i , y1_i) in the first image are matched to (x2_i , y2_i) in the second image.
    - Input thresh: the threshold on distance used to determine if transformed points agree.
    - Outpuy H: 3 × 3 matrix representing the homograph matrix computed in final step of RANSAC.
    - Output inlier_ind: N × 1 vector representing if the correspondence is inlier or not. 1 means inlier, 0 means outlier.
'''

import numpy as np
import cv2
import random
import math
import sys

from est_homography import est_homography

def normalize_list(listObj):
  norm = np.linalg.norm(listObj)

  if (norm == 0):
    print('Divide by zero: ', listObj)

  normalized_list = listObj / norm
  return normalized_list

def get_max_inliers(t1, t2, thresh):
  combos = np.zeros((t1.shape[0], 2))

  size = int((combos.shape[0] * (combos.shape[0] - 1)) / 2)
  all_combos = np.zeros((size, 4))

  for i in range(0, t1.shape[0]):
    combos[i, 0] = t1[i]
    combos[i, 1] = t2[i]

  offset = 0

  for i in range(0, combos.shape[0]):
    temp_size = combos.shape[0] - i - 1

    offset_end = offset + temp_size

    copy_start = i + 1
    copy_end = copy_start + temp_size

    # print('copy range', copy_start, copy_end)
    # print('offset', offset, offset_end)
    # print(np.tile(combos[i], (temp_size, 1)).shape)
    # print(np.copy(combos[copy_start:copy_end]).shape)
  
    tiled_value = np.tile(combos[i], (temp_size, 1))

    all_combos[offset:offset_end, 0:2] = tiled_value
    all_combos[offset:offset_end, 2:4] = np.copy(combos[copy_start:copy_end])

    offset = offset_end

  results = []
  resultInliers = []

  k = 0

  while(len(all_combos) > 0 and k < 1000):
    print('Combo Shape: ', all_combos.shape)

    rand_combo_idx = random.randrange(all_combos.shape[0])
    selected_combo = all_combos[rand_combo_idx]

    all_combos = np.delete(all_combos, rand_combo_idx, 0)

    # print(all_combos)
    # print(rand_combo_idx)
    # print(selected_combo)

    # print('Selected Combo: ', selected_combo)
    # print('x1: ', x1)
    # print('x2: ', x2)

    # # Delete X1
    # index = np.argwhere(x1 == selected_combo[0])
    # print('X1: 1st Find: ', index)
    # if len(index) > 0:
    #   inlier_list_x1 = np.delete(x1, index[0], 0)
    #   print('X1: 1st Delete: ', inlier_list_x1)
    
    # index = np.argwhere(inlier_list_x1 == selected_combo[2])
    # print('X1: 2nd Find: ', index)
    # if len(index) > 0:
    #   inlier_list_x1 = np.delete(inlier_list_x1, index[0], 0)
    #   print('X1: 2nd Delete: ', inlier_list_x1)

    # # Delete X2
    # index = np.argwhere(x2 == selected_combo[1])
    # print('X2: 1st Find: ', index)
    # if len(index) > 0:
    #   inlier_list_x2 = np.delete(x2, index[0], 0)
    #   print('X2: 1st Delete: ', inlier_list_x1)

    # index = np.argwhere(inlier_list_x2 == selected_combo[3])
    # print('X2: 2nd Find: ', index)

    # if len(index) > 0:
    #   inlier_list_x2 = np.delete(inlier_list_x2, index[0], 0)
    #   print('X2: 2nd Delete: ', inlier_list_x1)

    print('Selected Combo: ', selected_combo)

    inlier_list_t1 = t1
    inlier_list_t2 = t2

    print('inlier t1', inlier_list_t1)
    print('inlier t2', inlier_list_t2)

    p1 = np.array([selected_combo[0], selected_combo[1]])
    p2 = np.array([selected_combo[2], selected_combo[3]])

    print('p1', p1)
    print('p2', p2)
    vec = normalize_list(np.subtract(p2, p1))

    tiled_origin = np.tile(p1, (inlier_list_t1.shape[0], 1))
    t1_sub = np.subtract(inlier_list_t1, tiled_origin[:, 0])
    t2_sub = np.subtract(inlier_list_t2, tiled_origin[:, 1])

    print('t1_sub ', t1_sub)
    print('t2_sub ', t2_sub)
    print('Tiled Origin', tiled_origin)

    tiled_vec = np.tile(vec, (inlier_list_t1.shape[0], 1))
    tiled_thresh = np.tile(thresh, (inlier_list_t1.shape[0]))

    print('Tiled Thresh', tiled_thresh)
    print('Tiled Vector', tiled_vec)

    dot_x = np.multiply(t1_sub, tiled_vec[:, 0])
    dot_y = np.multiply(t2_sub, tiled_vec[:, 1])
    dot = np.add(dot_x, dot_y)

    print('Dot', dot)

    t1_sub = np.multiply(t1_sub, t1_sub)
    t2_sub = np.multiply(t2_sub, t2_sub)
    dot = np.multiply(dot, dot)

    length = np.add(t1_sub, t2_sub)
    length = np.subtract(length, dot)

    bool_epsilon = np.less(length, 0)
    length[bool_epsilon] = 0

    print('Length Sq', length)    

    length = np.sqrt(length)

    print('Length', length)

    bool_thresh = np.less(length, tiled_thresh)
    num_inliers = np.sum(bool_thresh)

    if num_inliers < 2:
      print('Stuff Broke')
      return

    results.append(selected_combo)
    resultInliers.append(num_inliers)

    k += 1

  maxIndex = np.argmax(resultInliers)

  return results[maxIndex], resultInliers[maxIndex]

def ransac_est_homography(x1, y1, x2, y2, thresh):

  minH = None
  maxInliers = sys.float_info.min
  inlierIndices = []

  error = sys.float_info.max

  k = 0

  print(x1)
  print(x2)
  print(y1)
  print(y2)

  while(k < 1000):
    current_x1 = np.copy(x1)
    current_x2 = np.copy(x2)
    current_y1 = np.copy(y1)
    current_y2 = np.copy(y2)


    rand_idx = random.randrange(current_x1.shape[0])
    selected_u1x = current_x1[rand_idx]
    selected_u1y = current_y1[rand_idx]
    selected_v1x = current_x2[rand_idx]
    selected_v1y = current_y2[rand_idx]

    current_x1 = np.delete(current_x1, rand_idx, 0)
    current_x2 = np.delete(current_x2, rand_idx, 0)
    current_y1 = np.delete(current_y1, rand_idx, 0)
    current_y2 = np.delete(current_y2, rand_idx, 0)

    rand_idx = random.randrange(current_x1.shape[0])
    selected_u2x = current_x1[rand_idx]
    selected_u2y = current_y1[rand_idx]
    selected_v2x = current_x2[rand_idx]
    selected_v2y = current_y2[rand_idx]

    current_x1 = np.delete(current_x1, rand_idx, 0)
    current_x2 = np.delete(current_x2, rand_idx, 0)
    current_y1 = np.delete(current_y1, rand_idx, 0)
    current_y2 = np.delete(current_y2, rand_idx, 0)

    rand_idx = random.randrange(current_x1.shape[0])
    selected_u3x = current_x1[rand_idx]
    selected_u3y = current_y1[rand_idx]
    selected_v3x = current_x2[rand_idx]
    selected_v3y = current_y2[rand_idx]

    current_x1 = np.delete(current_x1, rand_idx, 0)
    current_x2 = np.delete(current_x2, rand_idx, 0)
    current_y1 = np.delete(current_y1, rand_idx, 0)
    current_y2 = np.delete(current_y2, rand_idx, 0)

    rand_idx = random.randrange(current_x1.shape[0])
    selected_u4x = current_x1[rand_idx]
    selected_u4y = current_y1[rand_idx]
    selected_v4x = current_x2[rand_idx]
    selected_v4y = current_y2[rand_idx]

    current_x1 = np.delete(current_x1, rand_idx, 0)
    current_x2 = np.delete(current_x2, rand_idx, 0)
    current_y1 = np.delete(current_y1, rand_idx, 0)
    current_y2 = np.delete(current_y2, rand_idx, 0)

    sourceX = np.array([selected_u1x, selected_u2x, selected_u3x, selected_u4x])
    sourceY = np.array([selected_u1y, selected_u2y, selected_u3y, selected_u4y])

    targetX = np.array([selected_v1x, selected_v2x, selected_v3x, selected_v4x])
    targetY = np.array([selected_v1y, selected_v2y, selected_v3y, selected_v4y])

    H = est_homography(sourceX, sourceY, targetX, targetY)

    currentIndices = []
    countInliers = 0
    currentError = 0

    for i in range(0, x1.shape[0]):
      uVec = np.array([x1[i], y1[i], 1])
      vVec = np.array([x2[i], y2[i], 1])

      vNew = np.matmul(H, uVec)
      
      vNew = vNew / vNew[2]

      vNew = vVec - vNew
      vNew = np.dot(vNew, vNew)
    
      if vNew < thresh:
        currentIndices.append(1)
        countInliers = countInliers + 1
        currentError = currentError + vNew
      else:
        currentIndices.append(0)

    

    if countInliers > maxInliers or (maxInliers == countInliers and currentError < error):
      maxInliers = countInliers
      minH = H
      inlierIndices = currentIndices
      error = currentError

    k = k + 1

  print('Selected H: ', minH)
  print('Inlier Indices: ', inlierIndices)
  print('Max Inliers: ', maxInliers)
  print('Error: ', error)

  return minH, inlierIndices