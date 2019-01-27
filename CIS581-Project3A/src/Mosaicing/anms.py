'''
  File name: anms.py
  Author:
  Date created:
'''

'''
  File clarification:
    Implement Adaptive Non-Maximal Suppression. The goal is to create an uniformly distributed 
    points given the number of feature desired:
    - Input cimg: H × W matrix representing the corner metric matrix.
    - Input max_pts: the number of corners desired.
    - Outpuy x: N × 1 vector representing the column coordinates of corners.
    - Output y: N × 1 vector representing the row coordinates of corners.
    - Output rmax: suppression radius used to get max pts corners.
'''

import numpy as np
import cv2

def getkey(item):
  return item[2]

DEBUG = True

PRINT_LOGICAL_PATCH = False

def anms(cimg, max_pts, img):
  greater_threshold_factor = 0.9

  good_corners_map = cimg > (0.01 * cimg.max())
  good_corners = cimg[good_corners_map]

  # good_corners = good_corners[:10]

  # print(good_corners)
  # print('MAX: ', cimg.max())
  # return


  # indices of the image pixels
  nc_img = cimg.shape[1]
  nr_img = cimg.shape[0]
  Y_img, X_img = np.meshgrid(np.arange(nc_img), np.arange(nr_img))
  X_img = np.multiply(X_img + 1, good_corners_map) - 1
  Y_img = np.multiply(Y_img + 1, good_corners_map) - 1
  corner_indices_x = X_img[good_corners_map]
  corner_indices_y = Y_img[good_corners_map]

  # patch indices relative to image pixels
  patch_size_x = 40
  patch_size_y = 40
  X_patch, Y_patch = np.meshgrid(np.arange(-patch_size_x / 2, patch_size_x / 2 + 1, 1), np.arange(-patch_size_y / 2, patch_size_y/ 2 + 1, 1))

  x = np.tile(X_patch, (good_corners.shape[0], 1))
  y = np.tile(Y_patch, (good_corners.shape[0], 1))

  for i in range(0, good_corners.shape[0]):
    x[(i * (patch_size_x + 1)):((i + 1) * (patch_size_x + 1)), :] += corner_indices_x[i]
    y[(i * (patch_size_y + 1)):((i + 1) * (patch_size_y + 1)), :] += corner_indices_y[i]

  x = np.clip(x, 0, nr_img - 1).astype(np.int32)
  y = np.clip(y, 0, nc_img - 1).astype(np.int32)

  # print('Good Corner X', x)
  # print('Good Corner Y', y)

  # img[y, x] = [0,0,255]

  # cv2.imshow('dst', img)
  # while(1):
  #  key = cv2.waitKey(33)
  #  if key == 27:
  #    cv2.destroyAllWindows()
  #    break


  # getting the distance of each patch pixel to the pixel in consideration
  distance = np.sqrt((X_patch * X_patch) + (Y_patch * Y_patch))
  distance_tiled = np.tile(distance, (good_corners.shape[0], 1))

  # corner magnitudes in local patch
  mag_patch = cimg[x, y]

  logical_patch = np.zeros(mag_patch.shape)

  # corner magnitudes relative to the pixel magnitude in consideration
  for i in range(0, good_corners.shape[0]):
    mag_patch[(i * (patch_size_x + 1)):((i + 1) * (patch_size_x + 1)), :] /= np.full((patch_size_x + 1, patch_size_y + 1), good_corners[i])

    logical_patch[(i * (patch_size_x + 1)):((i + 1) * (patch_size_x + 1)), :] = np.greater(mag_patch[(i * (patch_size_x + 1)):((i + 1) * (patch_size_x + 1)), :], 10.0 / 9.0)


  if PRINT_LOGICAL_PATCH:
    for i in range(0, good_corners.shape[0]):
      bool_patch = logical_patch[(i * (patch_size_x + 1)):((i + 1) * (patch_size_x + 1)), :]
      x_patch = x[(i * (patch_size_x + 1)):((i + 1) * (patch_size_x + 1)), :]
      y_patch = y[(i * (patch_size_x + 1)):((i + 1) * (patch_size_x + 1)), :]
      x_test = x_patch[bool_patch.astype(np.bool)].astype(np.int32)
      y_test = y_patch[bool_patch.astype(np.bool)].astype(np.int32)

      img[y_test, x_test] = [0,0, (50 * i) % 256]

    print(x_test)
    print(y_test)

    cv2.imshow('dst', img)
    while(1):
      key = cv2.waitKey(33)
      if key == 27:
        cv2.destroyAllWindows()
        break

  # only keep relative magnitudes that are larger (greater than zero)
  #logical_pos = np.greater(mag_patch, 0.0)
  # only keep relative magnitudes that are less than or equal 0.9 greater than this value
  #logical_less = np.less(mag_patch, 0.9)
  # cull mag_patch values outside of desirable range
  #logical_range = np.logical_and(logical_pos, logical_less)
  #mag_patch = mag_patch * logical_range

  # find the minimum distance of a local pixel with a greater corner mag
  logical_distance = distance_tiled * logical_patch

  logical_distance[logical_distance == 0] += (patch_size_x + 1) * (patch_size_x + 1) + 1

  x_result = []
  y_result = []
  r_result = []

  for i in range(0, good_corners.shape[0]):
    bool_part = logical_distance[(i * (patch_size_x + 1)):((i + 1) * (patch_size_x + 1)), :]
    # distance_part = distance_tiled[(i * (patch_size_x + 1)):((i + 1) * (patch_size_x + 1)), :]

    #print('Boolean Patch Arg Min: ', bool_part.argmin())
    #print('Current Patch: ', corner_indices_x[i], corner_indices_y[i])


    min_distance_coord = np.unravel_index(bool_part.argmin(), bool_part.shape)
    min_coord_x = min_distance_coord[0] + (i * (patch_size_x + 1))
    min_coord_y = min_distance_coord[1]

    if (bool_part[min_distance_coord] == (patch_size_x + 1) * (patch_size_x + 1) + 1):
        continue

    #print('Value in Boolean Patch: ', bool_part[min_distance_coord])
    min_distance = distance_tiled[min_coord_x, min_coord_y]
    # print(logical_distance)

    if(min_distance > 0.0):
      # get pixel coordinates of min value
      x_chosen = x[min_coord_x, min_coord_y]
      y_chosen = y[min_coord_x, min_coord_y]

      #print('Query Point: ', x_chosen, y_chosen)

      if(x_chosen not in x_result and y_chosen not in y_result):
        # print('Inserted Point: ', x_chosen, y_chosen)
        x_result.append(x_chosen)
        y_result.append(y_chosen)
        r_result.append(min_distance)


  indices = np.array(r_result).argsort()

  x_sorted = np.array(x_result)
  y_sorted = np.array(y_result)
  r_sorted = np.array(r_result)

  x_sorted = x_sorted[indices]
  y_sorted = y_sorted[indices]
  r_sorted = r_sorted[indices]


  return x_sorted[-max_pts:], y_sorted[-max_pts:], r_sorted[-max_pts:]
