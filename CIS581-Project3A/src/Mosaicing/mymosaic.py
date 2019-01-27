'''
  File name: mymosaic.py
  Author:
  Date created:
'''

'''
  File clarification:
    Produce a mosaic by overlaying the pairwise aligned images to create the final mosaic image. If you want to implement
    imwarp (or similar function) by yourself, you should apply bilinear interpolation when you copy pixel values. 
    As a bonus, you can implement smooth image blending of the final mosaic.
    - Input imgLeft: M elements numpy array or list, each element is a input image.
    - Outpuy img_mosaic: H × W × 3 matrix representing the final mosaic image.
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import geometric_transform
from scipy import ndimage

from corner_detector import corner_detector
from anms import anms
from feat_desc import feat_desc
from feat_match import feat_match
from ransac_est_homography import ransac_est_homography
from interp2 import interp2

globalH = None

DEBUG_IMG = True

def shift_func(output_coords):
  global globalH

  inputValue = np.array([output_coords[0], output_coords[1], 1])

  value = np.matmul(globalH, inputValue)
  value = value / value[2]

  return (value[0], value[1])

def mymosaic(imgLeftPath, imgMiddlePath, imgRightPath):
  global globalH

  imgLeft = cv2.imread(imgLeftPath)
  imgMiddle = cv2.imread(imgMiddlePath)
  imgRight = cv2.imread(imgRightPath)

  leftColor = np.array(cv2.cvtColor(imgLeft, cv2.COLOR_BGR2RGB))
  middleColor = np.array(cv2.cvtColor(imgMiddle, cv2.COLOR_BGR2RGB))
  rightColor = np.array(cv2.cvtColor(imgRight, cv2.COLOR_BGR2RGB))

  featurePoints = 200

  print('Left Image')

  # Left Image
  leftGray = np.array(cv2.cvtColor(imgLeft,cv2.COLOR_BGR2GRAY))
  leftGray = np.pad(leftGray, ((0,0),(0,0)), 'constant', constant_values=(0, 0))

  leftCorners = corner_detector(leftGray) 
  leftX, leftY, rmax = anms(leftCorners, featurePoints, leftGray)
  leftDescs = feat_desc(leftGray, leftX, leftY)

  print('Left Feature Candidates: ', leftX.shape[0])
  print('Left Descriptors: ', leftDescs.shape[0])

  print('Middle Image')

  # Middle Image
  middleGray = np.array(cv2.cvtColor(imgMiddle,cv2.COLOR_BGR2GRAY))
  middleGray = np.pad(middleGray, ((0,0),(0,0)), 'constant', constant_values=(0, 0))

  middleCorners = corner_detector(middleGray) 
  middleX, middleY, rmax = anms(middleCorners, featurePoints, middleGray)
  middleDescs = feat_desc(middleGray, middleX, middleY)

  print('Middle Feature Candidates: ', middleX.shape[0])
  print('Middle Descriptors: ', middleDescs.shape[0])

  print('Right Image')

  # Right Image
  rightGray = np.array(cv2.cvtColor(imgRight,cv2.COLOR_BGR2GRAY))
  rightGray = np.pad(rightGray, ((0,0),(0,0)), 'constant', constant_values=(0, 0))

  rightCorners = corner_detector(rightGray) 
  rightX, rightY, rmax = anms(rightCorners, featurePoints, rightGray)
  rightDescs = feat_desc(rightGray, rightX, rightY)

  fig, axes = plt.subplots(1, 3, gridspec_kw = {'width_ratios':[leftColor.shape[1], middleColor.shape[1], rightGray.shape[1]]})
  axes[0].imshow(leftColor)
  axes[0].axis('off')
  axes[0].set_title('Left : [' + str(leftColor.shape[1]) + ', ' + str(leftColor.shape[0]) + ']')

  axes[1].imshow(middleColor)
  axes[1].axis('off')
  axes[1].set_title('Middle : [' + str(middleColor.shape[1]) + ', ' + str(middleColor.shape[0]) + ']')

  axes[2].imshow(middleColor)
  axes[2].axis('off')
  axes[2].set_title('Right : [' + str(rightGray.shape[1]) + ', ' + str(rightGray.shape[0]) + ']')

  fig.tight_layout()

  for i in range(0, leftX.shape[0]):
    circle = plt.Circle((leftY[i], leftX[i]), 50, color='r', linewidth=1, fill=False)
    axes[0].add_artist(circle)

  for i in range(0, middleX.shape[0]):
    circle = plt.Circle((middleY[i], middleX[i]), 50, color='r', linewidth=1, fill=False)
    axes[1].add_artist(circle)

  for i in range(0, rightX.shape[0]):
    circle = plt.Circle((rightY[i], rightX[i]), 50, color='r', linewidth=1, fill=False)
    axes[2].add_artist(circle)

  plt.show()

  print('Right Feature Candidates: ', rightX.shape[0])
  print('Right Descriptors: ', rightDescs.shape[0])

  print('Matching')

  mappingMR1 = feat_match(middleDescs, rightDescs)
  mappingMR2 = feat_match(rightDescs, middleDescs)

  print('Mapping Middle to Right: ', mappingMR1.shape[0])
  print('Mapping Right to Middle: ', mappingMR2.shape[0])

  mappingML1 = feat_match(middleDescs, leftDescs)
  mappingML2 = feat_match(leftDescs, middleDescs)

  print('Mapping Middle to Left: ', mappingML1.shape[0])
  print('Mapping Left to Middle: ', mappingML2.shape[0])


  mappingRUx = []
  mappingRUy = []
  mappingRVx = []
  mappingRVy = []

  # Map Middle and Right
  for i in range(0, mappingMR1.shape[0]):
    if (mappingMR1[i] != -1):
      if (mappingMR2[int(mappingMR1[i])] == i):
        mappingRUx.append(rightX[int(mappingMR1[i])])
        mappingRUy.append(rightY[int(mappingMR1[i])])

        mappingRVx.append(middleX[i])
        mappingRVy.append(middleY[i])

  mappingRUx = np.array(mappingRUx)
  mappingRUy = np.array(mappingRUy)
  mappingRVx = np.array(mappingRVx)
  mappingRVy = np.array(mappingRVy)

  print('Mapped Right X: ', mappingRUx)
  print('Mapped Right Y: ', mappingRUy)
  print('Mapped Middle X: ', mappingRVx)
  print('Mapped Middle Y: ', mappingRVy)

  mappingLUx = []
  mappingLUy = []
  mappingLVx = []
  mappingLVy = []

  # Map Middle and Left
  for i in range(0, mappingML1.shape[0]):
    if (mappingML1[i] != -1):
      if (mappingML2[int(mappingML1[i])] == i):
        mappingLUx.append(leftX[int(mappingML1[i])])
        mappingLUy.append(leftY[int(mappingML1[i])])

        mappingLVx.append(middleX[i])
        mappingLVy.append(middleY[i])

  mappingLUx = np.array(mappingLUx)
  mappingLUy = np.array(mappingLUy)
  mappingLVx = np.array(mappingLVx)
  mappingLVy = np.array(mappingLVy)

  print('Mapped Right X: ', mappingLUx)
  print('Mapped Right Y: ', mappingLUy)
  print('Mapped Middle X: ', mappingLVx)
  print('Mapped Middle Y: ', mappingLVy)

  mappedImageRMCheck = np.zeros((middleColor.shape[0], middleColor.shape[1] + rightColor.shape[1], 3))
  mappedImageRMCheck[0:, 0:middleColor.shape[1], :] = middleColor
  mappedImageRMCheck[0:, middleColor.shape[1]:middleColor.shape[1] + rightColor.shape[1], :] = rightColor

  mappedImageLMCheck = np.zeros((middleColor.shape[0], middleColor.shape[1] + leftColor.shape[1], 3))
  mappedImageLMCheck[0:, 0:leftColor.shape[1], :] = leftColor
  mappedImageLMCheck[0:, leftColor.shape[1]:leftColor.shape[1] + middleColor.shape[1], :] = middleColor

  # fig, axes = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[middleColor.shape[1], middleColor.shape[1]]})

  # axes[0].imshow(middleColor)
  # axes[0].axis('off')
  # axes[0].set_title('Middle : [' + str(middleColor.shape[1]) + ', ' + str(middleColor.shape[0]) + ']')

  # axes[1].imshow(rightColor)
  # axes[1].axis('off')
  # axes[1].set_title('Right : [' + str(rightColor.shape[1]) + ', ' + str(rightColor.shape[0]) + ']')

  # fig.tight_layout()

  # for i in range(0, mappingRVx.shape[0]):
  #   circle = plt.Circle((mappingRVy[i], mappingRVx[i]), 50, color='r', linewidth=1, fill=False)
  #   axes[0].add_artist(circle)

  # for i in range(0, mappingRUx.shape[0]):
  #   circle = plt.Circle((mappingRUy[i], mappingRUx[i]), 50, color='r', linewidth=1, fill=False)
  #   axes[1].add_artist(circle)

  # plt.show()

  # Show M <--- > R
  fig, axes = plt.subplots(1, 1, gridspec_kw = {'width_ratios':[mappedImageRMCheck.shape[1]]})

  axes.imshow(mappedImageRMCheck / 255)

  for i in range(0, mappingRUx.shape[0]):
    axes.plot([mappingRVy[i], middleGray.shape[1] + mappingRUy[i]], [mappingRVx[i], mappingRUx[i]], 'r')

    circle = plt.Circle((mappingRVy[i], mappingRVx[i]), 50, color='r', linewidth=1, fill=False)
    axes.add_artist(circle)

    circle = plt.Circle((middleGray.shape[1] + mappingRUy[i], mappingRUx[i]), 50, color='r', linewidth=1, fill=False)
    axes.add_artist(circle)

  plt.show()

  # Show L <--- > M
  fig, axes = plt.subplots(1, 1, gridspec_kw = {'width_ratios':[mappedImageLMCheck.shape[1]]})

  axes.imshow(mappedImageLMCheck / 255)

  for i in range(0, mappingLUx.shape[0]):
    axes.plot([leftColor.shape[1] + mappingLVy[i], mappingLUy[i]], [mappingLVx[i], mappingLUx[i]], 'r')

    circle = plt.Circle((leftColor.shape[1] + mappingLVy[i], mappingLVx[i]), 50, color='r', linewidth=1, fill=False)
    axes.add_artist(circle)

    circle = plt.Circle((mappingLUy[i], mappingLUx[i]), 50, color='r', linewidth=1, fill=False)
    axes.add_artist(circle)

  plt.show()

  print('Getting Hs')

  HRM, inlier_ind_rm = ransac_est_homography(mappingRUx, mappingRUy, mappingRVx, mappingRVy, 5)
  HLM, inlier_ind_lm = ransac_est_homography(mappingLUx, mappingLUy, mappingLVx, mappingLVy, 5)


  # Show M <--- > R
  fig, axes = plt.subplots(1, 1, gridspec_kw = {'width_ratios':[mappedImageRMCheck.shape[1]]})

  axes.imshow(mappedImageRMCheck / 255)

  for i in range(0, mappingRUx.shape[0]):
    col = 'b'
    if inlier_ind_rm[i] == 1:
      col = 'r'
      axes.plot([mappingRVy[i], middleGray.shape[1] + mappingRUy[i]], [mappingRVx[i], mappingRUx[i]], 'r')

    circle = plt.Circle((mappingRVy[i], mappingRVx[i]), 50, color=col, linewidth=1, fill=False)
    axes.add_artist(circle)

    circle = plt.Circle((middleGray.shape[1] + mappingRUy[i], mappingRUx[i]), 50, color=col, linewidth=1, fill=False)
    axes.add_artist(circle)

  plt.show()

  # Show L <--- > M
  fig, axes = plt.subplots(1, 1, gridspec_kw = {'width_ratios':[mappedImageLMCheck.shape[1]]})

  axes.imshow(mappedImageLMCheck / 255)

  for i in range(0, mappingLUx.shape[0]):
    col = 'b'
    if inlier_ind_lm[i] == 1:
      col = 'r'
      axes.plot([leftColor.shape[1] + mappingLVy[i], mappingLUy[i]], [mappingLVx[i], mappingLUx[i]], 'r')

    circle = plt.Circle((leftColor.shape[1] + mappingLVy[i], mappingLVx[i]), 50, color=col, linewidth=1, fill=False)
    axes.add_artist(circle)

    circle = plt.Circle((mappingLUy[i], mappingLUx[i]), 50, color=col, linewidth=1, fill=False)
    axes.add_artist(circle)

  plt.show()

  fig, axes = plt.subplots(1, 1, gridspec_kw = {'width_ratios':[middleColor.shape[1]]})
  axes.imshow(middleColor)
  axes.axis('off')
  axes.set_title('Right - Middle Mapping Check : [' + str(middleColor.shape[1]) + ', ' + str(middleColor.shape[0]) + ']')

  fig.tight_layout()

  for i in range(0, mappingRUx.shape[0]):
    pt = np.array([mappingRUx[i], mappingRUy[i], 1])
    worldPt = np.matmul(HRM, pt)
    worldPt = worldPt / worldPt[2]

    circle = plt.Circle((mappingRVy[i], mappingRVx[i]), 50, color='r', linewidth=1, fill=False)
    axes.add_artist(circle)

    circle = plt.Circle((worldPt[1], worldPt[0]), 50, color='b', linewidth=1, fill=False)
    axes.add_artist(circle)

  plt.show()

  fig, axes = plt.subplots(1, 1, gridspec_kw = {'width_ratios':[middleColor.shape[1]]})
  axes.imshow(middleColor)
  axes.axis('off')
  axes.set_title('Left - Middle Mapping Check : [' + str(middleColor.shape[1]) + ', ' + str(middleColor.shape[0]) + ']')
  fig.tight_layout()

  for i in range(0, mappingLUx.shape[0]):
    pt = np.array([mappingLUx[i], mappingLUy[i], 1])
    worldPt = np.matmul(HLM, pt)
    worldPt = worldPt / worldPt[2]

    circle = plt.Circle((mappingLVy[i], mappingLVx[i]), 50, color='r', linewidth=1, fill=False)
    axes.add_artist(circle)

    circle = plt.Circle((worldPt[1], worldPt[0]), 50, color='b', linewidth=1, fill=False)
    axes.add_artist(circle)

  plt.show()

  print('Transforming')

  R0 = np.array([0, 0, 1])
  R1 = np.array([0, rightGray.shape[1], 1])
  R2 = np.array([rightGray.shape[0], rightGray.shape[1], 1])
  R3 = np.array([rightGray.shape[0], 0, 1])
  worldR0 = np.matmul(HRM, R0)
  worldR0 = worldR0 / worldR0[2]

  worldR1 = np.matmul(HRM, R1)
  worldR1 = worldR1 / worldR1[2]

  worldR2 = np.matmul(HRM, R2)
  worldR2 = worldR2 / worldR2[2]

  worldR3 = np.matmul(HRM, R3)
  worldR3 = worldR3 / worldR3[2]


  L0 = np.array([0, 0, 1])
  L1 = np.array([0, leftGray.shape[1], 1])
  L2 = np.array([leftGray.shape[0], leftGray.shape[1], 1])
  L3 = np.array([leftGray.shape[0], 0, 1])
  worldL0 = np.matmul(HLM, L0)
  worldL0 = worldL0 / worldL0[2]

  worldL1 = np.matmul(HLM, L1)
  worldL1 = worldL1 / worldL1[2]

  worldL2 = np.matmul(HLM, L2)
  worldL2 = worldL2 / worldL2[2]

  worldL3 = np.matmul(HLM, L3)
  worldL3 = worldL3 / worldL3[2]

  # maxX = np.amax(np.array([worldR0[0], worldR1[0], worldR2[0], worldR3[0]]))
  # maxY = np.amax(np.array([worldR0[1], worldR1[1], worldR2[1], worldR3[1]]))

  globalH = np.linalg.inv(HRM)

  worldM0 = np.array([0, 0, 1])
  worldM1 = np.array([0, middleGray.shape[1], 1])
  worldM2 = np.array([middleGray.shape[0], middleGray.shape[1], 1])
  worldM3 = np.array([middleGray.shape[0], 0, 1])

  xPts = np.array([worldR0[0], worldR1[0], worldR2[0], worldR3[0], worldM0[0], worldM1[0], worldM2[0], worldM3[0], worldL0[0], worldL1[0], worldL2[0], worldL3[0]])
  yPts = np.array([worldR0[1], worldR1[1], worldR2[1], worldR3[1], worldM0[1], worldM1[1], worldM2[1], worldM3[1], worldL0[1], worldL1[1], worldL2[1], worldL3[1]])

  print('X PTS in Middle Image Space: ', xPts)
  print('Y PTS in Middle Image Space: ', yPts)

  xMax = int(xPts[np.argmax(xPts)])
  xMin = int(xPts[np.argmin(xPts)])
  yMax = int(yPts[np.argmax(yPts)])
  yMin = int(yPts[np.argmin(yPts)])

  print('Final Min Max X & Y Overall: ', xMin, yMin, xMax, yMax)

  # xNewImage = np.subtract(xPts , np.tile(xMin, xPts.shape[0]))
  # yNewImage = np.subtract(yPts , np.tile(yMin, yPts.shape[0]))

  resultImage = np.zeros((xMax - xMin + 1, yMax - yMin + 1, 3))
  resultImage1 = np.zeros((xMax - xMin + 1, yMax - yMin + 1, 3))
  resultImage2 = np.zeros((xMax - xMin + 1, yMax - yMin + 1, 3))
  resultImageMask = np.zeros((resultImage.shape[0], resultImage.shape[1]))
  resultImageMask.fill(False)

  resultYGrid, resultXGrid = np.meshgrid(np.arange(resultImage.shape[1]), np.arange(resultImage.shape[0]))
  resultXGrid = resultXGrid + 1
  resultYGrid = resultYGrid + 1

  offsetInNewSpace = np.array([xMin, yMin])

  print('Global Offset: ', offsetInNewSpace)

  # Paste Middle
  originMX = -offsetInNewSpace[0]
  originMY = -offsetInNewSpace[1]

  print('Offset for Middle: ', originMX, originMY)

  if DEBUG_IMG:
    resultImage[originMX:(originMX + middleGray.shape[0]), originMY:(originMY + middleGray.shape[1]), :] = 0.33 * middleColor
  else:
    resultImage[originMX:(originMX + middleGray.shape[0]), originMY:(originMY + middleGray.shape[1]), :] = middleColor

  #################################
  # RIGHT IMAGE
  #################################

  # Paste Right
  rightXValues = np.array([worldR0[0], worldR1[0], worldR2[0], worldR3[0]])
  rightYValues = np.array([worldR0[1], worldR1[1], worldR2[1], worldR3[1]])

  rxMax = int(rightXValues[np.argmax(rightXValues)])
  ryMax = int(rightYValues[np.argmax(rightYValues)])

  rxMin = int(rightXValues[np.argmin(rightXValues)])
  ryMin = int(rightYValues[np.argmin(rightYValues)])

  print('Bounds of R: ', rxMin, ryMin, rxMax, ryMax)

  xGridRight, yGridRight = np.meshgrid(np.arange(rxMin, rxMax + 1), np.arange(ryMin, ryMax + 1))
  xGridRight = xGridRight.flatten()
  yGridRight = yGridRight.flatten()
  stackedVecs = np.column_stack([xGridRight, yGridRight, np.ones(xGridRight.shape[0])])

  dotValues = np.dot(np.linalg.inv(HRM), stackedVecs.transpose())
  dotValues = dotValues.transpose()
  wMapped = dotValues[:,2]

  xRightMapped = dotValues[:,0] / wMapped
  yRightMapped = dotValues[:,1] / wMapped

  # Filter Grid and Valid Ranges
  xRightLogical = np.logical_and(np.greater_equal(xRightMapped, 0), np.less(xRightMapped, rightGray.shape[0]))
  yRightLogical = np.logical_and(np.greater_equal(yRightMapped, 0), np.less(yRightMapped, rightGray.shape[1]))
  rightValid = np.logical_and(xRightLogical, yRightLogical)

  xGridRightValid = xGridRight[rightValid].astype(np.int32) 
  yGridRightValid = yGridRight[rightValid].astype(np.int32)

  xGridRightValid = np.subtract(xGridRightValid, rxMin) 
  yGridRightValid = np.subtract(yGridRightValid, ryMin)

  xRightMapValid = xRightMapped[rightValid].astype(np.int32)
  yRightMapValid = yRightMapped[rightValid].astype(np.int32)

  print('stackedVecs', stackedVecs)
  print('xRightMapped', xRightMapped)
  print('yRightMapped', yRightMapped)

  warpedValuesR = interp2(rightColor[:,:,0], yRightMapped[rightValid], xRightMapped[rightValid])
  warpedValuesG = interp2(rightColor[:,:,1], yRightMapped[rightValid], xRightMapped[rightValid])
  warpedValuesB = interp2(rightColor[:,:,2], yRightMapped[rightValid], xRightMapped[rightValid])

  warpedRight = np.zeros((rxMax + 1 - rxMin, ryMax + 1 - ryMin, 3))
  warpedRightMask = np.zeros((rxMax + 1 - rxMin, ryMax + 1 - ryMin))
  warpedRightMask.fill(False)
  warpedRight[xGridRightValid, yGridRightValid, 0] = warpedValuesR
  warpedRight[xGridRightValid, yGridRightValid, 1] = warpedValuesG
  warpedRight[xGridRightValid, yGridRightValid, 2] = warpedValuesB
  warpedRightMask[xGridRightValid, yGridRightValid] = True

  rightYGrid, rightXGrid = np.meshgrid(np.arange(warpedRight.shape[1]), np.arange(warpedRight.shape[0]))
  rightXGrid = np.add(rightXGrid, 1)
  rightYGrid = np.add(rightYGrid, 1)

  print('warpedRight', warpedRight)

  fig, axes = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[middleColor.shape[1], rightGray.shape[1]]})
  axes[0].imshow(middleColor)
  axes[0].axis('off')
  axes[0].set_title('Middle : [' + str(middleColor.shape[1]) + ', ' + str(middleColor.shape[0]) + ']')

  axes[1].imshow(warpedRight / 255)
  axes[1].axis('off')
  axes[1].set_title('Right Warped : [' + str(warpedRight.shape[1]) + ', ' + str(warpedRight.shape[0]) + ']')

  fig.tight_layout()
  plt.show()

  originRX = rxMin - offsetInNewSpace[0]
  originRY = ryMin - offsetInNewSpace[1]

  print('Offset for Right: ', originRX, originRY)

  rightFullMask = np.copy(resultImageMask)
  rightFullMask[originRX:(originRX + warpedRight.shape[0]), originRY:(originRY + warpedRight.shape[1])] = warpedRightMask

  print(rightFullMask.shape)
  print(resultXGrid.shape)
  print(resultYGrid.shape)
  print(resultImage.shape)

  rightXIndices = np.multiply(resultXGrid, rightFullMask).astype(np.int32)
  rightYIndices = np.multiply(resultYGrid, rightFullMask).astype(np.int32)
  rightXIndices = rightXIndices[rightXIndices > 0]
  rightYIndices = rightYIndices[rightYIndices > 0]
  rightXIndices = rightXIndices - 1
  rightYIndices = rightYIndices - 1

  rightWarpedXIndices = np.multiply(rightXGrid, warpedRightMask).astype(np.int32)
  rightWarpedYIndices = np.multiply(rightYGrid, warpedRightMask).astype(np.int32)
  rightWarpedXIndices = rightWarpedXIndices[rightWarpedXIndices > 0]
  rightWarpedYIndices = rightWarpedYIndices[rightWarpedYIndices > 0]
  rightWarpedXIndices = rightWarpedXIndices - 1
  rightWarpedYIndices = rightWarpedYIndices - 1

  if DEBUG_IMG:
    resultImage1[rightXIndices, rightYIndices, 0] =  0.33 * warpedRight[rightWarpedXIndices, rightWarpedYIndices, 0]
    resultImage1[rightXIndices, rightYIndices, 1] =  0.33 * warpedRight[rightWarpedXIndices, rightWarpedYIndices, 1]
    resultImage1[rightXIndices, rightYIndices, 2] =  0.33 * warpedRight[rightWarpedXIndices, rightWarpedYIndices, 2]
  else:
    resultImage[rightXIndices, rightYIndices, 0] =  warpedRight[rightWarpedXIndices, rightWarpedYIndices, 0]
    resultImage[rightXIndices, rightYIndices, 1] =  warpedRight[rightWarpedXIndices, rightWarpedYIndices, 1]
    resultImage[rightXIndices, rightYIndices, 2] =  warpedRight[rightWarpedXIndices, rightWarpedYIndices, 2]

  #################################
  # LEFT IMAGE
  #################################

  # Paste Left
  rightXValues = np.array([worldL0[0], worldL1[0], worldL2[0], worldL3[0]])
  rightYValues = np.array([worldL0[1], worldL1[1], worldL2[1], worldL3[1]])

  lxMax = int(rightXValues[np.argmax(rightXValues)])
  lyMax = int(rightYValues[np.argmax(rightYValues)])

  lxMin = int(rightXValues[np.argmin(rightXValues)])
  lyMin = int(rightYValues[np.argmin(rightYValues)])

  print('Bounds of L: ', lxMin, lyMin, lxMax, lyMax)

  xGridRight, yGridRight = np.meshgrid(np.arange(lxMin, lxMax + 1), np.arange(lyMin, lyMax + 1))
  xGridRight = xGridRight.flatten()
  yGridRight = yGridRight.flatten()
  stackedVecs = np.column_stack([xGridRight, yGridRight, np.ones(xGridRight.shape[0])])

  dotValues = np.dot(np.linalg.inv(HLM), stackedVecs.transpose())
  dotValues = dotValues.transpose()
  wMapped = dotValues[:,2]

  xRightMapped = dotValues[:,0] / wMapped
  yRightMapped = dotValues[:,1] / wMapped

  # Filter Grid and Valid Ranges
  xRightLogical = np.logical_and(np.greater_equal(xRightMapped, 0), np.less(xRightMapped, leftGray.shape[0]))
  yRightLogical = np.logical_and(np.greater_equal(yRightMapped, 0), np.less(yRightMapped, leftGray.shape[1]))
  rightValid = np.logical_and(xRightLogical, yRightLogical)

  xGridRightValid = xGridRight[rightValid].astype(np.int32) 
  yGridRightValid = yGridRight[rightValid].astype(np.int32)

  xGridRightValid = np.subtract(xGridRightValid, lxMin) 
  yGridRightValid = np.subtract(yGridRightValid, lyMin)

  xLeftMapValid = xRightMapped[rightValid].astype(np.int32)
  yLeftMapValid = yRightMapped[rightValid].astype(np.int32)

  print('stackedVecs', stackedVecs)
  print('xRightMapped', xRightMapped)
  print('yRightMapped', yRightMapped)

  warpedValuesR = interp2(leftColor[:,:,0], yRightMapped[rightValid], xRightMapped[rightValid])
  warpedValuesG = interp2(leftColor[:,:,1], yRightMapped[rightValid], xRightMapped[rightValid])
  warpedValuesB = interp2(leftColor[:,:,2], yRightMapped[rightValid], xRightMapped[rightValid])

  warpedLeft = np.zeros((lxMax + 1 - lxMin, lyMax + 1 - lyMin, 3))
  warpedLeftMask = np.zeros((lxMax + 1 - lxMin, lyMax + 1 - lyMin))
  warpedLeftMask.fill(False)
  warpedLeft[xGridRightValid, yGridRightValid, 0] = warpedValuesR
  warpedLeft[xGridRightValid, yGridRightValid, 1] = warpedValuesG
  warpedLeft[xGridRightValid, yGridRightValid, 2] = warpedValuesB
  warpedLeftMask[xGridRightValid, yGridRightValid] = True

  leftYGrid, leftXGrid = np.meshgrid(np.arange(warpedLeft.shape[1]), np.arange(warpedLeft.shape[0]))
  leftXGrid = np.add(leftXGrid, 1)
  leftYGrid = np.add(leftYGrid, 1)

  print('warpedLeft', warpedLeft)

  fig, axes = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[middleColor.shape[1], leftGray.shape[1]]})
  axes[0].imshow(middleColor)
  axes[0].axis('off')
  axes[0].set_title('Middle : [' + str(middleColor.shape[1]) + ', ' + str(middleColor.shape[0]) + ']')

  axes[1].imshow(warpedLeft / 255)
  axes[1].axis('off')
  axes[1].set_title('Left Warped : [' + str(warpedLeft.shape[1]) + ', ' + str(warpedLeft.shape[0]) + ']')

  fig.tight_layout()
  plt.show()

  originLX = lxMin - offsetInNewSpace[0]
  originLY = lyMin - offsetInNewSpace[1]

  print('Offset for Left: ', originLX, originLY)

  leftFullMask = np.copy(resultImageMask)
  leftFullMask[originLX:(originLX + warpedLeft.shape[0]), originLY:(originLY + warpedLeft.shape[1])] = warpedLeftMask

  print(leftFullMask.shape)

  leftXIndices = np.multiply(resultXGrid, leftFullMask).astype(np.int32)
  leftYIndices = np.multiply(resultYGrid, leftFullMask).astype(np.int32)
  leftXIndices = leftXIndices[leftXIndices > 0]
  leftYIndices = leftYIndices[leftYIndices > 0]
  leftXIndices = leftXIndices - 1
  leftYIndices = leftYIndices - 1

  leftWarpedXIndices = np.multiply(leftXGrid, warpedLeftMask).astype(np.int32)
  leftWarpedYIndices = np.multiply(leftYGrid, warpedLeftMask).astype(np.int32)
  leftWarpedXIndices = leftWarpedXIndices[leftWarpedXIndices > 0]
  leftWarpedYIndices = leftWarpedYIndices[leftWarpedYIndices > 0]
  leftWarpedXIndices = leftWarpedXIndices - 1
  leftWarpedYIndices = leftWarpedYIndices - 1

  if DEBUG_IMG:
    resultImage2[leftXIndices, leftYIndices, 0] =  0.33 * warpedLeft[leftWarpedXIndices, leftWarpedYIndices, 0]
    resultImage2[leftXIndices, leftYIndices, 1] =  0.33 * warpedLeft[leftWarpedXIndices, leftWarpedYIndices, 1]
    resultImage2[leftXIndices, leftYIndices, 2] =  0.33 * warpedLeft[leftWarpedXIndices, leftWarpedYIndices, 2]
  else:
    resultImage[leftXIndices, leftYIndices, 0] =  warpedLeft[leftWarpedXIndices, leftWarpedYIndices, 0]
    resultImage[leftXIndices, leftYIndices, 1] =  warpedLeft[leftWarpedXIndices, leftWarpedYIndices, 1]
    resultImage[leftXIndices, leftYIndices, 2] =  warpedLeft[leftWarpedXIndices, leftWarpedYIndices, 2]

  if DEBUG_IMG:
    resultImage = resultImage + resultImage1 + resultImage2

  fig, axes = plt.subplots(1, 1, gridspec_kw = {'width_ratios':[resultImage.shape[1]]})
  axes.imshow(resultImage / 255)
  axes.axis('off')
  axes.set_title('Final Image : [' + str(resultImage.shape[1]) + ', ' + str(resultImage.shape[0]) + ']')

  circle = plt.Circle((worldR0[1] - offsetInNewSpace[1], worldR0[0] - offsetInNewSpace[0]), 50, color='r', linewidth=1, fill=False)
  axes.add_artist(circle)

  circle = plt.Circle((worldR1[1] - offsetInNewSpace[1], worldR1[0] - offsetInNewSpace[0]), 50, color='r', linewidth=1, fill=False)
  axes.add_artist(circle)

  circle = plt.Circle((worldR2[1] - offsetInNewSpace[1], worldR2[0] - offsetInNewSpace[0]), 50, color='r', linewidth=1, fill=False)
  axes.add_artist(circle)

  circle = plt.Circle((worldR3[1] - offsetInNewSpace[1], worldR3[0] - offsetInNewSpace[0]), 50, color='r', linewidth=1, fill=False)
  axes.add_artist(circle)

  circle = plt.Circle((worldL0[1] - offsetInNewSpace[1], worldL0[0] - offsetInNewSpace[0]), 50, color='g', linewidth=1, fill=False)
  axes.add_artist(circle)

  circle = plt.Circle((worldL1[1] - offsetInNewSpace[1], worldL1[0] - offsetInNewSpace[0]), 50, color='g', linewidth=1, fill=False)
  axes.add_artist(circle)

  circle = plt.Circle((worldL2[1] - offsetInNewSpace[1], worldL2[0] - offsetInNewSpace[0]), 50, color='g', linewidth=1, fill=False)
  axes.add_artist(circle)

  circle = plt.Circle((worldL3[1] - offsetInNewSpace[1], worldL3[0] - offsetInNewSpace[0]), 50, color='g', linewidth=1, fill=False)
  axes.add_artist(circle)

  circle = plt.Circle((originRY, originRX), 50, color='b', linewidth=1, fill=False)
  axes.add_artist(circle)

  circle = plt.Circle((originLY, originLX), 50, color='b', linewidth=1, fill=False)
  axes.add_artist(circle)

  fig.tight_layout()
  plt.show()

  plt.imshow(resultImage / 255)
  plt.show()


  # for i in range(0, x.shape[0]):
  #   cv2.circle(imgLeft, (x[i], y[i]), 20, (0,0,255))
  #   cv2.circle(imgLeft, (x[i], y[i]), 21, (0,0,255))

  # cv2.imshow('dst',imgLeft)

  # while(1):
  #   key = cv2.waitKey(33)
  #   if key == 27:
  #     cv2.destroyAllWindows()
  #     break

  return #img_mosaic


mymosaic('16L.jpg', '16M.jpg', '16R.jpg');
