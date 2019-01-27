'''
  File name: corner_detector.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line, 
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''
import cv2
import numpy as np

def corner_detector(gray):
  dst = cv2.cornerHarris(gray,2,3,0.04)

  #result is dilated for marking the corners, not important
  dst = cv2.dilate(dst,None)

  # Threshold for an optimal value, it may vary depending on the image.
  # img[dst>0.01*dst.max()]=[0,0,255]

  # cv2.imshow('dst',img)
  # if cv2.waitKey(0) & 0xff == 27:
  #   cv2.destroyAllWindows()
  return dst
