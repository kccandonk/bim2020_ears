#!/usr/bin/env python

import numpy as np
import pandas as pd
import math 
import re
import os
import os.path
from os import path
import numpy as np
import pandas as pd
import argparse
import errno
import cv2

baum_dim = [854,480]
rav_dim = [1280,720]

def find_face_center(top, right, bottom, left):

	center_dx = (right - left)/2 + left
	center_dy = (top - bottom)/2 + bottom

	return (int(center_dx), int(center_dy))

def crop_pic(face_img, dim, crop_size, center):

	crop_mid = crop_size / 2
	cX, cY = center

	#BAUM dataset has dY fixed at 480
	if dim == baum_dim: 
		startY = 0
		endY = crop_size
		#determine the X
		#TOO FAR LEFT
		if cX < crop_mid:
			c=1
			startX = 0
			endX = crop_size
		#TOO FAR RIGHT
		elif dim[0] - cX < crop_mid:
			c=2
			startX = dim[0] - crop_size
			endX = dim[0]
		else: 
			c=3
			startX = cX - crop_mid
			endX = cX + crop_mid
		#print("case + " + str(c))
	else: 

		#default values
		startX = cX - crop_mid
		startY = cY - crop_mid
		endX = cX + crop_mid
		endY = cY + crop_mid

		#TOP LEFT CORNER / x and y too small
		if cX < crop_mid and cY < crop_mid:
			startX = 0
			startY = 0
			endX = crop_size
			endY = crop_size
		#BOT RIGHT CORNER / x and y too big
		elif dim[0] - cX < crop_mid and dim[1] - cY < crop_mid:
			startX = dim[0] - crop_size
			startY = dim[1] - crop_size
			endX = dim[0]
			endY = dim[1]
		#TOP RIGHT CORNER / x too big and y too small
		elif dim[0] - cX < crop_mid and cY < crop_mid:
			startX = dim[0] - crop_size
			startY = 0
			endX = dim[0]
			endY = crop_size
		#BOT LEFT CORNER / x too small and y too big
		elif dim[0] - cX < crop_mid and dim[1] - cY < crop_mid:
			startX = dim[0] - crop_size
			startY = dim[0] - crop_size
			endX = dim[0]
			endY = dim[0]
		#TOO FAR LEFT
		elif cX < crop_mid:
			startX = 0
			endX = crop_size
		#TOO FAR RIGHT
		elif dim[0] - cX < crop_mid:
			startX = dim[0] - crop_size
			endX = dim[0]
		#TOO FAR UP
		elif cY < crop_mid:
			startY = 0
			endY = crop_size
		#TOO FAR DOWN
		elif dim[1] - cY < crop_mid:
			startY = dim[0] - crop_size
			endY = dim[0]

	cropped = face_img[int(startY):int(endY), int(startX):int(endX)]

	return cropped
