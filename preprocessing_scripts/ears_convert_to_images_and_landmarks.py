#!/usr/bin/env python

# modified from: https://github.com/amineHorseman/facial-expression-recognition-using-cnn/blob/master/convert_fer2013_to_images_and_landmarks.py

import numpy as np
import pandas as pd
import os
import argparse
import errno
import scipy.misc
import dlib
import cv2

import imageio

from PIL import Image

# initialization
ONE_HOT_ENCODING = True
SELECTED_LABELS = [0,3,4,6]
OUTPUT_FOLDER_NAME = "ears_features"

# Path based on my local directory
OUTPUT_FOLDER_PATH = "/Users/Kaitlynn/Desktop/CPSC_459/bim2020_ears/ears_features/"
SCRIPTS = "/Users/Kaitlynn/Desktop/CPSC_459/bim2020_ears/aws_scripts"
crop_dest = "/Users/Kaitlynn/Desktop/CPSC_459/bim2020_ears/DATA/cropped_frames_by_emotion/"
TEMP_PATH = os.path.join(SCRIPTS, "temp")

# preparing arrays:
#print( "preparing")
original_labels = [0, 1, 2, 3, 4, 5, 6]
new_labels = list(set(original_labels) & set(SELECTED_LABELS))
nb_images_per_label = list(np.zeros(len(new_labels), 'uint8'))

def return_frame_landmarks(cropped_image):   
	os.chdir('/Users/Kaitlynn/Desktop/CPSC_459/bim2020_ears/temp_files')
	CWD = os.getcwd()
	imageio.imwrite('temp.jpg', cropped_image)
	image2 = cv2.imread('temp.jpg')
	face_rects = [dlib.rectangle()]
	face_landmarks = get_landmarks(image2, face_rects)
	os.chdir('/Users/Kaitlynn/Desktop/CPSC_459/bim2020_ears/')
	return face_landmarks

def get_landmarks(image, rects):
    # this function have been copied from http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

def get_new_label(label, one_hot_encoding=False):
    if one_hot_encoding:
        new_label = new_labels.index(label)
        label = list(np.zeros(len(new_labels), 'uint8'))
        label[new_label] = 1
        return label
    else:
        return new_labels.index(label)

def get_label_from_folder(folder_name):
	#setting our default to 4 emotions: Angry, Happy, Sad, and Neutral
	# 0 3 4 6
	emotion = folder_name[2:]

	if emotion == "Anger":
		emot = ["Angry", 0]
	elif emotion == "Happiness":
		emot = ["Happy", 3]
	elif emotion == "Sadness":
		emot = ["Sad", 4]
	elif emotion == "Neutral":
		emot = ["Neutral", 6]
	else:
		emot = ["unknown", 7]

	return (emot[0], emot[1])

def get_full_data(): #Saves the data into a big list, one for images, labels, and landmarks respectively
	print("making dataset into one big array.....")

	full_images = []
	full_labels = []
	full_landmarks = []

	for emotion_folder in os.listdir(crop_dest):
		print("Parsing through : " + emotion_folder)
		emotion_name, emotion_label = get_label_from_folder(emotion_folder)
		emotion_path = os.path.join(crop_dest, emotion_folder)
		for video in os.listdir(emotion_path):
			#print("	for each video folder .." + video)
			video_path = os.path.join(emotion_path, video)
			#print("  Video path:"+video_path)
			os.chdir(video_path)
			for frame in os.listdir(video_path):
				frame_path = os.path.join(video_path, frame)
				#print(frame_path)
				cropped_frame = np.load(frame_path)
				if cropped_frame.size != 0:
					loc, temp, img = face_detector(cropped_frame)
					frame_landmarks = return_frame_landmarks(cropped_frame)
					name = emotion_folder + "_" + video + "_" + frame[:-8]
					full_images.append(temp)      
					full_labels.append(get_new_label(emotion_label, one_hot_encoding=ONE_HOT_ENCODING))
					full_landmarks.append(frame_landmarks)      

	print("______Done!_____")
	print ("full_images | full_labels ")
	print(str(len(full_images)) + "| " + str(len(full_labels)) + "| " + str(len(full_landmarks)))
	
	return full_images, full_labels, full_landmarks

