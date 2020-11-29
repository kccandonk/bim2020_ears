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

from skimage.feature import hog
from PIL import Image
#im = Image.fromarray(A)

# initialization
image_height = 480
image_width = 480
window_size = 240
window_step = 6
ONE_HOT_ENCODING = True
SAVE_IMAGES = False
GET_LANDMARKS = False
GET_HOG_FEATURES = False
GET_HOG_IMAGES = False
GET_HOG_WINDOWS_FEATURES = False
SELECTED_LABELS = []
IMAGES_PER_LABEL = 100000 # CHANGE THIS LATER BC IDK 
OUTPUT_FOLDER_NAME = "ears_features"
OUTPUT_FOLDER_PATH = "/home/ubuntu/ears/bim2020_ears/aws_scripts/ears_features/"

from video_preprocessing import crop_dest 

train_dir = "/home/ubuntu/ears/PROCESSED/train/"
test_dir = "/home/ubuntu/ears/PROCESSED/test/"
processed_dir = "/home/ubuntu/ears/PROCESSED/"

# parse arguments and initialize variables:
parser = argparse.ArgumentParser()
parser.add_argument("-j", "--jpg", default="no", help="save images as .jpg files")
parser.add_argument("-l", "--landmarks", default="yes", help="extract Dlib Face landmarks")
parser.add_argument("-ho", "--hog", default="yes", help="extract HOG features")
parser.add_argument("-hw", "--hog_windows", default="yes", help="extract HOG features from a sliding window")
parser.add_argument("-hi", "--hog_images", default="no", help="extract HOG images")
parser.add_argument("-o", "--onehot", default="yes", help="one hot encoding")
#parser.add_argument("-e", "--expressions", default="0,1,2,3,4,5,6", help="choose the faciale expression you want to use: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral")
parser.add_argument("-e", "--expressions", default="0,3,4,6", help="choose the faciale expression you want to use: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral")
											#setting our default to 4 emotions: Angry, Happy, Sad, and Neutral
args = parser.parse_args()
if args.jpg == "yes":
    SAVE_IMAGES = True
if args.landmarks == "yes":
    GET_LANDMARKS = True
if args.hog == "yes":
    GET_HOG_FEATURES = True
if args.hog_windows == "yes":
    GET_HOG_WINDOWS_FEATURES = True
if args.hog_images == "yes":
    GET_HOG_IMAGES = True
if args.onehot == "yes":
    ONE_HOT_ENCODING = True
if args.expressions != "":
    expressions  = args.expressions.split(",")
    for i in range(0,len(expressions)):
        label = int(expressions[i])
        if (label >=0 and label<=6 ):
            SELECTED_LABELS.append(label)
if SELECTED_LABELS == []:
    SELECTED_LABELS = [0,3,4,6]
print( str(len(SELECTED_LABELS)) + " expressions")

# loading Dlib predictor and preparing arrays:
print( "preparing")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
original_labels = [0, 1, 2, 3, 4, 5, 6]
new_labels = list(set(original_labels) & set(SELECTED_LABELS))
nb_images_per_label = list(np.zeros(len(new_labels), 'uint8'))
try:
    os.makedirs(OUTPUT_FOLDER_NAME)
except OSError as e:
    if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
        pass
    else:
        raise

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

def sliding_hog_windows(image):
    hog_windows = []
    for y in range(0, image_height, window_step):
        for x in range(0, image_width, window_step):
            window = image[y:y+window_size, x:x+window_size]
            hog_windows.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),
                                            cells_per_block=(1, 1), visualise=False))
    return hog_windows


# print( "importing csv file")
# data = pd.read_csv('fer2013.csv')

# for category in data['Usage'].unique():
#     print( "converting set: " + category + "...")
#     # create folder
#     if not os.path.exists(category):
#         try:
#             os.makedirs(OUTPUT_FOLDER_NAME + '/' + category)
#         except OSError as e:
#             if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
#                pass
#             else:
#                 raise
    
#     # get samples and labels of the actual category
#     category_data = data[data['Usage'] == category]
#     samples = category_data['pixels'].values
#     labels = category_data['emotion'].values
    
#     # get images and extract features
#     images = []
#     labels_list = []
#     landmarks = []
#     hog_features = []
#     hog_images = []
#     for i in range(len(samples)):
#         try:
#             if labels[i] in SELECTED_LABELS and nb_images_per_label[get_new_label(labels[i])] < IMAGES_PER_LABEL:
#                 image = np.fromstring(samples[i], dtype=int, sep=" ").reshape((image_height, image_width))
#                 images.append(image)
#                 if SAVE_IMAGES:
#                     scipy.misc.imsave(category + '/' + str(i) + '.jpg', image)
#                 if GET_HOG_WINDOWS_FEATURES:
#                     features = sliding_hog_windows(image)
#                     f, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
#                                             cells_per_block=(1, 1), visualise=True)
#                     hog_features.append(features)
#                     if GET_HOG_IMAGES:
#                         hog_images.append(hog_image)
#                 elif GET_HOG_FEATURES:
#                     features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
#                                             cells_per_block=(1, 1), visualise=True)
#                     hog_features.append(features)
#                     if GET_HOG_IMAGES:
#                         hog_images.append(hog_image)
#                 if GET_LANDMARKS:
#                     scipy.misc.imsave('temp.jpg', image)
#                     image2 = cv2.imread('temp.jpg')
#                     face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
#                     face_landmarks = get_landmarks(image2, face_rects)
#                     landmarks.append(face_landmarks)            
#                 labels_list.append(get_new_label(labels[i], one_hot_encoding=ONE_HOT_ENCODING))
#                 nb_images_per_label[get_new_label(labels[i])] += 1
#         except Exception as e:
#             print( "error in image: " + str(i) + " - " + str(e))

#     np.save(OUTPUT_FOLDER_NAME + '/' + category + '/images.npy', images)
#     if ONE_HOT_ENCODING:
#         np.save(OUTPUT_FOLDER_NAME + '/' + category + '/labels.npy', labels_list)
#     else:
#         np.save(OUTPUT_FOLDER_NAME + '/' + category + '/labels.npy', labels_list)
#     if GET_LANDMARKS:
#         np.save(OUTPUT_FOLDER_NAME + '/' + category + '/landmarks.npy', landmarks)
#     if GET_HOG_FEATURES or GET_HOG_WINDOWS_FEATURES:
#         np.save(OUTPUT_FOLDER_NAME + '/' + category + '/hog_features.npy', hog_features)
#         if GET_HOG_IMAGES:
#             np.save(OUTPUT_FOLDER_NAME + '/' + category + '/hog_images.npy', hog_images)




#my train and test split: 

#extract all of the RAVDESS for ANGER, SAD, HAPPY, and NEUTRAL
#extract all of the BAUM1 for ANGER, SAD, HAPPY, and NEUTRAL
'''
for each dataset, allocate 80% as testing and 20% as training
'''

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
		emotion = ["unknown", 7]

	return (emot[0], emot[1])

#Saves the data into train and test directories as a flattened np array
def split_data_train_test():
	print("Splitting data into train and test.....")
	test_np_images = []
	test_np_image_labels = []
	train_np_images = []
	train_np_images_labels = []
	for emotion_folder in os.listdir(crop_dest):
		print("Parsing through : " + emotion_folder)
		emotion_name, emotion_label = get_label_from_folder(emotion_folder)

		emotion_folder = os.path.join(crop_dest, emotion_folder)
		folder_name = str(emotion_folder)
		number_videos = len(os.listdir(crop_dest))

		#Take train and test from each emotion folder
		folder_test_size = int(number_videos / 5)
		folder_train_size = number_videos - folder_test_size
		# train_dir = "/home/ubuntu/ears/PROCESSED/train/"
		# test_dir = "/home/ubuntu/ears/PROCESSED/test/"
		video_counter = 0
		print("Folder: " + folder_name)
		for video in os.listdir(emotion_folder):
			video_path = os.path.join(emotion_folder, video)
			os.chdir(video_path)
			for frame in os.listdir(video_path):
				cropped_frame = np.load(frame)
				flat_frame = cropped_frame.flat
				flat_frame = np.asarray(flat_frame)
				name = emotion_folder + "_" + video + "_" + frame[:-8]
				#flat.shape gives you shape of (691200,)
				if video_counter < folder_test_size: 
					# flat_path = os.path.join(test_dir, name)
					# name = category
					# np.save(new_crop_filepath, flat_frame)
					test_np_images.append(flat_frame)
					test_np_image_labels.append(emotion_label)
					#test_np_image_labels.append([emotion_name, emotion_label])

				else: 
					# flat_path = os.path.join(train_dir, name)
					# np.save(new_crop_filepath, flat_frame)
					train_np_images.append(flat_frame)
					train_np_image_labels.append(emotion_label)
					#train_np_image_labels.append([emotion_name, emotion_label])
			video_counter = video_counter + 1

	print("___Done splitting!___")
	return train_np_images, train_np_images_labels, test_np_images, test_np_image_labels


train_np_images, train_np_images_labels, test_np_images, test_np_images_labels = split_data_train_test()

#do for train and test

# get samples and labels of the actual category (train or test)

train_samples = train_np_images #category_data['pixels'].values
test_samples = test_np_images  
train_labels = train_np_images #labels = category_data['emotion'].values
test_labels = test_np_images

# get images and extract features
train_landmarks = []
test_landmarks = []

train_hog_features = []
train_hog_images = []

test_hog_features = []
test_hog_images = []

train_images = []
test_images = []

train_labels_list = []
test_labels_list = []

#_________________train category_____________-
print("TRAIN CATEGORY")
category = "train"
for i in range(len(train_samples)):
	try:
		if train_labels[i] in SELECTED_LABELS and nb_images_per_label[get_new_label(train_labels[i])] < IMAGES_PER_LABEL:
			image = np.fromstring(train_samples[i], dtype=int, sep=" ").reshape((image_height, image_width))
			train_images.append(image)
			# if SAVE_IMAGES:
			# scipy.misc.imsave(category + '/' + str(i) + '.jpg', image)
			if GET_HOG_WINDOWS_FEATURES:
				features = sliding_hog_windows(image)
				f, train_hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
										cells_per_block=(1, 1), visualise=True)
				train_hog_features.append(features)
				if GET_HOG_IMAGES:
					train_hog_images.append(hog_image)
			elif GET_HOG_FEATURES:
				features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
										cells_per_block=(1, 1), visualise=True)
				train_hog_features.append(features)
				if GET_HOG_IMAGES:
					train_hog_images.append(hog_image)
			if GET_LANDMARKS:
				scipy.misc.imsave('temp.jpg', image)
				image2 = cv2.imread('temp.jpg')
				face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
				face_landmarks = get_landmarks(image2, face_rects)
				train_landmarks.append(face_landmarks)            
			train_labels_list.append(get_new_label(train_labels[i], one_hot_encoding=ONE_HOT_ENCODING))
			nb_images_per_label[get_new_label(train_labels[i])] += 1
	except Exception as e:
		print( "error in image: " + str(i) + " - " + str(e))

SAVE_PATH = os.path.join(OUTPUT_FOLDER_PATH, category)

np.save(os.path.join(SAVE_PATH, 'images.npy'), train_images)
if ONE_HOT_ENCODING:
	np.save(os.path.join(SAVE_PATH, 'labels.npy'), train_labels_list)
else:
	np.save(os.path.join(SAVE_PATH,'labels.npy'), train_labels_list)
if GET_LANDMARKS:
	np.save(os.path.join(SAVE_PATH,'landmarks.npy'),  train_landmarks)
if GET_HOG_FEATURES or GET_HOG_WINDOWS_FEATURES:
	np.save(os.path.join(SAVE_PATH, 'hog_features.npy'), train_hog_features)
	if GET_HOG_IMAGES:
		np.save(os.path.join(SAVE_PATH, 'hog_images.npy'), train_hog_images)


#_________________test category_____________-
print("TEST CATEGORY")
category = "test"
for i in range(len(test_samples)):
	try:
		if test_labels[i] in SELECTED_LABELS and nb_images_per_label[get_new_label(test_labels[i])] < IMAGES_PER_LABEL:
			image = np.fromstring(test_samples[i], dtype=int, sep=" ").reshape((image_height, image_width))
			test_images.append(image)
			# if SAVE_IMAGES:
			# scipy.misc.imsave(category + '/' + str(i) + '.jpg', image)
			if GET_HOG_WINDOWS_FEATURES:
				features = sliding_hog_windows(image)
				f, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
										cells_per_block=(1, 1), visualise=True)
				test_hog_features.append(features)
				if GET_HOG_IMAGES:
					test_hog_images.append(hog_image)
			elif GET_HOG_FEATURES:
				features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
										cells_per_block=(1, 1), visualise=True)
				test_hog_features.append(features)
				if GET_HOG_IMAGES:
					test_hog_images.append(hog_image)
			if GET_LANDMARKS:
				scipy.misc.imsave('temp.jpg', image)
				image2 = cv2.imread('temp.jpg')
				face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
				face_landmarks = get_landmarks(image2, face_rects)
				test_landmarks.append(face_landmarks)            
			test_labels_list.append(get_new_label(test_labels[i], one_hot_encoding=ONE_HOT_ENCODING))
			nb_images_per_label[get_new_label(test_labels[i])] += 1
	except Exception as e:
		print( "error in image: " + str(i) + " - " + str(e))

SAVE_PATH = os.path.join(OUTPUT_FOLDER_PATH, category)

np.save(os.path.join(SAVE_PATH, 'images.npy'), test_images)
if ONE_HOT_ENCODING:
	np.save(os.path.join(SAVE_PATH, 'labels.npy'), test_labels_list)
else:
	np.save(os.path.join(SAVE_PATH,'labels.npy'), test_labels_list)
if GET_LANDMARKS:
	np.save(os.path.join(SAVE_PATH,'landmarks.npy'),  test_landmarks)
if GET_HOG_FEATURES or GET_HOG_WINDOWS_FEATURES:
	np.save(os.path.join(SAVE_PATH, 'hog_features.npy'), test_hog_features)
	if GET_HOG_IMAGES:
		np.save(os.path.join(SAVE_PATH, 'hog_images.npy'), test_hog_images)





