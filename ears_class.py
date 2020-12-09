#!/usr/bin/env python
# Script for EARS system
# emotion detection adapted from: https://github.com/priya-dwivedi/face_and_emotion_detection
# model adapted from: https://github.com/kumarnikhil936/Facial-Emotion-Recognition

import time
import datetime
import pyaudio
import wave
import webrtcvad
import sys
import os
import dlib

import imageio
#from PIL import Image 

import pygame

from collections import Counter

import tensorflow as tf

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array


# define RGB for white
WHITE = (255, 255, 255)
# set width and height for display
X = 600
Y = 600
START_THRESHOLD = 3
SILENCE_THRESHOLD = 3

CHUNK = 480
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 0.5
BATCH_SIZE = 16
CWD = ""
MAIN_DIR = os.getcwd()

FACE_CLASSIFIER = cv2.CascadeClassifier('./Haarcascades/haarcascade_frontalface_default.xml')

#CLASS_LABELS = {0: 'mad', 1: 'NA', 2: 'NA', 3: 'happy', 4: 'sad', 5: 'NA', 6: 'neutral'}
#CLASS_LABELS = {0: 'mad', 3: 'happy', 4: 'sad', 6: 'neutral'}
CLASS_LABELS = {0: 'mad', 1: 'happy', 2: 'sad', 3: 'neutral'} #because of one hot incoding
#ngry, Happy, Sad, and Neutral

# loading Dlib predictor and preparing arrays:
print( "preparing")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def return_frame_landmarks(cropped_image):
	os.chdir('./temp_files')
	CWD = os.getcwd()
	imageio.imwrite('temp.jpg', cropped_image)
	image2 = cv2.imread('temp.jpg')
	face_rects = [dlib.rectangle()]
	face_landmarks = get_landmarks(image2, face_rects)
	os.chdir(MAIN_DIR)
	return face_landmarks

def get_landmarks(image, rects):
    # this function have been copied from http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

def face_detector(img):
	    # Convert image to grayscale
	    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	    faces = FACE_CLASSIFIER.detectMultiScale(gray, 1.3, 5)
	    if faces is ():
	        return (0,0,0,0), np.zeros((48,48), np.uint8), img
	    
	    for (x,y,w,h) in faces:
	        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	        roi_gray = gray[y:y+h, x:x+w]
	        #roi_img = img[y:y+h, x:x+w]

	    try:
	        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
	        #roi_img = cv2.resize(roi_img, (48, 48), interpolation = cv2.INTER_AREA)
	    except:
	        return (x,w,y,h), np.zeros((48,48), np.uint8), img
	    return (x,w,y,h), roi_gray, img
	    #return (x,w,y,h), roi_img, img

class EARS(object):

	def __init__(self):
		self.startTime = None
		self.emotionTrajectory = []
		self.currentEmotion = "neutral"
		self.audio_counter = 0
		self.hasSpoken = False
		self.runningSilence = 0
		self.keepGoing = True
		#self.classifier = tf.keras.models.load_model('./emotion_detector_models/_mini_xception.100_0.65.hdf5') #model_v6_23.hdf5')
		self.classifier = tf.keras.models.load_model('./emotion_detector_models/ears_model_full_model_big_dataset_12-08-2020-15-30.hdf5') #last from drive')
		self.cap = cv2.VideoCapture(0)

	def run(self):
		pygame.init()
		display_surface = pygame.display.set_mode((X,Y))
		pygame.display.set_caption('EARS')
		prompt = pygame.image.load('images/EARS_initial.jpg')
		prompt_x = prompt.get_width()
		prompt_y = prompt.get_height()
		prompt_center_x = X/2 - prompt_x/2
		prompt_center_y = Y/2 - prompt_y/2
		display_surface.fill(WHITE)
		display_surface.blit(prompt,(prompt_center_x,prompt_center_y))
		pygame.display.update()
		self.startTime = time.time()
		while self.keepGoing:
			self.audio_check()
			self.predict_emotions()
			image = self.responsePolicy()
			im_x = image.get_width()
			im_y = image.get_height()
			center_x = X/2 - im_x/2
			center_y = Y/2 - im_y/2
			display_surface.fill(WHITE)
			display_surface.blit(image,(center_x,center_y))
			pygame.display.update()
			if self.keepGoing == False:
				self.cap.release()
				cv2.destroyAllWindows()
				time.sleep(2)

	def responsePolicy(self):
		elapsed_time = time.time()-self.startTime
		os.chdir(MAIN_DIR)
		if elapsed_time < START_THRESHOLD or self.hasSpoken==False:
			image = pygame.image.load('images/EARS_initial.jpg')
		elif self.runningSilence < SILENCE_THRESHOLD:
			image = pygame.image.load('render_faces/emotion_faces/'+self.currentEmotion+'.jpg')
		else:
			response_filename = self.retrieveResponse()
			image = pygame.image.load('responses/'+response_filename+'.jpg')
			self.keepGoing = False
		return image

	def retrieveResponse(self):
		counts = Counter()
		counts.update(self.emotionTrajectory)
		most_common = counts.most_common(1)[0][0]
		end_counts = Counter()
		end_counts.update(self.emotionTrajectory[int(len(self.emotionTrajectory)/4)*3:])
		most_common_end = end_counts.most_common(1)[0][0]
		response_filename = most_common+"_endon_"+most_common_end
		return response_filename

	def audio_check(self):
		
		p = pyaudio.PyAudio()
		vad = webrtcvad.Vad(3)

		stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

		print("* listening")

		frames = []
		silence_or_speech = []

		for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
			data = stream.read(CHUNK)
			frames.append(data)
			silence_or_speech.append(vad.is_speech(data,RATE))

		if True in silence_or_speech:
			self.hasSpoken=True

		percent_speaking = float(sum(silence_or_speech))/len(silence_or_speech)

		if percent_speaking >= 0.25 or self.hasSpoken==False:
			self.runningSilence = 0
		else:
			self.runningSilence += RECORD_SECONDS

		stream.stop_stream()
		stream.close()
		p.terminate()

		self.audio_counter = self.audio_counter+1

	def predict_emotions(self):
		print("predict")
		ret, frame = self.cap.read()
		rect, face, image = face_detector(frame) #image is the same as frame
		#Get landmarks for the specific frame
		#frame_landmarks = return_frame_landmarks(face)
		frame_landmarks = return_frame_landmarks(frame)
		if np.sum([face]) != 0.0:
			roi = face.astype("float") / 255.0
			roi = img_to_array(roi)
			roi = np.expand_dims(roi, axis=0)
			frame_landmarks = np.expand_dims(frame_landmarks, axis=0)
			# print("HERE IS WHAT ROI LOOKS LIKE")
			# print(roi.shape)
			# print("HERE IS WHAT LANDMARKS LOOKS LIKE")
			# print(frame_landmarks.shape)
			# make a prediction on the ROI and landmarks, then lookup the class
			preds = self.classifier.predict([roi, frame_landmarks], batch_size=BATCH_SIZE)
			#angry, Happy, Sad, and Neutral
			# preds[1] = 0
			# preds[2] = 0
			# preds[5] = 0
			preds = np.round(preds[0], 3)
			print(preds)
			# print("PREDS shape: ")
			# print(preds[0][0])
			# print(preds[0][1])
			# print(preds[0][2])
			# print(preds[0][3])
			label = CLASS_LABELS[preds.argmax()]
			#print("NEW LABEL IS: " + label)
			self.currentEmotion = label
			self.emotionTrajectory.append(label)

if __name__=="__main__":
	node = EARS()
	node.run()
	sys.exit(0)