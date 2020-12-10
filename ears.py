#!/usr/bin/env python
# Script for EARS system
# emotion detection adapted from: https://github.com/priya-dwivedi/face_and_emotion_detection


import time
import datetime
import pyaudio
import wave
import webrtcvad
import sys
import os
import dlib
import imageio
import pygame
from collections import Counter
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# set directory values
CWD = ""
MAIN_DIR = os.getcwd()

# define RGB for white
WHITE = (255, 255, 255)
# set width and height for display
X = 600
Y = 600

# set audio thresholds
START_THRESHOLD = 3
SILENCE_THRESHOLD = 3
# initialize fixed values for audio input collection
CHUNK = 480
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 0.5

# set batch_size for predictions
BATCH_SIZE = 16

# pull in Haar Cascade Frontal Face Detector: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
FACE_CLASSIFIER = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# set emotion labels
CLASS_LABELS = {0: 'angry', 1: 'happy', 2: 'sad', 3: 'neutral'} 

# loading Dlib predictor (http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2") and preparing arrays
print( "preparing")
PREDICTOR = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def return_frame_landmarks(cropped_image):
	# adapted from: https://raw.githubusercontent.com/amineHorseman/facial-expression-recognition-using-cnn/master/predict.py
	os.chdir('./temp_files')
	CWD = os.getcwd()
	imageio.imwrite('temp.jpg', cropped_image)
	image2 = cv2.imread('temp.jpg')
	face_rects = [dlib.rectangle()]
	face_landmarks = get_landmarks(image2, face_rects)
	os.chdir(MAIN_DIR)
	return face_landmarks

def get_landmarks(image, rects):
    # this function has been copied from: http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in PREDICTOR(image, rects[0]).parts()])

def face_detector(img):
	# from: https://github.com/priya-dwivedi/face_and_emotion_detection/blob/master/src/EmotionDetector_v2.ipynb
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = FACE_CLASSIFIER.detectMultiScale(gray, 1.3, 5)
	if faces == ():
		return (0,0,0,0), np.zeros((48,48), np.uint8), img
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
	try:
		roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
	except:
		return (x,w,y,h), np.zeros((48,48), np.uint8), img
	return (x,w,y,h), roi_gray, img

class EARS(object):

	def __init__(self):
		self.startTime = None
		self.emotionTrajectory = []
		self.currentEmotion = "neutral"
		self.audio_counter = 0
		self.hasSpoken = False #keep track of whether or not user has spoken
		self.runningSilence = 0 #keep track of current length of silence
		self.keepGoing = True #end script when False
		self.classifier = tf.keras.models.load_model('./emotion_classifier_models/ears_model_full_model_big_dataset_12-08-2020-15-30.hdf5') #last from drive
		self.cap = cv2.VideoCapture(0)

	def run(self):
		# continous loop to run virtual agent until interaction is over
		pygame.init()
		# set up display
		display_surface = pygame.display.set_mode((X,Y))
		pygame.display.set_caption('EARS')
		prompt = pygame.image.load('images/EARS_initial.jpg')
		prompt_x = prompt.get_width()
		prompt_y = prompt.get_height()
		prompt_center_x = X/2 - prompt_x/2
		prompt_center_y = Y/2 - prompt_y/2
		display_surface.fill(WHITE)
		# show initial prompt
		display_surface.blit(prompt,(prompt_center_x,prompt_center_y))
		pygame.display.update()
		# start timer
		self.startTime = time.time()
		#continuous loop
		while self.keepGoing:
			self.audio_check()
			self.predict_emotions()
			# set up display image
			image = self.responseReaction()
			im_x = image.get_width()
			im_y = image.get_height()
			center_x = X/2 - im_x/2
			center_y = Y/2 - im_y/2
			display_surface.fill(WHITE)
			display_surface.blit(image,(center_x,center_y))
			pygame.display.update()
			# end script after 4 seconds if interaction has ended
			if self.keepGoing == False:
				self.cap.release()
				cv2.destroyAllWindows()
				time.sleep(5)

	def responseReaction(self):
		elapsed_time = time.time()-self.startTime
		os.chdir(MAIN_DIR)
		# check if have not yet met start threshold or user hasn't started speaking
		if elapsed_time < START_THRESHOLD or self.hasSpoken==False:
			# display initial prompt
			image = pygame.image.load('images/EARS_initial.jpg')
		# check if user still speaking
		elif self.runningSilence < SILENCE_THRESHOLD:
			# display cartoon face for current emotion
			image = pygame.image.load('images/emotion_faces/'+self.currentEmotion+'.jpg')
		# otherwise (silence threshold has been met)
		else:
			# gather appropriate response
			response_filename = self.retrieveResponse()
			image = pygame.image.load('images/responses/'+response_filename+'.jpg')
			# tell system interaction is over
			self.keepGoing = False
		return image

	def retrieveResponse(self):
		# create counter for emotions over entire interaction
		counts = Counter()
		counts.update(self.emotionTrajectory)
		# find most common emotions
		most_common = counts.most_common(1)[0][0]
		# create counter for emotions in last quarter of interaction
		end_counts = Counter()
		end_counts.update(self.emotionTrajectory[int(len(self.emotionTrajectory)/4)*3:])
		most_common_end = end_counts.most_common(1)[0][0]
		# pull image based on most common overall and of last quarter
		response_filename = most_common+"_endon_"+most_common_end
		return response_filename

	def audio_check(self):
		# adapted from: "Record" example from https://people.csail.mit.edu/hubert/pyaudio/
		
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
		# if speech has been detected, update indicator
		if True in silence_or_speech:
			self.hasSpoken=True
		# see what percent of clips were identified as speech
		percent_speaking = float(sum(silence_or_speech))/len(silence_or_speech)
		# if at least 25% of clips are speech, 0.5s increment is considered speech
		if percent_speaking >= 0.25 or self.hasSpoken==False:
			# reset runningSilence
			self.runningSilence = 0
		# otherwise, 0.5s increment is considered silence
		else:
			# increment runningSilence
			self.runningSilence += RECORD_SECONDS

		stream.stop_stream()
		stream.close()
		p.terminate()

		self.audio_counter = self.audio_counter+1

	def predict_emotions(self):
		# adapted from: https://github.com/priya-dwivedi/face_and_emotion_detection/blob/master/src/EmotionDetector_v2.ipynb
		print("predict")
		ret, frame = self.cap.read()
		rect, face, image = face_detector(frame) #image is the same as frame
		#Get landmarks for the specific frame
		frame_landmarks = return_frame_landmarks(frame)
		if np.sum([face]) != 0.0:
			roi = face.astype("float") / 255.0
			roi = img_to_array(roi)
			roi = np.expand_dims(roi, axis=0)
			frame_landmarks = np.expand_dims(frame_landmarks, axis=0)
			# make a prediction on the ROI and landmarks, then lookup the class
			preds = self.classifier.predict([roi, frame_landmarks], batch_size=BATCH_SIZE)
			preds = np.round(preds[0], 3)
			label = CLASS_LABELS[preds.argmax()]
			self.currentEmotion = label
			self.emotionTrajectory.append(label)

if __name__=="__main__":
	node = EARS()
	node.run()
	sys.exit(0)