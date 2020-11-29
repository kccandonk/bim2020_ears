#!/usr/bin/env python
# Script for EARS system

import time
import datetime
import pyaudio
import wave
import webrtcvad
import sys
import os

import pygame

from collections import Counter

import tensorflow as tf


# define RGB for white
white = (255, 255, 255)
# set width and height for display
X = 600
Y = 600
startThreshold = 5
silenceThreshold = 3


CHUNK = 480
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 0.5


def main(start_time):
	emotionTrajectory = []
	audio_counter = 0
	hasSpoken = False
	runningSilence = 0
	keepGoing = True

	#model = tf.keras.models.load_model(self.model_file, compile=False)

	while keepGoing:
		audio_counter, hasSpoken, runningSilence = audio_check(audio_counter,hasSpoken,runningSilence)
		emotion_prediction = "mad" # replace with prediction from model
		emotionTrajectory.append(emotion_prediction)
		keepGoing = responsePolicy(keepGoing, hasSpoken, runningSilence,emotion_prediction,emotionTrajectory)
		if keepGoing == False:
			time.sleep(10)
		

def responsePolicy(keepGoing, hasSpoken, runningSilence,emotion_prediction,emotionTrajectory):
	pygame.init()
	# create the display surface object # of specific dimension (X, Y).
	display_surface = pygame.display.set_mode((X,Y))
	# set the pygame window name
	pygame.display.set_caption('EARS')
	# load inital prompt
	prompt = pygame.image.load('images/EARS_initial.jpg')
	prompt_x = prompt.get_width()
	prompt_y = prompt.get_height()
	prompt_x_center = X/2 - prompt_x/2
	prompt_y_center = Y/2 - prompt_y/2

	elapsed_time = time.time()-start_time
	if elapsed_time < startThreshold or hasSpoken==False:
		display_surface.fill(white)
		display_surface.blit(prompt,(prompt_x_center, prompt_y_center))
		pygame.display.update()
	elif runningSilence < silenceThreshold:
		print(elapsed_time)
		emotion_face = pygame.image.load('render_faces/emotion_faces/'+emotion_prediction+'.jpg')
		ef_x = emotion_face.get_width()
		ef_y = emotion_face.get_height()
		ef_x_center = X/2 - ef_x/2
		ef_y_center = Y/2 - ef_y/2
		display_surface.fill(white)
		display_surface.blit(emotion_face,(ef_x_center, ef_y_center))
		pygame.display.update()
	else:
		response_filename = retrieveResponse(emotion_prediction,emotionTrajectory)
		response = pygame.image.load('responses/'+response_filename+'.jpg')
		response_x = response.get_width()
		response_y = response.get_height()
		response_x_center = X/2 - response_x/2
		response_y_center = Y/2 - response_y/2
		display_surface.fill(white)
		display_surface.blit(response,(response_x_center, response_y_center))
		pygame.display.update()
		keepGoing = False
	return keepGoing

def retrieveResponse(emotion_prediction,emotionTrajectory):
	counts = Counter()
	counts.update(emotionTrajectory)
	most_common = counts.most_common(1)[0][0]
	end_counts = Counter()
	end_counts.update(emotionTrajectory[len(emotionTrajectory)/4*3:])
	most_common_end = end_counts.most_common(1)[0][0]
	response_filename = most_common+"_endon_"+most_common_end
	return response_filename

def audio_check(audio_counter,hasSpoken,runningSilence):
	
	WAVE_OUTPUT_FILENAME = output_dir+"/audio_{}.wav".format(str(audio_counter))

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
		hasSpoken=True

	percent_speaking = float(sum(silence_or_speech))/len(silence_or_speech)

	if percent_speaking >= 0.25:
		runningSilence = 0
	else:
		runningSilence += RECORD_SECONDS

	stream.stop_stream()
	stream.close()
	p.terminate()

	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()
	audio_counter = audio_counter+1
	return audio_counter, hasSpoken, runningSilence

if __name__=="__main__":
	start_time = time.time()
	output_dir = "outputs/outputs_{}".format(datetime.datetime.now().strftime("%H-%M-%S"))
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	main(start_time)
	sys.exit(0)


