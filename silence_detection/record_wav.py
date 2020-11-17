#!/usr/bin/env python
# Script to record audio from microphone as WAV file
# adapted from: "Record" example from https://people.csail.mit.edu/hubert/pyaudio/

import pyaudio
import wave
import webrtcvad
import sys
import datetime
import os

CHUNK = 480
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1



def record_wav():
	
	WAVE_OUTPUT_FILENAME = output_dir+"/audio_{}.wav".format(datetime.datetime.now().strftime("%H-%M-%S"))

	p = pyaudio.PyAudio()
	vad = webrtcvad.Vad(3)

	stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

	print("* recording")

	frames = []
	silence_or_speech = []

	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data)
		silence_or_speech.append(vad.is_speech(data,RATE))

	print("* done recording")
	print(silence_or_speech)

	stream.stop_stream()
	stream.close()
	p.terminate()

	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()


if __name__=="__main__":
	output_dir = "outputs/outputs_{}".format(datetime.datetime.now().strftime("%H-%M-%S"))
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	while(True):
		record_wav()
	sys.exit(0)