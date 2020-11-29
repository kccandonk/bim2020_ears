#!/usr/bin/env python

import numpy as np
import pandas as pd
import math 
import re
import os
import os.path
from os import path
import argparse
import errno
import cv2
import sys

#here have the cropping and formatting code??
r_path = "/home/ubuntu/ears/DATA/RAVDESS"
path = "/home/ubuntu/ears/DATA/"
dest = "/home/ubuntu/ears/DATA/"
emot_path = "/home/ubuntu/ears/DATA/by_emotion"
frames_dest = "/home/ubuntu/ears/DATA/frames_by_emotion/"

s_baum = os.path.join(path, "original_raw/BAUM1s_MP4_all/") 
a_baum = os.path.join(path, "original_raw/BAUM1a_MP4_all/") 

def declare_emotion_labels(etype):
	# LABELS FOR THE A_BAUM DATASET

	A_Anger = 'S001_001,S002_005,S003_005,S004_005,S006_005,S007_007,S008_006,S008_007,S008_008,S008_015,S009_005,S011_007,S012_005,S013_005,S013_006,S013_008,S013_011,S014_006,S016_005,S016_006,S017_005,S018_006,S019_012,S019_013,S019_014,S020_009,S021_005,S022_006,S023_007,S023_008,S024_004,S025_006,S026_018,S026_019,S026_020,S027_004,S028_020,S028_021,S029_004,S029_005,S030_008,S031_005'
	A_Happiness = 'S002_002,S003_002,S004_002,S006_002,S007_002,S007_003,S008_002,S009_002,S010_003,S011_003,S012_003,S014_002,S017_001,S018_002,S019_005,S019_023,S021_001,S022_001,S023_002,S024_001,S025_001,S026_006,S027_001,S029_001,S030_002,S031_002'
	A_Sadness = 'S002_003,S003_003,S004_003,S006_003,S007_004,S009_003,S010_004,S011_004,S011_005,S012_004,S013_003,S014_003,S014_004,S016_002,S017_002,S017_003,S018_003,S018_004,S019_006,S019_008,S019_009,S020_004,S020_006,S021_003,S022_003,S023_003,S023_004,S025_003,S026_010,S026_011,S026_012,S027_002,S030_003,S030_004,S030_005,S031_003,S031_004'
	# A_Neutral = '[]'

	Anger_A = A_Anger.split(",")
	Happiness_A = A_Happiness.split(",")
	Sadness_A = A_Sadness.split(",")

	# LABELS FOR THE S_BAUM DATASET

	S_Anger = 'S001_015,S002_027,S002_028,S003_028,S003_046,S004_029,S004_031,S004_042,S007_029,S008_029,S008_030,S008_033,S009_026,S009_032,S009_033,S009_039,S009_044,S009_045,S010_043,S012_030,S013_032,S013_042,S015_040,S015_041,S015_043,S015_044,S017_044,S017_047,S018_033,S019_054,S019_055,S019_056,S019_057,S019_058,S020_044,S020_045,S020_046,S020_047,S020_058,S020_059,S021_049,S021_050,S021_064,S021_067,S021_068,S021_069,S021_074,S022_032,S022_039,S022_053,S024_015,S025_042,S026_064,S026_065,S026_068,S030_044'
	S_Happiness = 'S001_010,S001_011,S002_015,S002_017,S002_018,S002_020,S002_022,S002_026,S002_029,S002_033,S002_038,S002_039,S002_045,S003_015,S003_016,S003_017,S003_018,S003_020,S003_022,S003_029,S003_032,S003_039,S003_041,S003_042,S003_043,S004_017,S004_018,S004_022,S004_023,S006_016,S006_018,S006_019,S006_022,S006_026,S006_037,S007_018,S007_020,S008_022,S008_023,S008_024,S008_025,S008_034,S008_046,S008_049,S008_050,S009_015,S010_001,S010_016,S010_024,S010_025,S010_026,S010_051,S010_059,S010_072,S011_001,S011_020,S011_021,S011_022,S011_024,S012_001,S012_002,S012_012,S012_013,S012_014,S012_015,S012_017,S013_017,S013_018,S013_020,S013_021,S014_015,S014_016,S014_017,S014_018,S014_030,S015_001,S015_007,S015_016,S015_017,S015_019,S015_028,S015_045,S015_056,S016_004,S016_008,S016_012,S017_021,S017_024,S017_025,S017_026,S017_032,S017_034,S017_054,S017_055,S018_018,S018_021,S018_022,S018_023,S018_032,S019_034,S019_035,S019_036,S019_037,S019_038,S019_041,S019_042,S020_021,S020_023,S020_026,S021_010,S021_027,S021_028,S021_031,S021_038,S021_039,S021_070,S021_100,S022_022,S022_025,S022_028,S022_041,S022_061,S023_017,S023_018,S023_027,S023_028,S023_029,S023_031,S023_032,S023_033,S024_029,S025_016,S025_034,S025_035,S026_031,S026_032,S026_033,S026_035,S026_048,S026_049,S026_052,S026_079,S026_080,S027_015,S028_004,S028_006,S028_007,S028_013,S028_015,S028_024,S028_029,S028_034,S028_035,S028_036,S028_037,S029_024,S029_025,S029_045,S029_046,S030_001,S030_014,S030_017,S030_022,S030_023,S030_030,S030_031,S030_037,S030_049,S031_019,S031_020,S031_021,S031_023,S031_026'
	S_Sadness = 'S018_036,S001_014,S002_048,S003_025,S003_027,S003_036,S004_026,S004_027,S004_028,S006_029,S007_028,S008_032,S008_051,S009_028,S009_029,S009_031,S009_034,S009_035,S009_036,S009_046,S009_047,S009_050,S010_037,S010_038,S010_042,S010_048,S010_049,S010_060,S010_061,S010_062,S012_019,S012_021,S012_029,S012_032,S012_034,S012_040,S013_024,S013_027,S013_028,S013_034,S013_038,S015_033,S017_037,S017_038,S017_039,S017_040,S017_041,S017_042,S017_043,S017_045,S017_046,S017_048,S017_050,S018_028,S018_030,S018_031,S019_046,S019_047,S019_048,S019_049,S019_050,S019_051,S019_052,S019_053,S019_067,S019_068,S019_069,S020_033,S020_034,S020_035,S020_037,S020_038,S020_039,S020_040,S020_041,S020_042,S020_043,S020_063,S021_047,S021_051,S021_052,S021_053,S021_054,S021_055,S021_058,S021_065,S021_066,S021_083,S021_084,S021_086,S021_087,S022_048,S022_050,S022_052,S022_055,S022_058,S023_037,S023_038,S023_042,S025_024,S025_025,S025_027,S025_038,S026_013,S026_056,S026_057,S026_058,S026_059,S026_060,S026_061,S026_062,S026_067,S026_069,S026_076,S026_077,S027_031,S027_034,S027_035,S027_036,S029_030,S029_035,S029_036,S030_040,S030_041,S030_051,S031_032,S031_033,S031_034,S031_035,S031_037,S031_040,S031_042,S031_043,S031_045'
	S_Neutral = 'S002_001,S002_013,S002_019,S002_035,S002_040,S002_041,S002_043,S003_001,S003_013,S003_019,S003_021,S003_031,S003_040,S004_001,S004_025,S004_030,S006_001,S006_013,S006_030,S006_032,S006_036,S006_040,S006_042,S006_045,S006_047,S006_055,S007_001,S007_025,S008_001,S008_031,S009_001,S009_014,S009_016,S009_025,S009_038,S010_002,S010_032,S010_050,S010_053,S011_002,S011_033,S012_033,S012_035,S012_037,S013_001,S013_019,S013_023,S014_001,S014_022,S014_023,S014_024,S014_027,S014_028,S014_029,S014_036,S015_008,S015_013,S015_026,S015_036,S015_048,S015_050,S016_001,S016_014,S017_023,S017_028,S017_036,S018_001,S018_012,S018_015,S018_016,S019_001,S019_030,S019_031,S019_039,S019_040,S019_043,S019_044,S019_059,S019_066,S019_071,S020_001,S020_022,S020_025,S020_027,S020_036,S020_048,S020_051,S020_052,S021_014,S021_021,S021_048,S021_060,S021_073,S022_017,S022_023,S022_026,S022_033,S022_040,S022_054,S022_062,S022_068,S023_020,S023_021,S023_022,S023_023,S023_024,S023_025,S023_026,S023_040,S023_049,S023_050,S024_012,S024_014,S024_017,S024_020,S024_035,S024_036,S024_041,S025_013,S025_014,S025_018,S025_019,S025_020,S025_021,S025_023,S025_028,S025_029,S025_030,S025_048,S025_050,S026_001,S026_034,S026_043,S026_053,S026_055,S026_081,S027_018,S027_026,S027_033,S027_040,S028_005,S028_010,S028_014,S028_016,S028_017,S028_025,S029_010,S029_011,S029_012,S029_013,S029_014,S029_015,S029_017,S029_018,S029_020,S029_026,S029_027,S029_028,S029_029,S029_032,S029_033,S029_034,S029_039,S029_042,S029_043,S029_044,S030_016,S030_020,S030_028,S030_032,S030_033,S030_045,S030_046,S030_050,S030_053,S031_001,S031_013,S031_022,S031_025,S031_027,S031_029,S031_031,S031_044,S031_046,S031_047'

	Anger_S = S_Anger.split(",")
	Happiness_S = S_Happiness.split(",")
	Sadness_S = S_Sadness.split(",")
	Neutral_S = S_Neutral.split(",")
	if etype == "s":
		return Anger_S, Happiness_S, Sadness_S, Neutral_S 
	elif etype == "a":
		return Anger_A, Happiness_A, Sadness_A

def Scheck_emotion(filename, current_path):	
    Anger_S, Happiness_S, Sadness_S, Neutral_S  = declare_emotion_labels("s")
    if filename in Neutral_S:
        os.rename(current_path, dest + "S_Neutral/" + filename + ".mp4")       
    elif filename in Happiness_S:
        os.rename(current_path, dest + "S_Happiness/" + filename + ".mp4")       
    elif filename in Sadness_S:
        os.rename(current_path, dest + "S_Sadness/" + filename + ".mp4")         
    elif filename in Anger_S:
        os.rename(current_path, dest + "S_Anger/" + filename + ".mp4")
    else:
        return "none"
    return filename
    
def Acheck_emotion(filename, current_path):
    Anger_A, Happiness_A, Sadness_A = declare_emotion_labels("a")
    if filename in Anger_A:
        os.rename(current_path, dest + "A_Anger/" + filename + ".mp4")       
    elif filename in Sadness_A:
        os.rename(current_path, dest + "A_Sadness/" + filename + ".mp4")       
    elif filename in Happiness_A:
        os.rename(current_path, dest + "A_Happiness/" + filename + ".mp4") 
    else:
        return "none"
    return filename

def Rcheck_emotion(filename, emotion, current_path):
    #(01 = neutral, 02 = calm, 03 = happy, 04 = sad, 
    # 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
    if emotion == "01": #NEUTRAL 
        os.rename(current_path, dest + "R_Neutral/" + filename + ".mp4")
        return "NEUTRAL"
    elif emotion == "03": #HAPPY
        os.rename(current_path, dest + "R_Happiness/" + filename + ".mp4")
        return "HAPPY"
    elif emotion == "04": #SAD
        os.rename(current_path, dest + "R_Sadness/" + filename + ".mp4")
        return "SAD"
    elif emotion == "05": #ANGRY
        os.rename(current_path, dest + "R_Anger/" + filename + ".mp4")
        return "ANGRY"
    else:
        return "none"

#separate the baum datset by emotion 
def move_files():
	#FOR THE BAUM_S DATA
	for folder in os.listdir(s_baum):
	    v_path = os.path.join(s_baum, folder) 
	    for video in os.listdir(v_path):
	        if video.endswith(".mp4"): 
	            filename = str(video)[:-4] 
	            #check wich emotion it belongs to
	            emotion = Scheck_emotion(filename, os.path.join(v_path, video))
	            if emotion != "none":
	                print("moving "+ emotion)
	            
	#FOR THE BAUM_A DATA
	for folder in os.listdir(a_baum):
	    v_path = os.path.join(a_baum, folder) 
	    for video in os.listdir(v_path):
	        if video.endswith(".mp4"): 
	            filename = str(video)[:-4] 
	            #check wich emotion it belongs to
	            emotion = Acheck_emotion(filename, os.path.join(v_path, video))  
	            if emotion != "none":
	                print("moving "+ emotion)

	#FOR THE RAVDESS DATA
	for folder in os.listdir(r_path):
	    actor_path = os.path.join(r_path, folder) 
	    for video in os.listdir(actor_path):
	        video_path = os.path.join(actor_path, video)
	        if video.endswith(".mp4"): 
	            emotion = str(video)[6:8] 
	            actor = str(video)[18:20] 
	            filename = str(video)[:-4] 
	            #check wich emotion it belongs to
	            emotion = check_ravdess_emotion(filename, emotion, video_path)
	            if emotion != "none":
	                print("moving actor:" + actor + " and emotion:"+ emotion)

def split_videos_by_frame():
	for folder in os.listdir(emot_path):
			folder_path = os.path.join(emot_path, folder)
			folder_name = str(folder)
			print("Folder: " + folder_name)
			for video in os.listdir(folder_path):
				video_path = os.path.join(folder_path, video)
				video_name = str(video)
				# code modified from: https://gist.github.com/keithweaver/70df4922fec74ea87405b83840b45d57
				# Playing video from file:
				cap = cv2.VideoCapture(video_path)
				new_path = os.path.join(frames_dest, folder_name)
				try:
					if not os.path.exists(frames_dest + folder_name + "/" + video):
						os.makedirs(frames_dest + folder_name + "/" + video)
				except OSError:
					print ('Error: Creating directory of' + frames_dest + folder_name + "/" + video)

				current = os.getcwd()
				os.chdir(new_path)
				print("newdir: "+ str(new_path))
				currentFrame = 0
				while(True):
					# Capture frame-by-frame
					ret, frame = cap.read()
					if not ret: 
						break
					# Saves image of the current frame in jpg file
					name = './' + video + '/frame' + str(currentFrame) + '.jpg'
					print ('Folder: '+ folder_name + '| Creating...' + name)
					cv2.imwrite(name, frame)
					# To stop duplicate images
					currentFrame += 1
				os.chdir(current)
				# When everything done, release the capture
				cap.release()
				cv2.destroyAllWindows()

if __name__=='__main__':

	#move_files()
	split_videos_by_frame()

	sys.exit(0)
