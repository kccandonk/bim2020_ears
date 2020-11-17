#!/usr/bin/env python
# Script to display face for emotion using pygame
# adapted from: https://www.geeksforgeeks.org/python-display-images-with-pygame/

import pygame
import argparse
import sys


# define RGB for white
white = (255, 255, 255)
# set width and height for display
X = 600
Y = 600
surescale = 2 # TO DO-- calculate this based off model

def display_emotion(emotion):
	pygame.init()
	# create the display surface object # of specific dimension (X, Y).
	display_surface = pygame.display.set_mode((X,Y))
	# set the pygame window name
	pygame.display.set_caption('Image')
	# load image based on emotion
	image = pygame.image.load('emotion_faces/'+emotion+'.jpeg')
	im_x = image.get_width()
	im_y = image.get_height()
	# rescale image based on sureness
	image = pygame.transform.scale(image, (im_x*surescale,im_y*surescale))
	im_x = image.get_width()
	im_y = image.get_height()
	x_for_centering = X/2 - im_x/2
	y_for_centering = Y/2 - im_y/2
	while(True):
		# set white background
		display_surface.fill(white)
		# display image at center
		display_surface.blit(image, (x_for_centering, y_for_centering))
        # Draws the surface object to the screen.   
		pygame.display.update()

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("emotion",help="emotion to display",type=str)
	args = parser.parse_args()
	display_emotion(args.emotion)
	sys.exit(0)