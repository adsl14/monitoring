# Read csv
import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import datetime
import sys
from functions import *

def main():

	if len(sys.argv) < 2:
		print("Error -> Es necesario pasar el fichero csv")
		exit()
	else:

		dataframeName = sys.argv[1]

		if not os.path.exists(dataframeName):
			print("Fichero csv no encontrado.")
			exit()

	# Load dataframe
	df = pd.read_csv("icedex_regions.csv")
	regions = df.columns[1:]

	# Get videos filename
	videos = cleanListdir(os.listdir("./"),"mp4")
	videos = natural_sort(videos)
	num_regions = len(videos)

	# get the dates
	dates = df["system:time_start"].values
	dates = CEdate2normalDate(dates)
	dates = list(map(datetime.datetime.strptime, dates, len(dates)*['%Y-%m-%d']))
	formatter = dt.DateFormatter('%Y-%m-%d')  # Specify the format - %b gives us Jan, Feb...
	locator = dt.MonthLocator()  # every month

	if not os.path.exists("output"):
		os.mkdir("output")

	for i in range(0,num_regions):

		data = df[regions[i]].values

		videoName = videos[i]

		# READ VIDEO
		cap = cv2.VideoCapture(videoName)

		# Get size
		width  = int(cap.get(3))
		height = int(cap.get(4))

		# WRITE VIDEO
		out = cv2.VideoWriter(os.path.join("output",videoName.replace(".mp4","_plot.mp4")), cv2.VideoWriter_fourcc('m','p','4','v') , 2, (width*2,height))

		if (cap.isOpened()== False): 
		  print("Error opening video stream or file")

		num_dates = len(dates)

		for index in range(0,num_dates):

			# Get frame
			ret, frame = cap.read()

			# The video has ended
			if not ret:
				break

			# Get figure
			figure = createFigure(index,width,height)

			# Concat images
			image_output = np.concatenate((frame, figure), axis=1)

			# Write video
			out.write(image_output)		

			print("Video -> %s. Progress %d/%d" %(videoName,index+1,num_dates))

		cap.release()
		out.release()
		cv2.destroyAllWindows()

		os.remove("temp.png")

if __name__ == "__main__":
	main()