# Read csv
import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import datetime

def createFigure(index):

	# Creating figure
	plt.figure(figsize=(6,3.78))
	plt.grid()

	# NDVI
	plt.plot(dates,data,label="Trigo_NDVI", c='blue')
	plt.scatter(dates[index],data[index], s=20, c='red')

	# NDVI
	plt.legend()
	plt.gcf().autofmt_xdate(rotation=25)
	
	# Save the figure
	plt.savefig(fname="temp.png")

	plt.close()

	return cv2.imread("temp.png")

df = pd.read_csv("Trigo266.csv")
dates = df["date_ndvi"].dropna().values
data = df["NDVI"].dropna().values

# PREPARE DATES
# Get first day of the year in date object
start = datetime.datetime.strptime(dates[0].split("-")[0] + "-01-01" , "%Y-%m-%d")

# Convert all dates string into date objects
dates = list(map(datetime.datetime.strptime, dates, len(dates)*['%Y-%m-%d']))

# Get the days from actual date to January
nDates = len(dates)
for i in range(0,nDates):
  dates[i] = (dates[i] - start).days


# READ VIDEO
cap = cv2.VideoCapture("ndvi_timeStep_266Trigo.mp4")

# WRITE VIDEO
out = cv2.VideoWriter('ndvi_timeStep_266Trigo_plot.mp4', cv2.VideoWriter_fourcc('m','p','4','v') , 6, (1200,378))

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

num_dates = len(dates)

for index in range(0,num_dates):

	# Get figure
	figure = createFigure(index)

	# Get frame
	ret, frame = cap.read()

	# Concat iamges
	image_output = np.concatenate((frame, figure), axis=1)

	# Write video
	out.write(image_output)

	print("Progress %d/%d" %(index+1,num_dates))

cap.release()
out.release()
cv2.destroyAllWindows()

os.remove("temp.png")
