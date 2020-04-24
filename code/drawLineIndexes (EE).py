import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import matplotlib.lines as lines
import datetime
import numpy as np
import sys
import math
from functions import CEdate2normalDate, cleanListdir, natural_sort, convertString2NumberList

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def near(a, b, rtol=1e-5, atol=1e-8):

    return abs(a - b) < (atol + rtol * abs(b))

def crosses(line1, line2):
    """
    Return True if line segment line1 intersects line segment line2 and 
    line1 and line2 are not parallel.
    """
    (x1,y1), (x2,y2) = line1
    (u1,v1), (u2,v2) = line2
    (a,b), (c,d) = (x2-x1, u1-u2), (y2-y1, v1-v2)
    e, f = u1-x1, v1-y1
    denom = float(a*d - b*c)
    if near(denom, 0):
        # parallel
        return False
    else:
        t = (e*d - b*f)/denom
        s = (a*f - e*c)/denom
        # When 0<=t<=1 and 0<=s<=1 the point of intersection occurs within the
        # line segments
        return 0<=t<=1 and 0<=s<=1

def main():

	if not os.path.exists("plots"):
		os.mkdir("plots")

	files = os.listdir("./")
	files = cleanListdir(files,"csv")
	files = natural_sort(files)
	num_files = len(files)

	dfs = []
	for file in files:
		dfs.append(df)

	regions = df.columns.values[1:]
	num_regions = len(regions)
	cols = 1
	rows = int(math.ceil(num_files/cols))

	formatter = dt.DateFormatter('%m-%d')  # Specify the format - %b gives us Jan, Feb...
	locator = dt.MonthLocator()  # every month

	width_figure = 10
	height_figure = 4
	y_min_line = 0
	y_max_line = height_figure+0.59

	umbrals = sys.argv[1].split(",")

	print("Indexes:")
	print(files)
	# Region
	for i in range(0,num_regions):

		print("Processing: %s" %(i))
		fig = plt.figure(figsize=(width_figure*cols,height_figure*rows))

		# Indexes
		for j in range(0,num_files):

			df = dfs[j][["system:time_start",regions[i]]].dropna()
			dates = df.iloc[:,0].values
			data = df.iloc[:,1].values
			name = files[j].split('.')[0]

			if j != 0:

				# Convert string2number ONLY with B11 data
				if name in ['B11']:
					data = convertString2NumberList(data)

				dates = CEdate2normalDate(dates)
				start = datetime.datetime.strptime(dates[0].split("-")[0] + "-01-01" , "%Y-%m-%d")
				dates = list(map(datetime.datetime.strptime, dates, len(dates)*['%Y-%m-%d']))

				# Get the days from actual date to January
				nDates = len(dates)
				for k in range(0,nDates):
  					dates[k] = (dates[k] - start).days

				plt.subplot(rows, cols, j+1, sharex=ax0)
				plt.plot(dates,data,label=files[j].split(".")[0])

			else:

				# Convert string2number ONLY with B11 data
				if name in ['B11']:
					data = convertString2NumberList(data)

				dates = CEdate2normalDate(dates)
				start = datetime.datetime.strptime(dates[0].split("-")[0] + "-01-01" , "%Y-%m-%d")
				dates = list(map(datetime.datetime.strptime, dates, len(dates)*['%Y-%m-%d']))

				# Get the days from actual date to January
				nDates_first= len(dates)
				for k in range(0,nDates_first):
  					dates[k] = (dates[k] - start).days

				data_first = data
				dates_first = dates

				ax0 = plt.subplot(rows, cols, j+1)

				plt.plot(dates,data,label=files[j].split(".")[0])
				for umbral in umbrals:
					plt.axhline(y=float(umbral),xmin=0,xmax=1,color='blue',figure=fig, clip_on=False)

			plt.grid()
			plt.legend()

			plt.gca().axes.xaxis.set_major_locator(locator)
			plt.gca().axes.xaxis.set_major_formatter(formatter)

			# FIX subplot max and min values, similar to Earth Engine
			# B11 index
			if files[j].split('.')[0] in ['B11']:
				max_value = max(data) * 1.05 # 105%
				plt.gca().axes.set_ylim([0,max_value])

			# Sentinel-1 indexes
			elif files[j].split('.')[0] in ['SumVVVH']:
				min_value = min(data) * 1.20 # 105%
				plt.gca().axes.set_ylim([min_value,0])


		yys = umbrals
		xx, yy = [],[]
		xo,yo = dates_first,data_first

		for k in range(1,len(data_first)):
			for l in yys:
				l = float(l)
				p1 = np.array([xo[k-1],yo[k-1]],dtype='float')
				p2 = np.array([xo[k],yo[k]],dtype='float')
				k1 = np.array([xo[k-1],l],dtype='float')
				k2 = np.array([xo[k],l],dtype='float')
				if crosses((p2,p1),(k1,k2)):
					seg = line_intersection((p2,p1),(k1,k2))
					if seg is not None:
						xx.append(seg[0])
						yy.append(seg[1])
						plt.axvline(x=seg[0],ymin=y_min_line,ymax=y_max_line,color='blue',figure=fig, clip_on=False)

		#plt.show()
		plt.savefig(os.path.join("plots",str(i) + ".png"))
		plt.close()

if __name__ == "__main__":
	main()

# 