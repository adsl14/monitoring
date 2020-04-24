import re

def changeMonthToNumber(month):

	if month == "Jan":
		return "01"
	elif month == "Feb":
		return "02"
	elif month == "Marh":
		return "03"		
	elif month == "Apr":
		return "04"
	elif month == "May":
		return "05"
	elif month == "Jun":
		return "06"
	elif month == "Jul":
		return "07"
	elif month == "Aug":
		return "08"		
	elif month == "Sep":
		return "09"
	elif month == "Oct":
		return "10"
	elif month == "Nov":
		return "11"
	elif month == "Dec":
		return "12"		

def CEdate2normalDate(dates):

	dates_aux = []
	for date in dates:
		month = date.split(" ")[0]
		day = date.split(",")[0].split(" ")[1]
		year = date.split(",")[1].lstrip()

		month = changeMonthToNumber(month)

		dates_aux.append(year+"-"+month+"-"+day)

	return dates_aux

def convertString2NumberList(data):

	data_aux = []
	for element in data:
		data_aux.append(float(element.replace(",","")))

	return data_aux

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key=alphanum_key)

def createFigure(index, width, height):

	# Creating figure
	plt.figure(figsize=(width/80.0,height/80.0), dpi=80)
	plt.xticks(rotation=25)
	plt.grid()
	plt.plot(dates,data,label="ICEDEX", c='blue')
	plt.legend()
	plt.scatter(dates[index],data[index], s=20, c='red')

	X = plt.gca().xaxis
	X.set_major_locator(locator)
	X.set_major_formatter(formatter)
	ax = plt.gcf().axes[0]	
	ax.xaxis.set_major_formatter(formatter)
	
	# Save the figure
	plt.savefig(fname="temp.png")

	plt.close()

	return cv2.imread("temp.png")

def cleanListdir(listFiles,extension="mp4"):

	listFilesAux = []
	for listFile in listFiles:
		if listFile.split(".")[-1] == extension:
			listFilesAux.append(listFile)

	return listFilesAux