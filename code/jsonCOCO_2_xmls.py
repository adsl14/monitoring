import json
import os

def searchObjects(annotations, image_id):

	elements = []

	for annotation in annotations:
		if int(annotation["image_id"]) == image_id:
			elements.append(annotation)

	return elements

folder_name = os.path.basename(os.getcwd()) # Get name project directory

# read json file
myfile = open('via_export_coco.json')
data = myfile.read()
# parse file
obj = json.loads(data)

# Get list of images, annotations and classes
images = obj['images']
annotations = obj['annotations']
categories = obj['categories']

# Create annots folder
if not os.path.exists("annots"):
	os.mkdir("annots")

for image in images:

	# Get image information
	file_id = image["id"]
	size = [image["width"],image["height"]]	
	file_name = image["file_name"]

	print(file_name)

	# Create xml file
	xmlFileName = file_name.split(".jpg")[0]+".xml"
	path = os.path.join("annots",xmlFileName)

	file = open(path,'w')

	file.write("<annotation>\n")

	#file.write("\t<folder>" + folder_name + "</folder>\n")
	file.write("\t<filename>" + file_name + "</filename>\n")
	file.write("\t<size>\n")
	file.write("\t\t<width>" + str(size[0]) + "</width>\n")
	file.write("\t\t<height>" + str(size[1]) + "</height>\n")
	file.write("\t</size>\n")

	# Get each object from one image
	elements = searchObjects(annotations,file_id)
	
	# iterate for each element in the image
	for element in elements:

		file.write("\t<object>\n")

		category_name = categories[element["category_id"]-1]["name"]
		bndbox = element["bbox"]

		file.write("\t\t<name>" + category_name + "</name>\n")
		file.write("\t\t<bndbox>\n")

		# Bounding box
		file.write("\t\t\t<xmin>" + str(bndbox[0]) + "</xmin>\n")
		file.write("\t\t\t<ymin>" + str(bndbox[1]) + "</ymin>\n")
		file.write("\t\t\t<xmax>" + str(bndbox[0]+bndbox[2]) + "</xmax>\n")
		file.write("\t\t\t<ymax>" + str(bndbox[1]+bndbox[3]) + "</ymax>\n")

		file.write("\t\t</bndbox>\n")

		file.write("\t</object>\n")

	file.write("</annotation>\n")
	file.close()