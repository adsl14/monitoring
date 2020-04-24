import base64, os, re, sys

def natural_sort(l): 
	convert = lambda text: int(text) if text.isdigit() else text.lower() 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)

if len(sys.argv) < 2:
	print("Error --> Se necesita el nombre de la carpeta. Abortando programa...")
	sys.exit()

folder = sys.argv[1]
folder_base64 = folder + "_base64"

images_name = os.listdir(folder)
images_name = natural_sort(images_name)

i = 1
total = len(images_name)

if not os.path.exists(folder_base64):
	os.mkdir(folder_base64)

for image_name in images_name:
	with open(os.path.join(folder_base64,image_name+".js"), mode='w') as file:
		file.write("var imageTime = ")

		# Get base64 image
		image_file = open(os.path.join(folder,image_name),"rb")
		encoded_string = repr(base64.b64encode(image_file.read()))[2:-1]
		encoded_string = "data:image/png;base64," + encoded_string

		file.write("\"" + encoded_string + "\"")
		print(image_name)
		print("Progress: %d/%d" %(i,total))
		i=i+1
