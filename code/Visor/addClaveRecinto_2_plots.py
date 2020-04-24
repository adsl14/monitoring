import dbfread
import os

def convertId2Cod(table,filename):

	filename_clean = filename.split(".")[0]

	for t in table:
		for i in range(0,t[1]):

			if filename_clean == t[0].records[i]["ID"]:

				j = 1

				pathInput = os.path.join("plots",filename)

				#nameOutput = str(t[0].records[i]["CLAVE_REC"]) + "-" + str(t[0].records[i]["NUM_EXPEDI"]) + "-" + str(t[0].records[i]["NUM_PARCEL"]) + "-" + str(t[0].records[i]["CAMPANA"])
				nameOutput = str(t[0].records[i]["CLAVE_REC"]) + "-" + t[0].records[i]["ID"]
				pathOutput = os.path.join("plots",nameOutput + ".png")

				#while True:

				#	if not os.path.exists(pathOutput):
				#		break
				#	else:
				#		pathOutput = os.path.join("plots",nameOutput + " (" + str(j) + ")" + ".png")
				#		j = j + 1



				os.rename(pathInput,pathOutput)

				break

# Names csv
plots = os.listdir("plots")

# Get shapes
table_17 = dbfread.DBF('ARROZ_2017.dbf', load=True)
table_18 = dbfread.DBF('ARROZ_2018.dbf', load=True)
table_19 = dbfread.DBF('ARROZ_2019.dbf', load=True)

table = [[table_17,len(table_17.records)],
[table_18,len(table_18.records)],
[table_19,len(table_19.records)]]

i = 1
tam = len(plots)
for plot in plots:
	print("Progress: %d/%d" % (i,tam))
	convertId2Cod(table,plot)
	i=i+1

