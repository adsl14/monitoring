from cleanData import path_radar, pd, indexes_sentinel1_v2, num_files_radar
from downloadData_EE import os, indexes_sentinel2
import csv, datetime
import matplotlib
import matplotlib as mpl
import matplotlib.dates as dt
import matplotlib.pyplot as plt
from win32api import GetSystemMetrics

# Change this line in order to use other campaings (make sure the only difference is the date)
campaings = ["rice_test_A,B_DESC,ASC_2016-09-01_2017-08-31"]
# Change this line to rename the class name
labels_header = ["class"]

num_files_s2 = len(indexes_sentinel2)
figureCounter = 0

myDPI = 96
screenSize = [GetSystemMetrics(1),GetSystemMetrics(0)]
figureSize = [screenSize[0]*0.90,screenSize[1]*0.40]

# Change x axis text size
matplotlib.rc('xtick', labelsize=8)

def createWindow(numPlots,figureCounter):

	fig, axs = plt.subplots(numPlots,figsize=(figureSize[1]/myDPI,figureSize[0]/myDPI), dpi=myDPI)
	fig.tight_layout()

	thisManager = plt.get_current_fig_manager()

	# Maximized window
	thisManager.window.state('normal')
	# Figure absolute position
	thisManager.window.wm_geometry('+'+str(int(figureSize[1]*figureCounter))+'+'+str(0))
	# Padding modification
	matplotlib.pyplot.subplots_adjust(hspace=numPlots*0.07, top=1.00, bottom=0.06)

	figureCounter+=1

	return axs, fig, figureCounter

def etiquetar(path_radar,epoch,areas,output_writer,num_areas,actual,figureCounter):

  path_dataset = os.path.join(path_radar,epoch,'dataset')

  for area in areas:
    s1 = []
    dates_s1 = []

    area_path = os.path.join(path_dataset,area)
    dataframe = pd.read_csv(area_path)

    for index in indexes_sentinel1_v2:
     s1_aux = dataframe[["date", index]].drop_duplicates().dropna()
     dates_s1_aux = s1_aux["date"]

     dates_s1.append(list(map(datetime.datetime.strptime, dates_s1_aux, len(dates_s1_aux)*['%Y-%m-%d'])))
     s1.append(s1_aux)

    # get the uniques dates whit it correspond data
    s2 = dataframe[ ["date"] + indexes_sentinel2].drop_duplicates().dropna()
    dates_s2 = s2["date"]
    dates_s2 = list(map(datetime.datetime.strptime, dates_s2, len(dates_s2)*['%Y-%m-%d']))

    formatter = dt.DateFormatter('%Y-%m-%d')  # Specify the format - %b gives us Jan, Feb...
    locator = dt.MonthLocator()  # every month

    # Show s2
    j = 0
    axs, fig, figureCounter = createWindow(num_files_s2,figureCounter)
    fig.canvas.set_window_title('SENTINEL2')
    for index in indexes_sentinel2:

      data_s2 = s2[index].values
      
      axs[j].plot(dates_s2, data_s2,label=index)
      axs[j].legend()
      axs[j].grid()
      plt.setp(axs[j].xaxis.get_majorticklabels(), rotation=25)
      X = axs[j].xaxis
      X.set_major_locator(locator)
      X.set_major_formatter(formatter)

      j = j + 1

    # Show the plot in background
    plt.draw()

    # Show radar
    j = 0
    axs, fig, figureCounter = createWindow(num_files_radar,figureCounter)
    fig.canvas.set_window_title('RADAR')
    for i in range(0,num_files_radar):

      data_s1 = s1[i][indexes_sentinel1_v2[i]].values

      axs[j].plot(dates_s1[i],data_s1,label=indexes_sentinel1_v2[i])
      axs[j].legend()
      axs[j].grid()
      plt.setp(axs[j].xaxis.get_majorticklabels(), rotation=25)
      X = axs[j].xaxis
      X.set_major_locator(locator)
      X.set_major_formatter(formatter)

      j = j + 1

    # Show the plot in background
    plt.draw()

    print("Campaña: %s" %(campaing))
    print("Progreso: %d/%d" %(actual,num_areas))
    print("Etiquetando recinto: %s" %(area))
    #print("0 -> NO actividad; 1 -> SI actividad; -1 -> Ignorar muestra en caso de duda")
    #print("0 -> NO ; 1 -> SI; -1 -> Ignorar muestra en caso de duda")
    print("0 -> SI con incidencias")
    print("1 -> SI")
    print("2 -> NO")
    print("-1 -> Ignorar muestra")
    while True:
      tag_0 = input()
      if tag_0 in ['0','1','2','-1']:
        if tag_0 != '-1':
          print("Opción seleccionada: %s" %(tag_0))
        else:
          print('Recinto %s ignorado.' %(area))
        output_writer.writerow([area,tag_0])
        break  
      else:
        print("Introduzca un número permitido.")

    # Close all the figures
    try:
    	plt.close('all')
    except:
    	print("The windows are already closed. No need to close them")

    # Reset var
    figureCounter = 0

    # Increase the number of plot tagged
    actual+=1

    # Clean the window output
    #clear_output()

  print("Proceso de etiquetado finalizado.")

for campaing in campaings:

  areas = os.listdir(os.path.join(path_radar,campaing,'dataset'))
  tags_path = os.path.join(path_radar,campaing,'tags.csv')
  actual = 1

  # Creamos el fichero de etiquetas si no existe
  if not os.path.exists(tags_path):
    with open(tags_path, mode='w', newline='') as output_file:
      output_writer = csv.writer(output_file,delimiter=',')
      output_writer.writerow(['id']+labels_header)

      # Tag the areas
      num_areas = len(areas)

      etiquetar(path_radar,campaing,areas,output_writer,num_areas,actual,figureCounter)

  # if 'tags.csv' exists
  else:

    # Update tags
    tags_path_temp = os.path.join(path_radar,campaing,'tags_temp.csv')
    if os.path.exists(tags_path_temp):
      # Remove original and rename temporal file
      os.remove(tags_path)
      os.rename(tags_path_temp,tags_path)

    dataframe_o = pd.read_csv(tags_path)
    dataframe_o_ids = dataframe_o["id"]

    num_areas = len(areas)

    # Remove the areas that are already taged
    for id_name in dataframe_o_ids.values:
      id_name = id_name.split('/')[-1]
      if id_name in areas: 
        areas.remove(id_name)
        actual = actual + 1

    # If the size of the dataframe is not equal to the size of the areas in the folder, that means there are some areas that we have to tag
    if num_areas > 0:

      with open(tags_path, mode='r') as file_input:
        with open(tags_path_temp, mode='w', newline='') as output_file:
          output_writer = csv.writer(output_file,delimiter=',')
          output_writer.writerow(['id']+labels_header)

          rows = dataframe_o.shape[0]

          # Copy the data from original file to temporal one, and remove the name of the area from the array of areas
          for i in range(0,rows):
            id_name = dataframe_o_ids.iloc[i]
            output_writer.writerow(dataframe_o.iloc[i,:])

            print("Recinto %s ya etiquetado" %(id_name))

          # Tag the areas
          etiquetar(path_radar,campaing,areas,output_writer,num_areas,actual,figureCounter)

        # Remove original and rename temporal file
        os.remove(tags_path)
        os.rename(tags_path_temp,tags_path)

    else:
      print("Todos los recintos pertenecientes a la campaña %s ya han sido etiquetados." % (campaing))