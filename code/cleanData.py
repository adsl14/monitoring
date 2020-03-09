from downloadData_EE import time, os, time_convert, start_date, end_date, indexes_sentinel1, indexes_sentinel2, sentinels, orbits, nameUser, kmls
import pandas as pd

tables_folder = 'tables'
nameExperimentsFolder = 'experiments'
radar_folder = 'radar'
nameExperiment = 'rice'
indexes_sentinel1_v2 = []

# Update indexes_sentinel1 in other var
for i in indexes_sentinel1:
  for o in orbits:
    for s in sentinels:
      indexes_sentinel1_v2.append(i+"_"+s+"_"+o)

campaing_date = start_date+"_"+end_date
experimentFolder = nameExperiment + "_" + ",".join(map(str,sentinels))  + "_" + ",".join(map(str,orbits))
campaingFolder = experimentFolder + "_" + campaing_date
num_files_radar = len(indexes_sentinel1_v2)

path_tables = os.path.join(tables_folder)
path_radar = os.path.join(tables_folder,radar_folder)
path_experiments = os.path.join(nameExperimentsFolder)
path_epoch = os.path.join(path_radar,campaingFolder)
path_dataset = os.path.join(path_epoch,'dataset')

def main():

  # tables folder
  if not os.path.exists(path_tables):
    os.mkdir(path_tables)

  # radar folder
  if not os.path.exists(path_radar):
    os.mkdir(path_radar)

  # epoch folder
  if not os.path.exists(path_epoch):
    os.mkdir(path_epoch)

  # dataset folder
  if not os.path.exists(path_dataset):
    os.mkdir(path_dataset)

  now = time.time()
  for kml in kmls:

    # SENTINEL-2
    dataFrame_s2 = pd.read_csv(os.path.join("dataEE",kml[0]+"_s2.csv"))

    # Get the number of areas
    areas = dataFrame_s2["regions"].loc[0]

    # Get number of observations (sentinel-2)
    observations_s2 = int(dataFrame_s2.shape[0]/areas)

    # Get the dates
    dates_s2 = [i.split('T')[0] for i in dataFrame_s2["date"].iloc[0:observations_s2]]  

    # SENTINEL-1
    dataFrames_s1 = []
    observations_s1 = []
    names = []
    for orbit in orbits:
      for sentinel in sentinels:
        names.append(sentinel+"_"+orbit)
        nameFile = kml[0]+"_"+names[-1]
        dataFrame_s1 = pd.read_csv(os.path.join("dataEE",nameFile+".csv"))
        observation_s1 = int(dataFrame_s1.shape[0]/areas)

        dataFrames_s1.append(dataFrame_s1)
        observations_s1.append(observation_s1)

    # Iterate over each area
    for i in range(0,areas):

      # name area
      name = dataFrame_s2["id"].iloc[i*observations_s2]

      # Save data
      path = os.path.join(path_dataset,str(name)+".csv")

      if os.path.exists(path):
        print("File %s already saved" %(name))
        continue

      # S1 BLOCK
      # get S1 data over each radar file
      df_s1_aux = []
      for j in range(0,num_files_radar):

        # Get the observations indexes
        start = i*observations_s1[j]
        end = start+observations_s1[j]

        # load dataframe
        dataFrame_s1 = dataFrames_s1[j][["date"] + indexes_sentinel1].iloc[start:end]

        # Rename all the indexes-sentinel-1 column
        for index_sentinel1 in indexes_sentinel1:
          dataFrame_s1 = dataFrame_s1.rename(columns={index_sentinel1 : index_sentinel1 + "_" + names[j]})

        # Set the 'date' as the index of the dataframe
        dataFrame_s1["date"] = [i.split('T')[0] for i in dataFrame_s1["date"].iloc[0:observations_s1[j]]]
        dataFrame_s1["date"] = pd.to_datetime(dataFrame_s1['date'])
        dataFrame_s1 = dataFrame_s1.set_index('date')

        # Save in the list, the dataframe
        df_s1_aux.append(dataFrame_s1)

      # Concat all the dataframes of radar for one area
      df_s1 = df_s1_aux[0].join(df_s1_aux[1:], how='outer')

      # S2 BLOCK
      # get S2 data
      start = i*observations_s2
      end = start+observations_s2
      df_s2 = dataFrame_s2[["date"] + indexes_sentinel2].iloc[start:end]

      # set index to date
      df_s2["date"] = [i.split('T')[0] for i in df_s2["date"].iloc[0:observations_s2]]
      df_s2['date'] = pd.to_datetime(df_s2['date'])
      df_s2 = df_s2.set_index('date')

      # concat and save dataframe
      df = df_s1.join(df_s2, how='outer')
      pd.DataFrame.to_csv(df,path_or_buf=path, index=True)   

      print("File saved in %s" %(path))

    #os.remove(kml[0]+".csv")
    #os.remove(kml[0]+"_ndvi.csv")

  end = time.time()

  elapsed = end - now
  time_convert(elapsed)

if __name__ == '__main__':
  main()