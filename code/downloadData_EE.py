# Change the first 7 values from this code. Then, run it. When it finishes, download
# the folder 'dataEE' from your drive and move it to the path where
# this code 'downloadData_EE.py' is located. Don't change any
# value from this code until you run "cleanData.py" and save the data in "tables" 
# folder.

import ee
import os
import time

# Change this values if you want to download data from other dates or indexes
start_date = '2016-09-01' # PagoBásico 09-01, Rice 11-01
end_date = '2017-08-31' # PagoBásico 08-31, Rice 02-01
sentinels = ["A","B"] # A, B or AB
orbits = ["DESC", "ASC"] # ASC, DESC or ASC_DESC.
indexes_sentinel1 = ['VH_Sum_VV'] # Rice VH_Sum_VV
indexes_sentinel2 = ['ICEDEX','B11']
buffer_value = 0 # 0 or greater means there is no buffer reduction. Less than 0 means apply buffer.

# Change this line to use others user shapefiles
# USER
nameUser = "Dani" # "Dani", "soysusanacanton"

# Select the shape you want to use. You have to upload your shapes to Earth Engine
# before using this code.

# ACTIVITY
#kmls = [["Trigo", "users/"+nameUser+"/"+"Trigo"],["Avena", "users/"+nameUser+"/"+"Avena"],["Girasol", "users/"+nameUser+"/"+"Girasol"],["Barbecho tradicional", "users/"+nameUser+"/"+"Barbecho_tradicional"],["Barbecho sin produccion", "users/"+nameUser+"/"+"Barbecho_sin_produccion"]]

# WATER
kmls = [["ARROZ_17", "users/"+nameUser+"/"+"ARROZ_2017"]]
#kmls = [["ARROZ_18", "users/"+nameUser+"/"+"ARROZ_2018"]]
#kmls = [["ARROZ_19", "users/"+nameUser+"/"+"ARROZ_2019"]]
#kmls = [["Arroz_PDR_2019_revisar_30N", "users/"+nameUser+"/"+"Arroz_PDR_2019_revisar_30N"]]

def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))

# Cloud masking function.
def maskL8sr(image):
  cloudShadowBitMask = ee.Number(2).pow(3).int()
  cloudsBitMask = ee.Number(2).pow(5).int()
  qa = image.select('pixel_qa')
  mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
    qa.bitwiseAnd(cloudsBitMask).eq(0))
  return image.updateMask(mask).select(bands).divide(10000)

def maskS2clouds(image):

  qa = image.select('QA60')

  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloudBitMask = 1 << 10
  cirrusBitMask = 1 << 11

  # Both flags should be set to zero, indicating clear conditions.
  mask = qa.bitwiseAnd(cloudBitMask).eq(0)
  mask = mask.bitwiseAnd(cirrusBitMask).eq(0)

  return image.updateMask(mask).divide(10000)

def addNDVI(img):

  nd = img.normalizedDifference(['B8', 'B4']);
  return img.addBands(nd.float().rename('NDVI'));

def addICEDEX(img):

  b8 = img.select('B8')
  b4 = img.select('B4')
  b11 = img.select('B11')

  x1 = b8.divide(b4)
  x2 = b8.divide(b11)

  icedex = x1.subtract(x2)

  return img.addBands(icedex.float().rename('ICEDEX'))

def normalizedDifference(a, b):
  
  nd = (a - b) / (a + b)
  nd_inf = (a - b) / (a + b + 0.000001)

  if np.isinf(nd):
    return nd
  else:
    return nd_inf

def toNatural(img):

  return ee.Image(10.0).pow(img.select('..').divide(10.0)).copyProperties(img, ['system:time_start'])

def maskEdge(img):
  mask = img.select(0).unitScale(-25, 5).multiply(255).toByte().connectedComponents(ee.Kernel.rectangle(1,1), 100)
  return img.updateMask(mask.select(0))

def addPol(img):

  num = img.select('VH').subtract(img.select('VV'))
  den = img.select('VH').add(img.select('VV'))
  nd = num.divide(den)
  return img.addBands(nd.float().rename('POL'))

def addDiv(img):
  num = img.select('VH');
  den = img.select('VV');
  nd = num.divide(den);
  return img.addBands(nd.float().rename('VH_Div_VV'));

def addVH_Sum_VV(img):

  a = img.select('VH');
  b = img.select('VV');
  result = a.add(b);
  return img.addBands(result.float().rename('VH_Sum_VV'));

def getFeaturesRange(fc,id1,id2):
  
  features = [];
  tam = id2+1

  for i in range(id1,tam):
    feature = fc.toList(fc.size()).get(i)
    features.append(feature);    

  return ee.FeatureCollection(features)

# Modify buffer featureCollection
def getBuffer(fc,size):

  features = []
  tam = fc.size().getInfo()

  for i in range(0,tam):
    feature = ee.Feature(fc.toList(fc.size()).get(i)).buffer(size)
    features.append(feature);    

  return ee.FeatureCollection(features)

def loadSentinel2(table, start_date, end_date):
  
  s2 = ee.ImageCollection('COPERNICUS/S2')
  s2 = s2.filterDate(start_date, end_date)
  s2 = s2.filterBounds(table)
  s2 = s2.filterMetadata('CLOUD_COVERAGE_ASSESSMENT','less_than',5)
  #s2 = s2.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
  
  # ADD NDVI BANDS
  s2 = s2.map(addNDVI)

  # ADD ICEDEX BANDS
  s2 = s2.map(addICEDEX)
  
  return s2;

def loadSentinel1(table, start_date, end_date, type_network='', orbit=''):

  s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  s1 = s1.filterMetadata('instrumentMode', 'equals', 'IW')

  if type_network in ('A','B'):
    s1 = s1.filterMetadata('platform_number', 'equals', type_network)

  if orbit in ("ASC","DESC"):
    s1 = s1.filterMetadata('orbitProperties_pass', 'equals', orbit+'ENDING')

  s1 = s1.filter(ee.Filter.eq('transmitterReceiverPolarisation', ['VV', 'VH']))
  s1 = s1.filterBounds(table)
  s1 = s1.filterDate(start_date, end_date)
  s1 = s1.sort('system:time')
  
  s1_n = s1.map(toNatural);

  # ADD POL BAND
  s1 = s1.map(addPol); 
  s1_n = s1_n.map(addPol);

  # ADD VH_Div_VV BAND
  s1 = s1.map(addDiv); 
  s1_n = s1_n.map(addDiv);

  # ADD VH_Sum_VV BAND
  s1 = s1.map(addVH_Sum_VV); 
  s1_n = s1_n.map(addVH_Sum_VV);
  
  return [s1, s1_n];

def getTimeSeriesTask(imgc, fc):

  def prepareTask(img):
    fco = img.reduceRegions(collection=fc,reducer=ee.Reducer.mean(),scale= 10)
    fco = fco.map(lambda f: f.set('date', img.date()))
    fco = fco.map(lambda f: f.set('regions', fc.size()))
    return fco

  tableSeries = imgc.map(prepareTask).flatten().sort('id')

  return tableSeries

def exportTableSeries(tableSeries,nameFile,indexes):

    featureNames = ["id","date"] + indexes + ['regions']
    return ee.batch.Export.table.toDrive(collection=tableSeries,description='dataset', 
                                         fileNamePrefix=nameFile, fileFormat='CSV',
                                         folder="dataEE",
                                         selectors=featureNames)

def main():

  #os.system("earthengine authenticate")
  ee.Initialize()

  now = time.time()
  # We will export the information for each kml (Trigo, Girasol, Barbecho, etc.)
  for kml in kmls:
    
      table = ee.FeatureCollection(kml[1])

      if buffer_value < 0:
        table = getBuffer(table,buffer_value)

      # Get the areas
      #table = getFeaturesRange(table,0,11)

      # Get radar
      for orbit in orbits:
        for sentinel in sentinels:

          nameOutputFile = kml[0]+"_"+sentinel+"_"+orbit

          # Check if the file already exists
          if not os.path.exists(os.path.join("dataEE",nameOutputFile+".csv")):       

            # Load the imagery with the current feature collection
            s1 = loadSentinel1(table, start_date, end_date, sentinel, orbit)

            # Export the table (sentinel-1)
            tableSeries = getTimeSeriesTask(s1[0],table);  #s1[0] = db; s1[1] = natural
            datasetTask = exportTableSeries(tableSeries,nameOutputFile,indexes_sentinel1)

            # Start the task.
            datasetTask.start()

            while datasetTask.active():
              print('Descargando datos de %s...' %(nameOutputFile))
              time.sleep(30)
            print('Terminado.')

          else:
            print("Fichero %s ya descargado." %(nameOutputFile))


      nameOutputFile = kml[0]+"_s2"

      # Check if the file already exists
      if not os.path.exists(os.path.join("dataEE",nameOutputFile+".csv")): 

        # Get Sentinel-2
        # Load the imagery with the current feature collection
        s2 = loadSentinel2(table, start_date, end_date)

        # Export the table (sentinel-2)
        tableSeries_s2 = getTimeSeriesTask(s2,table);
        datasetTask_s2 = exportTableSeries(tableSeries_s2,kml[0]+"_s2",indexes_sentinel2)

        # Start the task.
        datasetTask_s2.start()

        while datasetTask_s2.active():
          print('Descargando datos de %s...' %(kml[0]+"_s2"))
          time.sleep(30)
        print('Terminado.')

      else:
        print("Fichero %s ya descargado." %(nameOutputFile))

  end = time.time()

  elapsed = end - now
  time_convert(elapsed)

if __name__ == '__main__':
  main()