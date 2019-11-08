function NDVI(entrada){
var ndvi = entrada.addBands(entrada.normalizedDifference(['B8','B4'])
.rename('NDVI'))
return (ndvi)
}
var parcela = ee.FeatureCollection('ft:1MIl3kw3LTwfv7X7CAHKLS2fozpN9qs9EP3Adm0LO');
var sentinel = ee.ImageCollection('COPERNICUS/S2')
.filterBounds(parcela)
.filterDate('2017-09-01','2018-08-31')
.filterMetadata('CLOUD_COVERAGE_ASSESSMENT','less_than',5)
Map.centerObject(parcela,15);
var coleccion_ndvi = sentinel.map(NDVI)

var regiones =
ui.Chart.image.seriesByRegion(coleccion_ndvi.select('NDVI'),parcela,ee.Reducer.mean(),'NDVI'
,10)
.setOptions({
title: 'Evolución NDVI',
vAxis: {title: 'NDVI'},
hAxis: {title: 'Tiempo'},
})

var serie_media=ui.Chart.image.series(coleccion_ndvi.select('NDVI'),
parcela, ee.Reducer.mean(), 10)
print(serie_media)

print(regiones)