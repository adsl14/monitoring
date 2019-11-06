function NDVI(entrada){
var ndvi = entrada.addBands(entrada.normalizedDifference(['B8','B4'])
.rename('NDVI'))
return (ndvi)
}
var parcela = ee.FeatureCollection("ft:12oGtwHqqG-XQd2zuduf4fBiQgqXn4YyOu4bjh48k");
var sentinel = ee.ImageCollection('COPERNICUS/S2')
.filterBounds(parcela)
.filterMetadata('CLOUD_COVERAGE_ASSESSMENT','less_than',5)
sentinel = sentinel.filter(ee.Filter.date('2017','2018'));
Map.centerObject(parcela,15);
var coleccion_ndvi = sentinel.map(NDVI)
var serie_media=ui.Chart.image.series(coleccion_ndvi.select('NDVI'),
parcela, ee.Reducer.mean(), 10)
print(serie_media)