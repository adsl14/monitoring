var parcela = ee.FeatureCollection("ft:12oGtwHqqG-XQd2zuduf4fBiQgqXn4YyOu4bjh48k");

var date_filter = ee.Filter.date('2017‐10‐01:00:00','2018‐10‐01:00:00');
var sentinel_1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(parcela);
sentinel_1 = sentinel_1.filter(ee.Filter.date('2017','2018'));

// Filter by metadata properties.
var vh = sentinel_1
  // Filter to get images with VV and VH dual polarization.
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  // Filter to get images collected in interferometric wide swath mode.
  .filter(ee.Filter.eq('instrumentMode', 'IW'));
  
  // Filter to get images from different look angles.
var vhAscending = vh.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'));
var vhDescending = vh.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));

// Create a composite from means at different polarizations and look angles.
var composite = ee.Image.cat([
  vhAscending.select('VH').mean(),
  ee.ImageCollection(vhAscending.select('VV').merge(vhDescending.select('VV'))).mean(),
  vhDescending.select('VH').mean()
]).focal_median();

var serie_media = ui.Chart.image.series(composite,parcela,ee.Reducer.mean(), 10);

print(serie_media);