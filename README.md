# SkyMaticsImageProcessing
The repo for my work with Skymatics.

October 1, Sat: Tile the large (1 GB+) tiff images into smaller images using VIPS library and its scripts.
As an alternative, GDAL library seems to be a good candidate, as it 's done specifically for georeferenced TIFFs.

October : Mastered tiling using gdal2tiles.py, the instructions to follow soon.
Added the unsupervised clustering of images based on the histogram, fractal dimensions and Gabor-filtered images.

Nov 5:
We need to start a Google Drive account / Dropbox account so we could also store the images for each classifier, without rebuilding them all the time.
With the next commit I will add the comments into each file and explain what the file purpose is.


if GDAL2Tiles.py does not work out of the box, consider remapping from latlong to UTM and then running!!!

Ex. :
 gdalwarp -s_srs '+proj=latlong +zone=12 +datum=WGS84' -t_srs '+proj=utm +zone=12 +datum=WGS84' CanolaJune29.tif CanolaJune29UTM2.tif



Now when the images are tiled, I can analyze them on the scale of 1 tile.
Example usage :


gdal2tiles.py -s adlee15GBPRJ.prj adlee15GB.tif ./adlee15GB 

