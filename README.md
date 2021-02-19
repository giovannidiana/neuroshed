# README
# Introduction
NeuroShed provides a platform to quantify the fluorescence intensity of neurons in bright field images. The analysis follows the steps below:

1. 3D image filtering
2. 3D watershed segmentation
3. Neuron classification 
4. Intensity quantification 
5. Background subtraction

Steps 1 and 2 are performed automatically by the script `proc_script` across multiple image while neurons are classified in step 3 through a GUI followed by steps 4,5 performed automatically after classification. 

#Usage
To segment images contained in multiple folders, open `proc_script` and set the main folder `NAS` to the location of the folders. Then prepare a list of all the folders to be analyzed and copy it into the file `folderlist.txt`. To run the segmentation use the command
```
./proc_script
```

The script creates automatically a local folder for each folder analysed where all the segmentation output are stored. 

To proceed with step 3 of the analysis the user has to classify neurons manually. Single folders are processed with the command
```
./classify_script <folder_name> <START_ID>
```
Single images are processed with the command
```
./single_script <folder_name> <ID>
```
The script runs the GUI program thourghout the images in the selected folder in order of image ID starting from `START_ID`. 

For each image, the user interface displays two panels: 

- panel A: the source image (top) and the overlay of source and segmentation (bottom).
- panel B: the maximum intensity projection.


Neurons are associated to their segmented region by clicking on the segmented image (panel A, bottom). The neuron under evaluation is shown at the bottom toolbar of panel A.  

- left mouse clik: Identify single neuron 
- space: Move forward in the list of neurons. 
- s: Show the segmented region restricted to the size of the neuron.
- x: clear current neuron.
- q: quit panels and move to next channel.
- right mouse click: Associate multiple neurons to single segmented volume. After the right click, the terminal asks the number of neurons observed in the volume.

#Other tools
- display original image
```
display <original image filename> <width> <height> <z planes>
```
- display watershed map 
```
displayWS -f <watershed .tiff file>
```
