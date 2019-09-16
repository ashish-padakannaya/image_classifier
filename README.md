# Pre-requisites

After cloning the repository, Make sure you have Python3 and mongodb installed and running in your system.

 - [https://www.python.org/downloads/](https://www.python.org/downloads/)
 - [https://www.mongodb.com/download-center/community](https://www.mongodb.com/download-center/community)

Create a new virtual environment and activate it.

    python3 -m venv /path/to/new/virtual/environment_name
    source /path/to/new/virtual/environment_name/bin activate
    
Now install all the required libraries. Make sure you're inside the project directory before running the command.

    pip install -r requirements.txt



# config.ini

config.ini holds all the parameters to be manipulated.

|  KEY|POSSIBLE VALUES |DESCRIPTION  |
|--|--|--|
| model| sift, color_moment  |Specify which model to run the search against. |
rebuild_vectors | True, False | Generate the vectors for the model before running the similarity check. Vectors are stored in the database don’t need to be generated each time we search unless the dataset changes.|
|file_name| A valid file name in the dataset|The name of the file in the dataset with its extension in the dataset. |
| k | Integer Values only | The number of similar images to retrieve.|
| images_directory | Path to image dataset | The absolute or relative path to the image dataset
| chrome_path | Chrome app OS path  | Chrome browser path to open the output file of k similar images with their scores. Varies with OS.
| visualize_single_vector | True, False | if set to True, it stores the descriptors and visual outputs for the specified model in the output directory 



# Run the program

Run the file using the command

    python3 get_nearest.py

The inputs in the config file can be changed for the program to ingest. 
For instance, If the system has to find the **10** most similar images using **color moments** for an image **Hand_0000002.jpeg** which is present in a folder **Hands_dataset/** inside the project directory, the following values must be set in the file config.ini 

	{
		model: color_moment,
		rebuild_vectors: True,
		file_name: Hand_0000002.jpeg,
		k: 10,
		visualize_single_vector: False,
		images_directory: Hands_dataset/
	}

If we need to find the **100** most similar for the same image but this time with **SIFT**, we use the same config as earlier but change the following variables

	{
		model: sift,
	    	'',
	    	'',
	    	k: 100,
		visualize_single_vector: False,
	    	''
	}

In addition to similar images, to get the feature descriptor as an output for the target image (*file_name*). Set the following variable in **config.ini**.

	{
		visualize_single_vector: True
	}

The vectors and visualisation for the model will be stored in the folder 'output' and varies based on the model set in the config file. 

 1. **SIFT**: it outputs a file **sift.jpg** which visualises the keypoints detected in *file_name* and **sift_descriptors.txt** which contains the descriptor array.
 2. **COLOR_MOMENT**:  generates file **YUV.jpeg** which contain the Y, U and V channel of the image and also has 3 files which contain the color moments (mean, standard deviation, skewness) of the 100*100 blocks of all 3 channels, **Y_moments.txt,U_moments.txt,V_moments.txt**
