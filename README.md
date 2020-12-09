# EARS: Emotionally-Aware Reaction System

CPSC 459/559 *Building Interactive Machines* Final Project: EARS

## Contributors
- Kate Candon
- Kaitlynn Pineda

## Repository Structure

This repo contains all the code to run our emotionally-ware reaction system and work with our pretrained and custom models. It does not contain the training data. Below is an overview of the directory structure: 

* ``colab_model`` contains the google colab file used to load our data, train and evaluate our model. It includes the .ipynb file and a .py version. 
* ``emotion_detector_model`` contains various models saved as an .hdf5 (including a pretrained model and our trained custom model) that can be loaded into our EARS system.
* ``images`` contains the intial image displayed by our virtual agent when running the system. 
* ``preprocessing_scripts`` contains the scripts we used to preprocess the BAUM-1 and RAVDESS datasets.  
    * ``video_processing.py``: the main script to obtain the images, landmarks, and emotion labels lists from the data.
* ``render_faces`` KATE TO DO
* ``responses`` contains the various images that could potentially be displayed at the end of the users' interaction with our system
* ``silence_detection`` KATE TO DO
* ``ears.py`` script that autonomously runs our EARS system

## Model Training and Testing
All of the model training and testing code was written on Google Colaboratory to facilitate training. The loading of our data and writing out of our models and logs occurs on Google Drive through Google File Stream

## Running EARS
First run the file ``EARS_install_python_deps.s`` to load the necessary packages and files needed for EARS to function. 
In the linux terminal window, run ``python ears.py`` to autonomously start EARS. The system will close on its own when finished. Run the command again if you wish to re-use our system.




