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
    * ``video_processing.py``: the main script to obtain the images, landmarks and emotion labels lists from the data.
* ``render_faces`` KATE TO DO
* ``responses`` contains the various images that could potentially be displayed at the end of the users' interaction with our system
* ``silence_detection`` KATE TO DO
* ``ears_class.py`` script that autonomously runs our EARS system

## Data Pre-Processing

## Model Training and Testing

## Running EARS





