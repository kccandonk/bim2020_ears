# EARS: Emotionally-Aware Reaction System

![](images/ears_demo.gif)

CPSC 459/559 Building Interactive Machines | FALL 2020 | Final Project

## Overview
EARS is a virtual agent that detects a user's emotions and responds based on the detected emotion. EARS feeds visual data into a machine learning model to classify emotions, and uses a rule-based policy based on audio and visual data to respond appropriately.

More specifically, our system:
* prompts a user to vent or tell a story;
* captures audio and visual input;
* uses a machine learning model to detect and classify emotions;
* reflects the detected emotion on the screen using a cartoon face; and
* once sustained silence is detected, uses a predefined policy to suggest an action for the user based on their emotion trajectory.

## Contributors
- Kate Candon
- Kaitlynn Pineda

## Repository Structure

This repo contains all the code to run our emotionally-aware reaction system and work with our pretrained and custom models. It does not contain the training data.

Below is an overview of the directory structure: 

* ``colab_model_training_eval`` contains the google colab file used to load our data, train and evaluate our model. It includes the .ipynb file and a .py version 
* ``emotion_classifier_models`` contains various models saved as an .hdf5 (including a pretrained model and our trained custom model) that can be loaded into our EARS system
  * ```_mini_xception.100_0.65.hdf5```: pre-trained model from [https://github.com/kumarnikhil936/Facial-Emotion-Recognition](https://github.com/kumarnikhil936/Facial-Emotion-Recognition)
* ``images`` contains: 
   * ears_demo.gif: gif demonstrating system
   * EARS_initial.jpg: the intial image displayed by our virtual agent when running the system
   * emotion_faces: folder of cartoon faces for system
   * responses: folder for images of possible reactions and the end of the interaction
* ``preprocessing_scripts`` contains the scripts we used to preprocess the BAUM-1 and RAVDESS datasets
   * ``video_processing.py``: the main script to obtain the images, landmarks, and emotion labels lists from the data
* ``temp_files`` is a folder to house temporary frames collected from the webcam for analysis
* ``EARS_install_python_deps.sh``: bash script to install python dependencies required for our system
* ``ears.py``: script that autonomously runs our EARS system with our custom emotion classifier
* ``ears_pretrained.py``: script that autonomously runs our EARS system with the pre-trained emotion classifier
* ``haarcascade_frontalface_default.xml``: Haar Cascade Frontal Face detector
* ``shape_predictor_68_face_landmarks.dat``: Landmark detector

## Model Training and Testing
All of the model training and testing code was written on Google Colaboratory to facilitate training. The loading of our data and writing out of our models and logs occurs on Google Drive through Google File Stream. Open the ``model_train_eval.ipynb`` file in Google Colaboratory and execute each code block one at a time. Tensorboard data will be saved to ``logs/``, which wil be located under a ``/content`` root directory in Google Colaboratory.

## Getting Started

### Requirements
* Python 3.6+
* ```pip```
* ```git```

### Installation
1. First clone the git respository ```bim2020_ears``` into your machine to load the necessary files for EARS.
```
$ git clone https://github.com/kccandonk/bim2020_ears.git
```
2. Set up virtual environment for EARS system.
```
$ python -m pip install --upgrade pip
$ python -m pip install --user virtualenv
$ python -m venv ears_env
$ source ears_env/bin/activate
```
3. Navigate to ```bim2020_ears``` directory.
```
$ cd bim2020_ears
```
4. Install necessary packages with ```EARS_install_python_deps.sh```.
```
$ ./EARS_install_python_deps.sh
```
5. You can deactivate the virtual environment for EARS with the ```deactivate``` command.
```
$ deactivate
```

## Running EARS
If your virtual environment for EARS system is not active, activate the virtual environment.
```
$ source ears_env/bin/activate
```
Run the code below to autonomously start EARS. The system will close on its own when finished. Repeat the command if you wish to re-use our system.
```
$ python ears.py
```
If you want to run the version of EARS with the pre-trained emotion classifier, you can run ```ears_pretrained.py```.
```
$ python ears_pretrained.py
```
