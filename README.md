# EARS: Emotionally-Aware Reaction System
![](images/ears_demo.gif)
CPSC 459/559 Building Interactive Machines | FALL 2020 | Final Project

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
$cd bim2020_ears
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

