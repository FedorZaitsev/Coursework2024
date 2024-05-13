# Cucumber Diagnostician Bot

___

## What is in this repository?

This repository is a part of the author's coursework and contains a python package for image classification as well as asynchronous telegram bot backend for obtained models' inference. As the author's project was dedicated to cucumber leaf disease prevention, built in models are trained to diagnose cucumber plant the picture of its leaf, hence the name. However this package and bot backend can be used to implement any kind of image classification.

The author's bot:
https://t.me/cucumber_diagnostician_bot

## Table of Contents
1. [Implementing and training a model](#implementing-and-training-a-model)
2. [Starting the bot](#starting-the-bot)
3. [Adding custom models](#adding-custom-models)
4. [Using the bot](#using-the-bot)

This repository provides a tool for training deep learning image classification models and deploying them using an easy and user friendly frontend system such as Telegram Bot API.

### Implementing and training a model
Classification package backend is implemented using pytorch library and currently available models are located at classification/models.py file. Project architecture allows to easily add custom models with some requirements. To add a custom model one should just add a custom model function in this file and make sure that it is able to take number of classes as an argument, returns an object based on pytorch nn.Module class and that its name is initialized in the namspace dictionary.

To start the training process one should run next command:

```shell
python3 train_model.py <config> <training_data> <validating_data>
```

where ```config``` is a path to a JSON configuration file and ```training_data``` and ```validating_data``` are paths to directories, containing training and validating data respectively. Both training and validating data directories should contain the same set of folders corresponding with the classes of the task. ```validating_data``` argument can be omitted, in this case a part of the training data not involved in the training will be used for validation.

Configuration file is a JSON file that sets all the parameters for the training (such as model, optimizer, scheduler, loss function and their parameters) and for data preprocessing as well (such as the ratio of data to be used in training process and the set of augmentations to use before training, implemented via albumentations package).

One can also provide two extra arguments - Weights and Biases API key and project name to log and track crucial statistics (loss and accuracy) in a W&B run.

```shell
python3 train_model.py <config> <training_data> <validating_data> <wandb_key> <wandb_project_name>
```

Training process by default uses GPU resources if they are available and CPU resources otherwise. There is also a possibility to quantize acquired model using pytorch static quantization functionality if the model architecture allow to do it. In this case quantized model will be saved separately.

### Starting the bot

To start the bot one should at first set an environment variable ```'BOT_KEY'```for example by creating a .env file and placing it there:
```
'BOT_KEY'=YOUR_BOT_TOKEN
```

After that one should just run ```./start_bot.sh``` script, which will parse all saved models and start the bot. 
There are four built in models: a relatively small VGG-inspired convolutional network, ResNet18, its quantized version and a tiny SwinTransformer V2. They were all trained or fine-tuned on a publicly available dataset dedicated to cucumber leaf diseases.
https://www.kaggle.com/datasets/kareem3egm/cucumber-plant-diseases-dataset

There is also an option of building a docker image to run the bot remotely. In order to perform this one should paste bot's API key into the Dockerfile and build docker image using given Dockerfile via ```docker build``` command. The bot will start working after the image is run.

### Adding custom models

To make a custom model available for the bot to use, one should make sure that ```config.json``` file in ```models/``` directory contains an item with a unique key. This item should provide information about the model itself - its type (corresponding with the model namespace dictionary in ```classification/models.py```) and path to the saved model's state_dict. It also should contain its inference preferences - list of classes and albumentations instance corresponding with preprocessing transforms. If these requirements are met, the starting script will be able to copy information about available models in the bot configuration file and will initialize given models before starting the bot.

### Using the bot

To use the bot one should just simply send it a photo to classify (note that it will only parse pictures sent the "quick way" i.e. as a photo and not as a file). After that bot will respond with the name of the most possible class of the picture according to the chosen model. It should be clear that possible output depends entirely on the chosen model and its classes. Model can be changed via ```/config``` command, which will show the user a list of all available models and will offer to choose one of them with a keyboard button.

