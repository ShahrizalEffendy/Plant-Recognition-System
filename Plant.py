# img_viewer.py

import PySimpleGUIQt as sg
import os.path
import os 
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tkinter import *
import pygame

batch_size = 100
img_height = 300
img_width = 300
class_names = ['Cactus', 'Coconut', 'Daisy', 'Hibiscus', 'Rambutan', 'Sunflower']

new_model = tf.keras.models.load_model('source code/saved_model/my_model')

pygame.mixer.init()
pygame.mixer.music.load("sound2.mp3")
pygame.mixer.music.play()

# First the window layout in 2 columns
file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="source code"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(values=[], enable_events=True, size=(40, 20), key="/"),
    ],

    
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="good")],
    [sg.Text(size=(40, 1), key="result")],
    [sg.Image(key="sun")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column)
      
    ]
]

window = sg.Window("Plant Recognition", layout)


def result(img_path):
    img = keras.preprocessing.image.load_img(
    img_path, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    currentClassName = class_names[np.argmax(score)]
    confidence = str(round(100 * np.max(score),2))
    # return "This image most likely belongs to " + currentClassName + " with a " + confidence + " percent confidence."
    return "Result: " + currentClassName + ", Confidence: " + confidence

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "source code":
        folder = values["source code"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif",".jpg"))
        ]
        window["/"].update(fnames)
    elif event == "/":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["source code"], values["/"][0]
            )
            window["good"].update(filename)
            window["sun"].update(filename=filename)
            window["result"].update(result(filename))

        except:
            pass

        
sg.theme('MynewTheme')
window.close()


