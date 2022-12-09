from flask import Flask, flash, render_template, request, redirect, url_for
import numpy as np
import keras.utils as image
# from keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import inception_v3
import cv2
import math
import os
import glob
from keras.models import load_model
import base64
from werkzeug.utils import secure_filename

# we define a variable called app
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.debug = True
app.secret_key = "Assignment3ML"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * \
    1024 * 1024  # Define threshold size(16MB)

# using the inceptionV3 model


def get_model():
    model = InceptionV3(weights="imagenet")
    print("[+] model loaded")
    return model


# Initializing the model
model = get_model()

# split the video incoming from request


def split_video(video_path):
    count = 0
    # capturing the video from the given path
    cap = cv2.VideoCapture(video_path)
    frameRate = cap.get(5)  # frame rate
    x = 1
    # removing all other files from the temp folder
    files = glob.glob('static/frames/*')
    for f in files:
        os.remove(f)

    while (cap.isOpened()):
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            # storing the frames of this particular video in temp folder
            filename = 'static/frames/' + "_frame%d.jpg" % count
            count += 1
            cv2.imwrite(filename, frame)
    cap.release()


# preprocess the images before sending to the model

def preprocess():
    # reading all the frames from temp folder
    directory = os.getcwd()
    images = glob.glob(directory + "/static/frames/*.jpg")

    input_images = []

    for i in range(len(images)):
        img = image.load_img(images[i], target_size=(299, 299, 3))
        input_img = image.img_to_array(img)
        input_img = np.expand_dims(input_img, axis=0)
        input_img = inception_v3.preprocess_input(input_img)
        input_images.append(input_img)

    return input_images

# predict images and store them in array


def predictImages():

    input_images = preprocess()

    predict_images = []

    for i in range(len(input_images)):
        predict_img = model.predict(input_images[i])
        predict_images.append(predict_img)

    return predict_images


#  Getting the top five predictions

def topFivePredictions():

    top_five_predict = []

    predict_images = predictImages()

    # print(len(predict_images))
    for i in range(len(predict_images)):
        top_five_predict.append(
            inception_v3.decode_predictions(predict_images[i], top=5))

    # for i in range(len(top_five_predict)):
    #     print(top_five_predict[i])
    #     print('\n')

    return top_five_predict

# getting all objects in the video


def getObjects(top_five_predict):

    objects = []
    video_objects = []

    # displaying all the objects found

    for i in range(len(top_five_predict)):
        for j in range(1):
            for k in range(5):
                for l in range(3):
                    objects.append(top_five_predict[i][j][k][1])

    video_objects = set(objects)
    return video_objects

# getting the most accurate image based on the search


def searchObject(object, top_five_predict):
    display_images = []
    temp = 0
    img_index = -1

# picking the image with a higher accuracy when a user type a word
    # print(object)
    for i in range(len(top_five_predict)):
        for j in range(1):
            for k in range(5):
                for l in range(3):
                    if (top_five_predict[i][j][k][1] == object):
                        highAcc = top_five_predict[i][j][k][2]

                        if (highAcc > temp):
                            temp = highAcc
                            img_index = i

    return img_index


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No video selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # calling the split method
        split_video('static/uploads/' + filename)

        #print('upload_video filename: ' + filename)
        flash('Video successfully uploaded and frames created successfully')
        # return render_template('upload.html', filename=filename)
        return redirect(url_for('search_form'))


@app.route('/search')
def search_form():
    return render_template('search.html')


@app.route('/search', methods=['POST'])
def search_object():
    object = request.form.get("object")
    if object == '':
        flash('No video selected for uploading')
        return redirect(request.url)
    else:
        top_five_predict = topFivePredictions()
        img_index = searchObject(object, top_five_predict)

        # display all objects
        objects = getObjects(top_five_predict)
        flash('Objects that might be in the video: ' + str(objects))

        if (img_index >= 0):
            return render_template('search.html', image_path="/static/frames/_frame"+str(img_index)+".jpg", passedObject=object)
        else:
            return render_template('search.html', message="No image found", passedObject=object)


# @app.route('/display/<filename>')
# def display_video(filename):
#     #print('display_video filename: ' + filename)
#     return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run()
