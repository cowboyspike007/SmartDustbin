# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
import numpy as np
from keras.preprocessing import image

app = Flask(__name__)

STATIC_FOLDER = 'static'
# Path to the folder where we'll store the upload before prediction
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'
# Path to the folder where we store the different models
MODEL_FOLDER = STATIC_FOLDER + '/models'
    
def predict(fullpath):
# predicting images
    print('[INFO] : Model loading ................')
    # load the model we saved
    model = load_model(MODEL_FOLDER+'/Garbage.h5')
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
    print('[INFO] : Model loaded')
    img = image.load_img(fullpath, target_size=(300, 300))
    img = image.img_to_array(img, dtype=np.uint8)
    img=np.array(img)/255.0
    result=model.predict(img[np.newaxis, ...])
    return result
# Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Process file and predict his label
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)
        result = predict(fullname)
        l=['cardboard','glass','metal','paper','plastic','trash']
        accuracy=(np.max(result[0], axis=-1))*100
        label = l[np.argmax(result[0], axis=-1)]
        
        return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=accuracy)


@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


def create_app():
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=False)