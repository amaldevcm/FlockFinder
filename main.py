from flask import Flask, flash, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import sys
import cv2
import json
import numpy as np
from tensorflow.python import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import requests


app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

train_datagen = ImageDataGenerator(rescale = 1./255)
train_data = train_datagen.flow_from_directory(directory = './static/train_data',
                                               batch_size= 32,
                                               target_size= (300,300),
                                               class_mode = "categorical")

classifiedBird = 'pigeon'
model = load_model(os.path.join('./static/models','model.keras'))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def getInfo(name):
    animal_api_url = 'https://api.api-ninjas.com/v1/animals?name={}'.format(name)
    response = requests.get(animal_api_url, headers={'X-Api-Key': 'GqOcjTFCs/8JGu3jrb8z3A==VnkDnU3PJIDfv7QO'})
    if response.status_code == requests.codes.ok:
        data = json.loads(response.text)
        print(type(data))
        return data
    else:
        print("Error:", response.status_code, response.text)
        return 'error'

def classifyBird(filename):
    file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    global modal

    img = cv2.imread(file)
    img = cv2.resize(img,(300,300))
    img = img.astype('float32')/255.0
    img = np.expand_dims(img, axis=0)

    y_pred = model.predict(img)
    pred = np.argmax(y_pred,axis=1)
    print('prediction: ', pred, file=sys.stderr)
    # Map the label
    labels = (train_data.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    pred = [labels[k] for k in pred]
    print(pred[0])
    global classifiedBird
    classifiedBird = pred[0]
    return pred[0]


@app.route("/", methods=['GET', 'POST'])
def classify():
    birdname = ''
    if request.method == 'GET':
        return render_template('main.html', name=birdname, display='none')
    else:
        # print('post called', file=sys.stderr)
        if 'file' not in request.files:
            # print('not in extensions', file=sys.stderr)
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            # print('empty filename', file=sys.stderr)
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print('allowed file called', file=sys.stderr)
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            birdname = classifyBird(filename)
            birdInfo = getInfo(birdname)
            if(birdInfo == 'error'):
                return render_template('main.html', display='none')

            if(len(birdInfo) == 0):
                return render_template('main.html', display='none')

            birdInfo[0]['img'] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            return render_template('birdInfo.html', data=birdInfo[0])

        

@app.route("/info", methods=['GET'])
def showInfo():
    # print('classified bird: ',classifiedBird)
    birdInfo = getInfo(classifiedBird)
    filename = 'NICOBAR-PIGEON.jpg'
    if(birdInfo == 'error'):
        return render_template('main.html', display='none')

    if(len(birdInfo) == 0):
        return render_template('main.html', display='none')

    birdInfo[0]['img'] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return render_template('birdInfo.html', data=birdInfo[0])


# if __name__ == '__main__':
    # app.run(debug=True)