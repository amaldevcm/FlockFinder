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
from groq import Groq
import json
import secrets


app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = secrets.token_hex(32)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

def format_ouput(data):
    print(data)
    try:
        data = data.strip()

        if data.startswith("```json"):
            data = data[7:]  
        if data.endswith("```"):
            data = data[:-3] 

        # Try parsing JSON
        return json.loads(data)

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")

def getInfoLLM(name):
    prompt = """
You are an expert ornithologist assistant.

Given a bird species name, return detailed information about the bird in the following JSON format. Do not include any text outside the JSON. The output must be valid and easily parsable by any Python JSON parser.

Example Input:
Bird Name: "Bald Eagle"

Output format (JSON only):
{
  "name": "Bald Eagle",
  "scientific_name": "Haliaeetus leucocephalus",
  "location": ["North America", "Canada", "Alaska", "continental United States"],
  "habitat": "Near large bodies of open water with abundant food supply and old-growth trees for nesting",
  "wingspan": 180 to 230 centimeter,
  "most_distinctive_feature": ["white head and tail", "yellow beak", "large hooked beak", "powerful talons"],
  "biggest_threat": ["habitat destruction", "lead poisoning", "human disturbance"],
  "lifespan": "20 to 30 years in the wild",
  "estimated_population_size": "About 316,000 individuals globally"
}

Rules:
- Output must be valid JSON with no trailing commas, quotation mismatches, or formatting errors.
- Do not include any explanation, header, or markdown formatting — just the JSON.
- Follow grammar and spelling rules.
- Avoid assumptions — only list what's explicitly mentioned.
- Ensure the output is properly structured for parsing.


"""
    print("\n Name: ", name)
    prompt += f"Bird Name: {name}"
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",  # Or your desired Groq model
        )
        generated_content = chat_completion.choices[0].message.content
        print("\n LLM output:\n",generated_content)
        return format_ouput(generated_content)
    except Exception as e:
        return "Error"

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
    birdname = 'pigeon'
    if request.method == 'GET':
        return render_template('main.html', name=birdname, display='none')
    else:
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            birdInfoLLM = getInfoLLM(birdname)
            print("\n Gpt output: \n", json.dumps(birdInfoLLM))
            if(type(birdInfoLLM) is not dict):
                return render_template('error.html')

            birdInfoLLM['img'] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            return render_template('birdInfo.html', data=birdInfoLLM)

        

@app.route("/info", methods=['GET'])
def showInfo():
    # print('classified bird: ',classifiedBird)
    birdInfo = getInfo(classifiedBird)
    filename = 'NICOBAR-PIGEON.jpg'
    if(birdInfo == 'error'):
        return render_template('main.html', display='none')

    if(len(birdInfo) == 0):
        return render_template('main.html', display='none')

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    birdInfo[0]['img'] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return render_template('birdInfo.html', data=birdInfo[0])