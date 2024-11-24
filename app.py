"""from flask import Flask, request, send_file
import os

app = Flask(__name__)

# Directory to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    print("request: ",request)
    if 'file' not in request.files:
        return 'No file part in the request', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Save the uploaded file
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    # Modify the file (e.g., add a prefix)
    modified_filename = f"modified_{file.filename}"
    with open(os.path.join(app.config['UPLOAD_FOLDER'], file.filename), 'r') as original_file:
        with open(os.path.join(app.config['UPLOAD_FOLDER'], modified_filename), 'w') as modified_file:
            modified_file.write("Modified content: " + original_file.read())

    return modified_filename, 200

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
    app.run(port=5000)"""

"""from flask import Flask, jsonify, request
    

app = Flask(__name__)

@app.route('/api/data', methods=['GET'])
def get_data():
    # Code pour récupérer des données depuis la base de données ou un autre endroit
    data = {"message": "Données récupérées avec succès"}
    return jsonify(data)

@app.route('/api/data', methods=['POST'])
def post_data():
    content = request.get_json()
    # Code pour traiter les données reçues et les enregistrer dans la base de données ou autre
    return jsonify(content), 201
if __name__ == '__main__':
    app.run(debug=True)
    app.run(port=5000) """





"""from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/data', methods=['POST'])
def receive_data():
    print("aaaa")
    data = request.json
    # Traitez les données reçues comme vous le souhaitez
    print(data)
    return jsonify({"message": "Données reçues avec succès"}), 200
if __name__ == '__main__':
    app.run(debug=True,port=5000)"""
    




from flask import Flask, request, jsonify
import librosa
import numpy as np
import tensorflow as tf
from skimage.transform import resize
import os
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize
from tensorflow.keras.models import load_model


# Load the saved model

model = tf.keras.models.load_model('content/myModel.h5')

def extract_features(audio, sr, label=1):
    target_shape = (10, 10)

    # Calcul du mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spectrogram_resized = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    mel_spectrogram_resized_flat = np.reshape(mel_spectrogram_resized, (target_shape[0], target_shape[1]))

    # Concatenate label with mel_spectrogram
    features = np.concatenate(([label], mel_spectrogram_resized_flat.flatten()))
    return features

    
    
def prediction(path):
    print("1")
    data = []
    print("2")
    #audio_path_to_test = 'assets/thia.wav'
    audio_path_to_test = path
    print("3")
    audio, sr = librosa.load(audio_path_to_test, sr=None)
    print("4")
    print("audio: ", audio)
    print("5")

    features = extract_features(audio, sr, label=-1)
    print("features",features)

    data.append(features)
    print("6")
    data = np.array(data)

    X = data[:, 1:]
 
    y = data[:, 0]


    print(X.shape)

    # Predict using the loaded model
    prediction = model.predict(X)

    print('Raw prediction:', prediction)



    #    Map the prediction to the corresponding label based on the threshold
    if prediction >= 0.5:
        return "The audio represents a car accident."
    else:
        return "The audio does not represent a car accident."

app = Flask(__name__)

@app.route('/api/upload_audio', methods=['POST'])
def receive_audio():
    if 'file' not in request.files:
        return jsonify({"message":'Aucun fichier n\'a été envoyé'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message":'Aucun fichier sélectionné'}), 400

    else:
        path= 'assets/'+ file.filename
        print("path:",path)
        print("prediction ",prediction(path))
        return jsonify({"message": prediction(path)}), 200
        
        
if __name__ == '__main__':
    app.run(debug=True)
    app.run(port=5000)



"""from flask import Flask, request
import librosa
import numpy as np
import tensorflow as tf
from skimage.transform import resize

# Load the saved model
model = tf.keras.models.load_model('content/myModel.h5')

def extract_features(audio, sr, label=1):
    target_shape = (10, 10)

    # Calcul du mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spectrogram_resized = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    mel_spectrogram_resized_flat = np.reshape(mel_spectrogram_resized, (target_shape[0], target_shape[1]))

    # Concatenate label with mel_spectrogram
    features = np.concatenate(([label], mel_spectrogram_resized_flat.flatten()))
    return features 


app = Flask(__name__)
@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    # Read audio file
    audio, sr = librosa.load(file, sr=None)
    features = extract_features(audio, sr, label=-1)
    X = np.expand_dims(features[1:], axis=0)  # Exclude label for prediction

    # Predict using the loaded model
    prediction = model.predict(X)
    result = {'prediction': float(prediction)}

    # Map the prediction to the corresponding label based on the threshold
    if prediction >= 0.5:
        return "The audio represents a car accident."
    else:
        return "The audio does not represent a car accident."
if __name__ == '__main__':
    app.run(debug=True)
    app.run(port=5000)"""