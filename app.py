import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import librosa

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

class_names=['neutral','joy','disgust','surprise','sadness','fear','anger']
hop_length = 512 # in num. of samples
n_fft = 2048 # window in num. of samples


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print('FIle is ',request.files['audio'].filename )
    f=request.files['audio']
    f.save(f.filename)
    prediction = predict_audio(f)
    
    #output = round(prediction[0], 2)

    
    return render_template('index.html',prediction_text=prediction)

#Function to predict for single sound file for demo 
def predict_audio(aud):
  #print('audio',aud.name)
  audio, sample_rate = librosa.load(aud.filename, res_type='kaiser_fast') 
  mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=40)
  mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
  #data=features_extractor(aud.name)
  #print('data',mfccs_scaled_features)
  
  prediction=model.predict_proba(mfccs_scaled_features)
  #print('prediction_prob',prediction)
  #print('predct',loaded_model.predict(mfccs_scaled_features))
  return {class_names[i]: str(round(float(prediction[i])*100,2))+'%' for i in range(7)}

if __name__ == "__main__":
    app.run(debug=True)