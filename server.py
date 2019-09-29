!pip install face_recognition
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
import cv2
from __future__ import print_function
import json
from os.path import join, dirname
from ibm_watson import ToneAnalyzerV3
import flask
import firebase_admin
from firebase_admin import credentials, storage, db
import io
import pickle
import os, random
import numpy as np
import binascii, struct
import zipfile


app = flask.Flask(__name__)
cred = {
  "type": "service_account",
  "project_id": "sgih-4b054",
  "private_key_id": "03c4583d19986875fc771564c38bfaef128e50b7",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCLugjXHyR4lfI2\n8TcRuWxAjvCkHGI7gRNii5h4Y2AQJflfnq0g6uacmfS01h6291mK7UNO+Lpl+x6a\nDVFRXc16aOnM2lQeNSilK3W9ez5D8rbYYJ4RMA/bADH+IV4LRVSuVS7LB4Z8UZ8d\nSSRzjJGuzTumwDxDp9UgQb+aoj5dCgT+a0iYd9IZH9Wcto6kRmCBC2IC4pRNYa5Q\nR7DrqRQMIVJLgKFYLYhOp02p77aH65jpYH4gNHhD2YWMJFEIKliAWpJRxIcNU1zY\n7WtLVEgYtqMXz3mlqzClsnGqIMAJTf9Es+CMxKD7mhAVwLcDpzvQZ2ioGrs4wwXq\nNGLay7FTAgMBAAECggEAFRWuFtQ+uymZYtwUfMq3wKgxPPqBgkwwgirhh60rRlSM\npHa71l9MDG6ZBB7ZhK+fpfy2rviOnCwASqvezQH0K1ggi8aYcfSAnSgJPN4Xn8ZE\n/DLcN2jCw7/sS1Z6rAW6yRHWnVGV5DWm5pLuerdIFpImwbt5fJYbbaIuSSXZdk1/\nLG8FZjSvzBFCjXOKfF+W/YN/abTT38zHeUfVdPdt6ciKFbo6JfR4FyhAjRz33oZN\nCubK9hRXbzcz4BvwQ+IDhYRr2pqIO48GrDIEw3d0mTiNwV9JYCAC/JiRONNFACJL\nHRSrYfAhfC8ijOJI/xEp33VGWPL9F92kwP3WU1mz1QKBgQC/3EI2ZWqoXPdIGLyE\nnGU06nB0i+P1GdkNodYBYXu/Qr9Cn/wgK04VoORVdSqSEB0r2Gi/j9hrOIInZfSv\nyEJnNxK5pnkoldxbNkxiU/ZvyOM/3uD+B92SS/TlkJB5NaSGFrxpA+I0QEtgjqgY\niL92oAL7dxCnYEw88FHBD0nedQKBgQC6cBXXkxlu1MmZhWyFlkonlhMoPypGz2O7\no+M3OgLNtTFzXPilQXZPhbGvawbXeFvGaB4L3OjiM5CMNI7AWzi1nj14KlH2FC9p\nnR+RKoMAsIGTS3BitBFv5S/nT+YKqD5WavkFYy5dCV0VzY/uUQSBXUz944PRh3/Q\n8g1p1mDnpwKBgG7ObMcxx2m5V2+iKa6FDMaE57HH4T37UapX31soy+loSUHXWCvU\nFuLS60yXwKBfAhoCgGyyTRGPr0vFzI2BISivW/cwuCTCeGONdowLZfallOmcdWEB\ndew6RhQXa/k1C/INS39zKL94qo/3lAmnYLzIKTDGUS35pc5EFVNk2wrJAoGAP7Sx\nIfqXxDFdueHHlVYnfKNhZG1BCvUuxR1ZNLPT5Wq2vQ7Vv9JAlSe/8YsGyXXNFlzZ\nd4BC65hnGiGTbdM964Foy7jaTNXU5afU84utO/0UKbqram7RToTn+4hnuNiIhIsF\nEHw1iYD7l8moFu0ENxgkiNTHZD+Cw2kSgEnKzx0CgYEAtrcZVvVl6C/5Awu0c/tm\ngdewaqurNdrutkQJnfEsYdyzdrZjoC1HqfGHdLAQ0OoyscVnI1aqHQc6/hWpZp+1\nqKY5xnkocUFcAHacUqqWp7wFirax3tJUBGOC+hZsC/zvXrDnV1tnShvLYWh9q9dj\nfxISL0Tf7OIvckiFegnL7ns=\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-ele3c@sgih-4b054.iam.gserviceaccount.com",
  "client_id": "118247825974048574181",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-ele3c%40sgih-4b054.iam.gserviceaccount.com"
}  #Firebase Credentials File removed for security purposes

certi = firebase_admin.credentials.Certificate(cred)
fireapp = firebase_admin.initialize_app(certi, {'storageBucket': 'sgih-4b054.appspot.com/','databaseURL': 'https://sgih-4b054.firebaseio.com/'})

emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

face_image  = cv2.imread("index1.jpg")
plt.imshow(face_image)

print (face_image.shape)

# resizing the image
face_image = cv2.resize(face_image, (48,48))
#face_image = cv2.resize(face_image, (64,64))
face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])

model = load_model("model_v6_23.hdf5")
#model = load_model("_mini_XCEPTION.102-0.66.hdf5")

print(face_image.shape)

predicted_class = np.argmax(model.predict(face_image))

label_map = dict((v,k) for k,v in emotion_dict.items()) 
predicted_label = label_map[predicted_class]

print(predicted_label)


@app.route("/upload", methods = ['POST'])
def upload():
  print(flask.request)
  if flask.request.method == "POST":
    print("Uploading File")
    if flask.request.files["file"]:
      print("Reading input")
      weights = flask.request.files["file"].read()
      print("Input Read")
      weights_stream = io.BytesIO(weights)
      bucket = storage.bucket()
      blob = bucket.blob('nf')
      print("Saving at Server")
      with open("file.txt", "wb") as f:
        f.write(weights_stream.read())
      print("Starting upload to Firebase")
      with open("file.txt", "rb") as upload:
        blob = bucket.blob('diary' + str(ctr + 1))
        blob.upload_from_file(upload)
        print("File Successfully Uploaded to Firebase")
        ref.update({'num': ctr + 1})
        return "File Uploaded\n"
    else:
      print("File not found")