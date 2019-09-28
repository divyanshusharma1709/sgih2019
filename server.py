from __future__ import print_function
import json
from os.path import join, dirname
from ibm_watson import ToneAnalyzerV3
from ibm_watson.tone_analyzer_v3 import ToneInput
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

# If service instance provides API key authentication
service = ToneAnalyzerV3(
    ## url is optional, and defaults to the URL below. Use the correct URL for your region.
    url='https://gateway-syd.watsonplatform.net/tone-analyzer/api',
    version='2017-09-21',
    iam_apikey='Yp62E9xjIkN-9BCHgbV2ihn2Aj1LhZD8P74zCFpmkfqN')

certi = firebase_admin.credentials.Certificate(cred)
fireapp = firebase_admin.initialize_app(certi, {'storageBucket': 'sgih-4b054.appspot.com/','databaseURL': 'https://sgih-4b054.firebaseio.com/'})

@app.route("/upload", methods = ['POST'])
def upload():
	ref = db.reference('/')
	z = ref.get()
	data=z['text']
	tone_analysis = service.tone({'text': data},content_type='application/json').get_result()
	print(json.dumps(tone_analysis, indent=2))



# service = ToneAnalyzerV3(
#     ## url is optional, and defaults to the URL below. Use the correct URL for your region.
#     # url='https://gateway.watsonplatform.net/tone-analyzer/api',
#     username='YOUR SERVICE USERNAME',
#     password='YOUR SERVICE PASSWORD',
#     version='2017-09-21')


'''
text = 'Team, I know that times are tough! Product '\
    'sales have been disappointing for the past three '\
    'quarters. We have a competitive product, but we '\
    'need to do a better job of selling it!'

tone_analysis = service.tone(
    {'text': text},
    content_type='application/json'
).get_result()
print(json.dumps(tone_analysis, indent=2))
'''