import pyrebase

from config import config


firebase = pyrebase.initialize_app(config.FIREBASE_CONFIG)
auth = firebase.auth()
db = firebase.database()
storage = firebase.storage()

import cv2
import datetime
import numpy as np
from flask import current_app as app
from google.cloud import storage

def upload_frame_to_firebase(frame, user_id, timestamp, folder="threat"):
    storage_client = storage.Client()
    bucket = storage_client.bucket(app.config['FIREBASE_STORAGE_BUCKET'])

    filename = f"{user_id}/{folder}/{timestamp.replace(':', '-').replace(' ', '_')}.png"
    _, buffer = cv2.imencode('.png', frame)
    blob = bucket.blob(filename)
    blob.upload_from_string(buffer.tobytes(), content_type='image/png')
