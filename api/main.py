from urllib import response
from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
import flask
import tensorflow_hub as hub

model = None
interpreter = None
input_index = None
output_index = None

class_names = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

BUCKET_NAME = "padiq-model" # Here you need to put the name of your GCP bucket


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def cors_configuration(bucket_name):
    """Set a bucket's CORS policies configuration."""
    # bucket_name = "your-bucket-name"

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    bucket.cors = [
        {
            "origin": ["*"],
            "responseHeader": ["*"],
            "method": ["*"],
            "maxAgeSeconds": 3600
        }
    ]
    bucket.patch()

    print(f"Set CORS policies for bucket {bucket.name} is {bucket.cors}")
    return bucket



def prediction(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/model_padiq.h5",
            "/tmp/model_padiq.h5",
        )
        model = tf.keras.models.load_model("/tmp/model_padiq.h5", custom_objects = {"KerasLayer" : hub.KerasLayer})
        cors_configuration(BUCKET_NAME)

    image = request.files["file"]

    image = np.array(
        Image.open(image).convert("RGB").resize((150, 150)) # image resizing
    )

    image = image/255 # normalize the image in 0 to 1 range

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    print("Predictions:",predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]),2)
    if confidence < 78:
        predicted_class = 'Unknown'
        confidence = 0
    response = {"class": predicted_class, "confidence": confidence}
    responses = flask.jsonify(response)
    return responses

    def diseaseArticle():
        