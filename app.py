
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
# from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

MODEL = tf.keras.models.load_model("./model/model_padiq_sparse.h5", custom_objects = {"KerasLayer" : hub.KerasLayer})
CLASS_NAMES = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']
@app.get("/ping")
async def ping():
    return "Hello, World!"


def read_file_as_image(data) -> np.ndarray :
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image =  read_file_as_image (await file.read())
    image_resize = tf.image.resize(image,[224,224])
    image_batch = np.expand_dims(image_resize,0)
    prediction = MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        'class' : predicted_class,
        'confidence' : float(confidence)
    }

if __name__ == "__app__" :
     uvicorn.run(app, host = '127.0.0.1', port = 8000, debug = True)