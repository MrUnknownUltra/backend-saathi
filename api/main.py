from io import BytesIO
from pyexpat import model
from fastapi import FastAPI, File, Form, UploadFile,HTTPException,Depends,Query
from fastapi.responses import JSONResponse
import joblib
import cv2
import numpy as np
from requests import Session
import tensorflow as tf
import uvicorn
from PIL import Image
from loguru import logger
import pandas as pd
# from sqlalchemy.orm import Session

from fastapi.middleware.cors import CORSMiddleware
import base64
import google.generativeai as genai
# models.Base.metadata.create_all(bind=engine)
import os
import io
from PIL import Image
# Create a FastAPI application
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

img ="C:/Users/tusha/OneDrive/Desktop/testcrop/api/lateblight.JPG"
app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",  # Add your frontend origin here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # You can restrict the methods if needed
    allow_headers=["*"],  # You can restrict the headers if needed
)

# # Dependency
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# @app.post("/users/", response_model=schemas.User)
# def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
#     db_user = crud.get_user_by_email(db, email=user.email)
#     if db_user:
#         raise HTTPException(status_code=400, detail="Email already registered")
#     return crud.create_user(db=db, user=user)


# @app.get("/users/", response_model=list[schemas.User])
# def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
#     users = crud.get_users(db, skip=skip, limit=limit)
#     return users


# @app.get("/users/{user_id}", response_model=schemas.User)
# def read_user(user_id: int, db: Session = Depends(get_db)):
#     db_user = crud.get_user(db, user_id=user_id)
#     if db_user is None:
#         raise HTTPException(status_code=404, detail="User not found")
#     return db_user


# @app.post("/users/{user_id}/items/", response_model=schemas.Item)
# def create_item_for_user(
#     user_id: int, item: schemas.ItemCreate, db: Session = Depends(get_db)
# ):
#     return crud.create_user_item(db=db, item=item, user_id=user_id)


# @app.get("/items/", response_model=list[schemas.Item])
# def read_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
#     items = crud.get_items(db, skip=skip, limit=limit)
#     return items

# Load the saved model and scaler
loaded_scaler = joblib.load('scaler.joblib')
loaded_model = tf.keras.models.load_model('../crop_model')

# Define API endpoint for prediction
@app.post("/soilpredict", response_model=None)
async def predict(data: dict):
    try:
        df = pd.read_csv('Crop_recommendation.csv')
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data])

        # Standardize the input data
        input_data_scaled = loaded_scaler.transform(input_data)

        # Make predictions using the loaded model
        predictions = loaded_model.predict(input_data_scaled)

        # Convert predictions to crop names
        predicted_crop_index = np.argmax(predictions)
        predicted_crop = df['label'].unique()[predicted_crop_index]

        return {"predicted_crop": predicted_crop}
    

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Load the saved model and scaler
loaded_waterscaler = joblib.load('scalerwater.joblib')
loaded_watermodel = tf.keras.models.load_model('../water_model')
# Define API endpoint for prediction
@app.post("/waterpredict", response_model=None)
async def predict(data: dict):
    try:
        df = pd.read_csv('water_potability.csv')
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data])

        # Standardize the input data
        input_data_scaled = loaded_waterscaler.transform(input_data)

        # Make predictions using the loaded model
        predictions = loaded_watermodel.predict(input_data_scaled)

        # Convert predictions to crop names
        predicted_water_index = np.argmax(predictions)
        potability_label  = df['Potability'].unique()[predicted_water_index]

        return {"Potability": int(potability_label)}
    

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"Hello": "World"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
MODEL=tf.keras.models.load_model('../models/2')
@app.post("/croppredict")
async def predict(file: UploadFile = File(...)):
    try:
        # Logging: Received prediction request
        logger.info("Received prediction request.")

        # Read the file as an image
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)

        # Logging: Input data
        logger.info("Received input image batch: %s", img_batch)

        print("Starting model prediction...")
        predictions = MODEL.predict(img_batch)
        print("Model prediction completed.")

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        # Encode the image as base64
        _, img_encoded = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(img_encoded).decode()

        # Logging: Prediction result
        logger.info("Prediction successful.")

        # Return the response with image and prediction result
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'image': img_base64
        }

    except Exception as e:
        # Logging: Error during prediction
        logger.error("Error during prediction: %s", str(e))
        # Raise HTTPException with proper status code and detail
        return {"error": "Internal Server Error"}

GOOGLE_API_KEY = "AIzaSyCcHoanogS4RP-4Wlr7oupTNNwvUrH7OPg"
genai.configure(api_key=GOOGLE_API_KEY)

def get_geminivison_response(img):
    model = genai.GenerativeModel('gemini-pro-vision')
    promt="""  {
        "crop_name": crop_name,
        "disease": disease,
        "accuracy": accuracy,
        "Cause": cause in minimum 500 words,
        "Cure": cure in minimum 500 words 
    }"""
    response = model.generate_content([promt, img],stream=False)
    return response.text

def get_realtime_trans(input):
    model=genai.GenerativeModel('gemini-pro')
    promt=f"translate it in hindi language exactly as it is :{input}"
    response=model.generate_content(promt)
    return response.text


def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    
    return response.text

@app.post("/cropimage")
async def  cropimage(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))
    response = get_geminivison_response(img)
    return {"response":response}
    

@app.get("/cropresult")
async def crop_result(disease: str = Query(..., title="Disease Name")):
    question = f"Tell me in points the cause and cure of potato {disease} simple language"
    response = get_gemini_response(question)
    return {"disease": disease, "response": response}

@app.post("/translate")
async def translate(text):
    response=get_realtime_trans(text)
    return { "response":response}

if __name__ == "__main__":
    # Set up logging configuration
    logger.add("app.log", rotation="500 MB", level="INFO")

    # Run the application with UVicorn
    uvicorn.run(app, host="127.0.0.0", port=8000)

'''
json sample input for soilpredict
{
    "N": 10,
    "P": 5,
    "K": 20,
    "temperature": 25,
    "humidity": 60,
    "ph": 7,
    "rainfall": 50
}

json sample input for waterpredict
{
  "ph": 7.5,
  "Hardness": 150,
  "Solids": 300,
  "Chloramines": 0.8,
  "Sulfate": 200,
  "Conductivity": 500,
  "Organic_carbon": 15,
  "Trihalomethanes": 50,
  "Turbidity": 5
}
'''