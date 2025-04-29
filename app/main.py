from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# โหลดโมเดล, Scaler, และ PCA
knn = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")
knn2 = joblib.load("models/model2.pkl")
scaler2 = joblib.load("models/scaler2.pkl")
pca2= joblib.load("models/pca2.pkl")

# สร้าง FastAPI app
app = FastAPI()

# อนุญาตทุก Origins (ทุก Domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ถ้าเป็น production ควรกำหนด origin ที่เฉพาะเจาะจง
    allow_credentials=True,
    allow_methods=["*"],  # อนุญาตทุก Methods (GET, POST, PUT, DELETE ฯลฯ)
    allow_headers=["*"],  # อนุญาตทุก Headers
)

# สร้าง Data Model สำหรับรับข้อมูล
class InputData(BaseModel):
    features: list[float]

@app.get("/")
def read_root():
    return "This is ML API!!"

@app.post("/predict")
def predict(data: InputData):
    try:
        print("Received data:", data.features)  

        # แปลงข้อมูลเป็น NumPy array
        input_data = np.array(data.features).reshape(1, -1)
        
        # Standardize ข้อมูลด้วย Scaler
        input_scaled = scaler.transform(input_data)
        
        # ลดมิติข้อมูลด้วย PCA
        input_pca = pca.transform(input_scaled)
        
        # ทำนายผลด้วย KNN
        prediction = knn.predict(input_pca)

        # แปลงผลลัพธ์จากตัวเลขเป็น AQI bucket


        return {"prediction": str(prediction)}  # ส่งผลลัพธ์กลับไป
    except Exception as e:
        return {"error": str(e)}    
    
@app.post("/predict-2")
def predict(data: InputData):
    try:
        print("Received data:", data.features)  

        # แปลงข้อมูลเป็น NumPy array
        input_data = np.array(data.features).reshape(1, -1)
        
        # Standardize ข้อมูลด้วย Scaler
        input_scaled = scaler2.transform(input_data)
        
        # ลดมิติข้อมูลด้วย PCA
        input_pca = pca2.transform(input_scaled)
        
        # ทำนายผลด้วย KNN
        prediction = knn2.predict(input_pca)


        return {"prediction": str(prediction)}  # ส่งผลลัพธ์กลับไป
    except Exception as e:
        return {"error": str(e)}