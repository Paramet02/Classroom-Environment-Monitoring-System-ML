from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# โหลดโมเดล, Scaler, และ PCA
knn = joblib.load("models/knn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")  # โหลด LabelEncoder ด้วย

# สร้าง FastAPI app
app = FastAPI()

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
        predicted_aqi_bucket = label_encoder.inverse_transform([prediction])[0]

        return {"prediction": str(predicted_aqi_bucket)}  # ส่งผลลัพธ์กลับไป
    except Exception as e:
        return {"error": str(e)}
