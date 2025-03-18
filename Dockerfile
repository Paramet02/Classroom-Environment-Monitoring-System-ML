# เริ่มจากใช้ Python 3.9 image
FROM python:3.9-slim

# ตั้ง directory สำหรับแอปพลิเคชัน
WORKDIR /app

# คัดลอกไฟล์ requirements.txt ที่ใช้ติดตั้ง dependencies
COPY requirements.txt .

# ติดตั้ง dependencies ที่ระบุใน requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์โค้ดทั้งหมดไปยัง container
COPY . .

# เปิดพอร์ตที่ FastAPI ใช้งาน (default: 8000)
EXPOSE 8000

# รันแอป FastAPI ด้วย Uvicorn โดยระบุเส้นทางของ main.py
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]