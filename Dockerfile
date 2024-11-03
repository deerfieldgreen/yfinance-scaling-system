FROM python:3.9-slim  # Use a lightweight Python base image

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]  

# Command to run your Python script
# k