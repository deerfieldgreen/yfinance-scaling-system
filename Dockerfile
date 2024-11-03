FROM python:3.9-slim  # Use a lightweight Python base image

WORKDIR /app

# Copy the requirements file into the working directory and install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy all other files into the container's working directory
COPY . .

# Command to run your Python script
CMD ["python", "main.py"]
