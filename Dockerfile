# use pytorch GPU enabled container
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Move over everything to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

COPY . .

# Run the Python script
CMD ["python", "main.py"]