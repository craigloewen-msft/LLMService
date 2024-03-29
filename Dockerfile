FROM python:3.9

# Set the working directory
WORKDIR /app

# Move over everything to the working directory
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Run the Python script
CMD ["python", "main.py"]