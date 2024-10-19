FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the contents of your local project to the /app directory in the container
COPY . .

# Install the required dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Gradio will use
EXPOSE 7860

# Set the Gradio server to listen on all interfaces
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Command to run the application
CMD ["python", "app.py"]