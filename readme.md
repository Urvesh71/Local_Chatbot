# Chatbot Application

This is a Python-based chatbot application designed to run inside a Docker container. The chatbot is capable of answering user queries, providing automated responses, or interacting with users in various ways depending on the business logic implemented.

## Project Structure

Here is an overview of the project structure:

chatbot-project/
│
├── Dockerfile               # Instructions to build the Docker image for the chatbot
├── docker-compose.yml       # Configuration for Docker Compose to manage services
├── README.md                # Documentation for setting up and running the application
├── chatbot.py               # Main Python script for the chatbot logic
└──ollama-entrypoint.sh      # To install the necessary model from Ollama and start the Ollama Server

- **Dockerfile**: This file contains instructions for building the Docker image for the chatbot application. This file also contains the packages required for the application.
- **docker-compose.yml** : Defines the services required to run the chatbot using Docker Compose.
- **chatbot.py** : Contains the main code logic for the chatbot.
- **README.md** : The file you're currently reading; provides comprehensive instructions for setting up, running, and deploying the chatbot application.
- **ollama-entrypoint.sh** : This script is used to set up and run the necessary models for the chatbot application. This script performs the following actions.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Docker**: Make sure Docker is installed and running on the machine.
- **Docker Compose**: Ensure Docker Compose is installed. Docker Desktop usually comes with Docker Compose by default.

## Installation Instructions

To set up the chatbot application on your local machine, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone 
   cd 

## Running the Application

To run the chatbot application using Docker Compose, you have two options depending on your needs:

### 1. `docker-compose up --build`

Use this command when you need to **rebuild the Docker images**:

- If you have made changes to the `Dockerfile`.
- If you have updated the application code or dependencies.
- If you want to ensure any new files or configuration changes are included in the image.
- If you want to use the latest version of the base image specified in the `Dockerfile`.

### 2. `docker-compose up `

- If no changes have been made to the Dockerfile, code, or dependencies.
- If docker container is already created and need to only activate it. 

## Running the application in Detached mode:

On the server, you can run Docker Compose in detached mode to keep the application running in the background:

### 1. `docker-compose up --build -d`

## Stopping the Application:

To stop the running containers, use:

### 1. `docker-compose down`

