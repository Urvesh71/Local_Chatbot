#!/bin/sh
echo "Starting Ollama server..."
ollama serve &

# Wait for Ollama to start
sleep 10
echo "Ollama server started."

echo "Pulling 'llama3:8b-instruct-q6_K' model..."
ollama pull llama3:8b-instruct-q6_K

echo "'llama3:8b-instruct-q6_K' model pulled successfully."

echo "Pulling 'nomic-embed-text' model..."
ollama pull nomic-embed-text:latest

echo "'nomic-embed-text' model pulled successfully."

# Keep the container running
tail -f /dev/null
