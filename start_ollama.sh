#!/bin/sh
echo "Starting Ollama server..."
ollama serve &

# Wait for Ollama to start
sleep 10
echo "Ollama server started."

echo "Pulling 'llama3.1:8b' model..."
ollama pull llama3.1:8b

echo "'llama3.1:8b' model pulled successfully."

echo "Pulling 'nomic-embed-text' model..."
ollama pull nomic-embed-text:latest

echo "'nomic-embed-text' model pulled successfully."

# Keep the container running
tail -f /dev/null
