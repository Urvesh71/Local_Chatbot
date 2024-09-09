#!/bin/sh

# Pull the necessary models
ollama pull nomic-embed-text:latest
ollama pull mistral:latest

# Start the Ollama server
ollama serve