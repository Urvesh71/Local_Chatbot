# Use the official Python image from Docker Hub
FROM python:3.10.10-slim

# Set the working directory
WORKDIR /app

# Install necessary system dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    libpoppler-cpp-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install the latest SQLite version
RUN SQLITE_VERSION=$(wget -qO- https://www.sqlite.org/download.html | grep -oP 'sqlite-autoconf-\d{7}\.tar\.gz' | head -1 | grep -oP '\d{7}') \
    && wget https://www.sqlite.org/2024/sqlite-autoconf-$SQLITE_VERSION.tar.gz \
    && tar xvfz sqlite-autoconf-$SQLITE_VERSION.tar.gz \
    && cd sqlite-autoconf-$SQLITE_VERSION \
    && ./configure \
    && make \
    && make install \
    && cd .. \
    && rm -rf sqlite-autoconf-$SQLITE_VERSION* \
    && ldconfig

# Verify SQLite version
RUN sqlite3 --version

# Copy the application code
COPY . /app/

# Install Python dependencies
RUN pip install chromadb==0.5.3
RUN pip install streamlit==1.36.0
RUN pip install langchain_core==0.2.9
RUN pip install langchain_community==0.2.5
RUN pip install PyPDF2
RUN pip install pypdf==4.2.0

# Expose the ports
EXPOSE 8501

# Set environment variables
# ENV BASE_URL=http://ollama:11434
ENV BASE_URL=http://host.docker.internal:11434

# Run the application
CMD ["streamlit", "run", "chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]