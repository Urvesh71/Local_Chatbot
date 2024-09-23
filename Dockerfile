# Use an NVIDIA CUDA base image
FROM nvidia/cuda:11.5.0-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app

# Install Python 3.10 and necessary tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-venv \
    wget \
    build-essential \
    libpoppler-cpp-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip for Python 3.10
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

# Verify Python and pip versions
RUN python3 --version && pip --version

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
WORKDIR /app

# Install PyTorch compatible with CUDA 11.5
RUN pip install --no-cache-dir \
    torch==1.11.0+cu115 \
    torchvision==0.12.0+cu115 \
    torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu115

# Install other Python dependencies
RUN pip install --no-cache-dir \
    chromadb==0.5.3 \
    streamlit==1.36.0 \
    langchain_core==0.2.9 \
    langchain_community==0.2.5 \
    PyPDF2 \
    pypdf==4.2.0 \
    langdetect==1.0.9

# Expose the port
EXPOSE 8501

# Set environment variables
ENV BASE_URL=http://ollama:11434

# Run the application
CMD ["streamlit", "run", "chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]
