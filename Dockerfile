# Dockerfile for RTD Integration
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install required packages
RUN apt-get update && apt-get install -y \
    g++ \
    make \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Make scripts executable
RUN chmod +x compile_all.sh

# Compile the project
RUN ./compile_all.sh

# Set the default command
CMD ["./bin/test_raytracedicom"]
