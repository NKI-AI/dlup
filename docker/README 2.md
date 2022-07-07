# Docker for DLUP
A Dockerfile is provided for DLUP which provides you with all the required dependencies.

To build the container, run the following command from the root directory:
```
docker build -t dlup:latest . -f docker/Dockerfile
```

Running the container can for instance be done with:
```
docker run -it --ipc=host --rm -v /data:/data dlup:latest /bin/bash
```
