# Simsopt and Docker Containers


This document explains how to build docker container for simsopt. It also provides
instructions on how to run simsopt docker container


## Build the container

0. Install docker
1. Build the docker image by running the `docker build` command:

   ```bash
   docker build -t <[user_name/]image_name:[tag]> -f Dockerfile.ubuntu .
   # In the above command, user_name is typically used for uploading the image to docker hub
   # For local builds, you can omit the user_name/ and just use your choice of image_name
   # Tag is optional. If not given, default is latest
   ```
2. Upload the image to docker hub if you want to distribute it
   ```bash
   docker push <image_name:tag>
   ```
   

## Run the container
There are two cases here: 1) docker image is built locally and 2) docker image downloaded from a repo

### For local builds:
0. Identify the local image using the command
   ```bash
    docker image ls
    ```
    After identifying the image built locally, you can use either <image_name:tag> or the image id, which 
    is an hexadecimal number, to run the docker image in the next command

### Remote repo
0. Assuming image is uploaded to docker hub, use the <user_name/image_name:tag> to automatically download
   the image in the next command

1. For interactive runs, run the docker container with the command:
   ```bash
   docker run -it --rm  <image>
   ```

2. To execute a python script
   ```bash
   docker run -v$PWD:$PWD <image> python3 $PWD/<python_script>
   ```
   **This command works only if no additional files are needed to run the script**



