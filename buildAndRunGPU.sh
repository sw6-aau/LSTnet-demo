#!/bin/bash
echo -e "\e[38;5;202m-===CLEANING===-\n\e[0m"
docker stop lstDockGPU
docker rm lstDockGPU
echo -e "\n\e[38;5;4m-===BUILDING===-\n\e[0m"
sudo docker build -t lst-gpu ./dockerfiles/gpu
echo -e "\n\e[38;5;2m-===RUNNING===-\n\e[0m"
sudo docker run --gpus all -it --name lstDockGPU lst-gpu