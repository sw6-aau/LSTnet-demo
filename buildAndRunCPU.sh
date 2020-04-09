#!/bin/bash
echo -e "\e[38;5;202m-===CLEANING===-\n\e[0m"
docker stop lstDockCPU
docker rm lstDockCPU
echo -e "\n\e[38;5;4m-===BUILDING===-\n\e[0m"
sudo docker build -t lst-cpu ./dockerfiles/gpu
echo -e "\n\e[38;5;2m-===RUNNING===-\n\e[0m"
sudo docker run --gpus all -it --name lstDockCPU lst-cpu