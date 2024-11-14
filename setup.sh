#!/bin/bash

set -e

sudo apt-get update
sudo apt-get install -y \
    libgl1-mesa-glx

python3 -m venv .venv
source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt