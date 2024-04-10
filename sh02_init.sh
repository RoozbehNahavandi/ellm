#!/bin/bash
module load cuda/11.8.0
MINICONDA_DIR="/users/PAS2138/roozbehn99/miniconda3"

Install dependencies using conda
echo "Installing dependencies using conda..."
conda env create -f conda_env.yml

# Activate the conda environment
echo "Activating conda environment..."
source ${MINICONDA_DIR}/bin/activate text-crafter

echo "Environment setup complete."
