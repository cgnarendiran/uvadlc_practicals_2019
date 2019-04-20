module purge
module load eb
module load Miniconda3/4.3.27
module load CUDA/9.0.176
module load cuDNN/7.3.1-CUDA-9.0.176

conda env create -f ../environment.yml
