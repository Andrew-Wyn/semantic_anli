conda create --name aganlis python=3.9 -y
conda activate aganlis

# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

python -c "import torch; print("torch_version:", torch.__version__); print("cuda_version:", torch.version.cuda); print("cuda available:", torch.cuda.is_available())"

pip install -r requirements.txt
conda install matplotlib -y