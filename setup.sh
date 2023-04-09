export CONDA_ENV_NAME=vif
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.9

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

which python
which pip

pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

conda activate $CONDA_ENV_NAME