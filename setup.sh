mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -f ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda create -y -n sam python=3.11
~/miniconda3/bin/conda init
exec bash
conda activate sam
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install biopython scikit-learn pandas