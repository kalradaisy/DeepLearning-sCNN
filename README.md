# DeepLearning-sCNN

Various packages that are needed to run CNN training and inference codes-
export LD_LIBRARY_PATH=/usr/nevis/gcc-9.2.0/lib/:/usr/nevis/gcc-9.2.0/lib64/:/usr/local/cuda/lib64

export PATH=/usr/nevis/gcc-9.2.0/bin/:${PATH}

Install Anaconda (using https://urldefense.proofpoint.com/v2/url?u=https-3A__docs.anaconda.com_anaconda_install_linux_-23&d=DwIDaQ&c=009klHSCxuh5AI1vNQzSO0KGjl4nbi2Q0M1QLJX9BeE&r=wrsPG6YeYqUrSNqAmhi0G2lBNhv6m2f8ycmaD-oma3o&m=0qpEZxt_ZK2u_73QZVO2CTZJ3h4gjjD41Wnx7KKTX3tKVcTAZ7tkUTBcgNWGluD1&s=KpVAydeHqCWJ75nYsT8xSplkCZ8TbEt2e0pyzVNRQyU&e= )

conda env list

cd /data/kalra

source anaconda/bin/activate

conda install pytorch=1.7.1 torchvision cudatoolkit=10.2 -c pytorch -c  conda-forge
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://urldefense.proofpoint.com/v2/url?u=https-3A__download.pytorch.org_whl_torch-5Fstable.html&d=DwIDaQ&c=009klHSCxuh5AI1vNQzSO0KGjl4nbi2Q0M1QLJX9BeE&r=wrsPG6YeYqUrSNqAmhi0G2lBNhv6m2f8ycmaDoma3o&m=0qpEZxt_ZK2u_73QZVO2CTZJ3h4gjjD41Wnx7KKTX3tKVcTAZ7tkUTBcgNWGluD1&s=4B5ym0VLOwmi6kiBZW6Qujt2chn3QDxQ3AkPWpc9ySY&e= 

conda install openblas-devel -c anacondapython

import torch

torch.cuda.is_available()—> should be True

git clone https://urldefense.proofpoint.com/v2/url?u=https-3A__github.com_NVIDIA_MinkowskiEngine.git&d=DwIDaQ&c=009klHSCxuh5AI1vNQzSO0KGjl4nbi2Q0M1QLJX9BeE&r=wrsPG6YeYqUrSNqAmhi0G2lBNhv6m2f8ycmaD-oma3o&m=0qpEZxt_ZK2u_73QZVO2CTZJ3h4gjjD41Wnx7KKTX3tKVcTAZ7tkUTBcgNWGluD1&s=ElSp9bEp25afjjHix_GitEKf6n8CWnJ0ELoo9jbWdl0&e= 

cd MinkowskiEngine/

edit line 146 of setup.py to libraries = [‘minkowski’, ‘openblas’]
edit line 21 of Makefile to BLAS := open
make clean

pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" —install-option=“--blas=openblas"

Modify examples/training.pyline
142: coordinates=input,
143: features=feats,203: loss = criterion(out.F.squeeze(), 
torch.max(labels, 1) [1])

Run:python -m examples.trainingOnce 
set up the following: 
export LD_LIBRARY_PATH=/usr/nevis/gcc-9.2.0/lib/:/usr/nevis/gcc-9.2.0/lib64/:/usr/local/cuda/lib64
export PATH=/usr/nevis/gcc-9.2.0/bin/:${PATH}

conda env list
cd /data/kalra
source anaconda/bin/activate
python
import torch
torch.cuda.is_available()—> should be True
cd MinkowskiEngine
python -m examples.training
