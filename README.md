# PinPoint_Code

# LLaVA Environment
conda create -n llava-next python=3.10 -y <br \>
conda activate llava-next <br \>
conda install nvidia/label/cuda-12.1.1::cuda-toolkit <br \>
pip install --upgrade pip <br \>
pip install -e ".[train]" <br \>
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 <br \>
pip install flash-attn==2.5.8 --no-build-isolation <br \>
pip install accelerate==0.28.0 <br \>
pip install anls <br \>

# Qwen Environment
conda create -n qwen_ft python=3.10 -y <br \>
conda activate qwen_ft <br \>
conda install nvidia/label/cuda-12.2.2::cuda-toolkit -y <br \>
pip install torch==2.6.0 torchvision==0.21.0 deepspeed==0.17.1 triton==3.2.0 accelerate==1.7.0 torchcodec==0.2 peft==0.17.1 <br \>
pip install -e . <br \>
pip install qwen-vl-utils==0.0.14 <br \>
pip install matplotlib <br \>