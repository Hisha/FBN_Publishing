# Install environment and pip installs 
# to run flux_schenll_cpu
python3 -m venv FBNP_env 

source FBNP_env/bin/activate

pip install --upgrade pip 

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install diffusers transformers accelerate safetensors scipy sentencepiece protobuf

#Using Real-ESRGAN for upscaling https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases

pip install pytz FastAPI dotenv itsdangerous bcrypt 
