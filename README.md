# 1. Create the conda environment with Python 3.12
conda create -n visnarr_env python=3.12 -y

# 2. Activate it
conda activate visnarr_env

# 3. Install all packages with pip in the same env
python -m pip install -r requirements.txt

# 4. Install spacy model
python -m spacy download en_core_web_sm

# 5. Run code
python main.py
