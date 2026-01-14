### This is the repo for the article `A Novel Method for the Computational Extraction and Analysis of Visual Narratives`.
This repo contains all the code used for extracting narratives. 
The main logic is invoked by the 'main.py' script.
The creation of temporal narratives is in a separate module with the same name.

### Usage
**1. Create the conda environment with Python 3.12**
conda create -n visnarr_env python=3.12 -y

**2. Activate it**
conda activate visnarr_env

**3. Install all packages with pip in the same env**
python -m pip install -r requirements.txt

**4. Install spacy model**
python -m spacy download en_core_web_sm

**5. Run code**
python main.py

