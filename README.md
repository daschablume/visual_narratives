### Extracting narratives from visual data
<img width="618" height="568" alt="traveling_narr_cop_m" src="https://github.com/user-attachments/assets/00ff0540-9f61-4272-a419-746f8ecf756c" />

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




