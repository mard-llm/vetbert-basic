# vetbert-basic
A basic AI tool for veterinary disease classification and recommendations.


# Usage on Linux
```
git clone https://github.com/mard-llm/vetbert-basic
cd vetbert-basic
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cat models/vetBERTDx.model_* > models/vetBERTDx.model
wget -o models/Mistral-7B-v0.1-GGUF https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_S.gguf?download=true
python mard.py
```
