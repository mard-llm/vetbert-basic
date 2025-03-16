# vetbert-basic
A basic AI tool for veterinary disease classification and recommendations.


# Usage on Linux
```
# Clone the repository
git clone https://github.com/mard-llm/vetbert-basic.git

# Navigate into the directory
cd vetbert-basic

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies from requirements.txt
pip3 install -r requirements.txt

# Install and start Ollama, then pull LLaMA 3.1 8B
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.1:8b

# Run the chatbot
python3 mard.py
```
