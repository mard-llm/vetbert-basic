import gradio as gr
from llama_cpp import Llama
from transformers import AutoTokenizer
import torch
import joblib
import os

# Load VetBERTDx model and tokenizer
try:
    vetbert_model = joblib.load('models/vetBERTDx.model')
    vetbert_tokenizer = joblib.load('models/vetBERTDx.tokenizer')
except Exception as e:
    print(f"Error loading VetBERTDx model or tokenizer: {e}")
    raise

# Load Mistral model
try:
    llm = Llama(model_path="models/mistral-7b-v0.1.Q4_K_S.gguf", n_gpu_layers=0, n_threads=8)
except Exception as e:
    print(f"Error loading Mistral model: {e}")
    raise

def vetbert_process(text):
    inputs = vetbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = vetbert_model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)
    label_map = vetbert_model.config.id2label
    sorted_probs = sorted(((prob.item(), label_map[idx]) for idx, prob in enumerate(probabilities[0])), reverse=True, key=lambda x: x[0])
    
    # Check if the top result is "unknown"
    if sorted_probs[0][1].lower() == "unknown":
        return None  # Indicates that VetBERTDx cannot recognize the disease
    else:
        top_2_results = sorted_probs[:2]  # Take only the top 2 results
        return ", ".join(f"{label}: {prob:.4f}" for prob, label in top_2_results)











   












    

   


    






def chat(prompt, history):
    system_prompt = (
        "You are highly knowledgeable veterinary assistant developed by students of DCA CUSAT. "
        "Your primary function is to help users diagnose potential pet diseases and offer preliminary recommendations based on symptoms provided. "
        "You utilize advanced AI models to analyze veterinary health data and assist users in understanding their pets' conditions. "
        "However, you are not a substitute for a professional veterinarian. Always remind users to consult a qualified vet for an accurate diagnosis and treatment. "
        "Provide clear, concise, and professional responses, ensuring empathy and care when addressing health concerns."
    )
    
    # Format chat history
    history_str = "\n".join([f"<|user|>\n{h[0]}\n<|assistant|>\n{h[1]}" for h in history])
    
    # Process input with VetBERTDx first
    vetbert_response = vetbert_process(prompt)
    
    # Prepare final prompt for Mistral
    if vetbert_response is None:
        # If VetBERTDx returns "unknown", pass the user input directly to Mistral
        full_prompt = f"<|system|>\n{system_prompt}\n{history_str}\n<|user|>\n{prompt}\n<|assistant|>\n"
    else:
        # Include VetBERTDx analysis in the prompt
        full_prompt = f"<|system|>\n{system_prompt}\n{history_str}\n<|user|>\n{prompt}\nVetBERTDx Analysis: {vetbert_response}\n<|assistant|>\n"
    
    print(full_prompt)
    
    # Streaming response from Mistral
    response_stream = llm.create_completion(
        full_prompt,
        max_tokens=200,
        stop=["<|user|>", "<|assistant|>", "\n"],  # Stop generation at these tokens
        stream=True
    )
    
    response_text = ""
    for chunk in response_stream:
        if "choices" in chunk and chunk["choices"]:
            response_text += chunk["choices"][0]["text"]
            yield response_text.strip()

# Gradio Interface
gradio = gr.ChatInterface(
    chat,
    title='MARD',
    description='A veterinary chatbot for disease prediction and recommendations.'

)

gradio.launch(server_name="0.0.0.0", server_port=7860, share=False)
