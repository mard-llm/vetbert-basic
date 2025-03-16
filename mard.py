import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import semantic_kernel as sk
from semantic_kernel.kernel import Kernel
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion, OllamaChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory
import asyncio
import nest_asyncio

nest_asyncio.apply()

device = torch.device("cpu")
print(f"Using device: {device}")

try:
    model_name = "havocy28/VetBERTDx"
    vetbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    vetbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    vetbert_model.to(device)
    vetbert_model.eval()
    print("VetBERTDx model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading VetBERTDx model or tokenizer: {e}")
    raise

kernel = Kernel()

ollama_service = OllamaChatCompletion(
    ai_model_id="llama3.1:8b",
    service_id="ollama_llama",
    host="http://localhost:11434"
)
kernel.add_service(ollama_service)

class VetBERTDxPlugin:
    @kernel_function(description="Detect diseases from text using VetBERTDx", name="vetbert_process")
    def vetbert_process(self, text: str) -> str:
        try:
            inputs = vetbert_tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                logits = vetbert_model(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1)
            label_map = vetbert_model.config.id2label
            sorted_probs = sorted(
                ((prob.item(), label_map[idx]) for idx, prob in enumerate(probabilities[0])),
                reverse=True,
                key=lambda x: x[0]
            )
            print(f"VetBERTDx output: {sorted_probs[:2]}")
            if sorted_probs[0][1].lower() == "unknown":
                return "None"
            return ", ".join(f"{label}: {prob:.4f}" for prob, label in sorted_probs[:2])
        except Exception as e:
            print(f"VetBERTDx error: {e}")
            return "Error in disease detection"

class OllamaPlugin:
    @kernel_function(description="Analyze disease or respond to casual queries", name="ollama_analyze")
    async def ollama_analyze(self, prompt: str, vetbert_result: str, history: str) -> str:
        try:
            system_prompt = (
                "You are a highly knowledgeable veterinary assistant named MARD, developed by students of DCA CUSAT. "
                "Your primary function is to help users diagnose potential pet diseases and offer preliminary recommendations based on symptoms provided. "
                "If the input is a casual greeting (e.g., 'hi', 'hello') or a question about your identity (e.g., 'who are you'), respond conversationally—introduce yourself and offer help without assuming a health issue. "
                "For symptom-related inputs, provide a clear, concise, empathetic, and professional analysis, always reminding users to consult a qualified veterinarian for an accurate diagnosis and treatment. "
                "If no clear symptoms or diseases are detected (e.g., 'blank' or 'unknown' as the top result), acknowledge the lack of information and suggest next steps."
            )
            history_str = history if history else ""
            full_prompt = (
                f"User input: {prompt}\n"
                f"Detected disease(s): {vetbert_result if vetbert_result != 'None' else 'Unknown'}\n"
                f"Respond appropriately based on the input and detected diseases."
            )
            print(f"Ollama prompt: {full_prompt}")

            chat_history = ChatHistory()
            chat_history.add_system_message(system_prompt)
            chat_history.add_user_message(full_prompt)

            execution_settings = OllamaChatPromptExecutionSettings(
                max_tokens=200,
                temperature=0.7
            )

            response = await ollama_service.get_chat_message_content(
                chat_history=chat_history,
                settings=execution_settings
            )
            output = response.content
            print(f"Ollama output: {output}")
            return output.strip()
        except Exception as e:
            print(f"Ollama error: {e}")
            return "Error in analysis and recommendation generation"

kernel.add_plugin(VetBERTDxPlugin(), plugin_name="VetBERTDx")
kernel.add_plugin(OllamaPlugin(), plugin_name="Ollama")

async def chat(prompt: str, history: list) -> str:
    try:
        history_str = "\n".join([f"<|user|>\n{h[0]}\n<|assistant|>\n{h[1]}" for h in history]) if history else ""
        print(f"Processing prompt: {prompt}, History: {history_str}")
        
        vetbert_result = await kernel.invoke(
            function_name="vetbert_process",
            plugin_name="VetBERTDx",
            text=prompt
        )
        print(f"VetBERTDx result: {vetbert_result.value}")
        
        ollama_result = await kernel.invoke(
            function_name="ollama_analyze",
            plugin_name="Ollama",
            prompt=prompt,
            vetbert_result=vetbert_result.value,
            history=history_str
        )
        print(f"Final output: {ollama_result.value}")
        
        return ollama_result.value
    except Exception as e:
        print(f"Chat error: {e}")
        return "Sorry, an error occurred."

def gradio_chat(prompt, history):
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(chat(prompt, history))

interface = gr.ChatInterface(
    gradio_chat,
    title='MARD',
    description='A veterinary chatbot for disease prediction and recommendations.'
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)
