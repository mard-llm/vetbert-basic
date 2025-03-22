import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import semantic_kernel as sk
from semantic_kernel.kernel import Kernel
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion, OllamaChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.core_plugins import TextMemoryPlugin
from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore
from sentence_transformers import SentenceTransformer
import asyncio
import nest_asyncio
import os

nest_asyncio.apply()

device = torch.device("cpu")
print(f"Using device: {device}")

try:
    model_name = "havocy28/VetBERTDx"
    vetbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    vetbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    vetbert_model.to(device)
    vetbert_model.eval()
    print("VetBERTDx loaded successfully.")
except Exception as e:
    print(f"Error loading VetBERTDx: {e}")
    raise

kernel = Kernel()

ollama_service = OllamaChatCompletion(
    ai_model_id="llama3.1:8b",
    service_id="ollama_llama",
    host="http://localhost:11434"
)
kernel.add_service(ollama_service)

class HuggingFaceEmbeddingService:
    def __init__(self, model_id: str, device: torch.device):
        self.model = SentenceTransformer(model_id, device=device)

    async def generate_embeddings(self, texts: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

embedding_service = HuggingFaceEmbeddingService(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    device=device
)

memory = SemanticTextMemory(storage=VolatileMemoryStore(), embeddings_generator=embedding_service)
kernel.add_plugin(TextMemoryPlugin(memory), "memory")

async def load_knowledge_base():
    file_path = "vet_knowledge.txt"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            lines = f.read().splitlines()
        for i, line in enumerate(lines):
            if line.strip():
                await memory.save_information(
                    collection="vet_knowledge",
                    id=f"vet_info_{i}",
                    text=line.strip()
                )
        print("Vet knowledge loaded into volatile memory.")
    else:
        print("Warning: vet_knowledge.txt not found.")

class VetBERTDxPlugin:
    @kernel_function(description="Detect diseases from text", name="vetbert_process")
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
            print(f"VetBERTDx: {sorted_probs[:2]}")
            if sorted_probs[0][1].lower() == "unknown":
                return "None"
            return ", ".join(f"{label}: {prob:.4f}" for prob, label in sorted_probs[:2])
        except Exception as e:
            print(f"VetBERTDx error: {e}")
            return "Error in disease detection"

class OllamaPlugin:
    @kernel_function(description="Analyze with RAG", name="ollama_analyze")
    async def ollama_analyze(self, prompt: str, vetbert_result: str, history: str) -> str:
        try:
            system_prompt = (
                "You are MARD, a veterinary assistant by DCA CUSAT students. "
                "Use the provided vet knowledge and detected diseases to assist. "
                "For greetings (e.g., 'hi'), introduce yourself. For symptoms, analyze and recommend, "
                "always advising to consult a vet. If no disease is detected, use retrieved info or suggest next steps."
            )
            history_str = history if history else ""

            greeting_keywords = {"hi", "hello", "hey", "who are you"}
            is_greeting = prompt.lower().strip() in greeting_keywords or vetbert_result.startswith("blank")

            if is_greeting:
                retrieved_context = "N/A (greeting detected)"
            else:
                query = f"{prompt} {vetbert_result}" if vetbert_result not in ["None", "blank"] else prompt
                search_results = await memory.search(collection="vet_knowledge", query=query, limit=5)
                print(f"RAG Search Results: {[f'{result.text[:50]}... (Score: {result.relevance:.4f})' for result in search_results]}")
                retrieved_context = "\n".join(
                    [result.text for result in search_results if result.relevance > 0.5]
                ) if search_results else "No relevant vet info found."
            print(f"RAG Retrieved: {retrieved_context}")

            full_prompt = (
                f"User input: {prompt}\n"
                f"Detected disease(s): {vetbert_result if vetbert_result != 'None' else 'Unknown'}\n"
                f"Vet Knowledge: {retrieved_context}\n"
                f"Respond appropriately."
            )
            print(f"Ollama prompt: {full_prompt}")

            chat_history = ChatHistory()
            chat_history.add_system_message(system_prompt)
            chat_history.add_user_message(full_prompt)

            settings = OllamaChatPromptExecutionSettings(max_tokens=200, temperature=0.7)
            response = await ollama_service.get_chat_message_content(chat_history=chat_history, settings=settings)
            output = response.content
            print(f"Ollama output: {output}")
            return output.strip()
        except Exception as e:
            print(f"Ollama error: {e}")
            return "Error in analysis"

kernel.add_plugin(VetBERTDxPlugin(), plugin_name="VetBERTDx")
kernel.add_plugin(OllamaPlugin(), plugin_name="Ollama")

async def chat(prompt: str, history: list) -> str:
    history_str = "\n".join([f"<|user|>\n{h[0]}\n<|assistant|>\n{h[1]}" for h in history]) if history else ""
    print(f"Processing: {prompt}")
    
    vetbert_result = await kernel.invoke(function_name="vetbert_process", plugin_name="VetBERTDx", text=prompt)
    print(f"VetBERTDx result: {vetbert_result.value}")
    
    ollama_result = await kernel.invoke(
        function_name="ollama_analyze",
        plugin_name="Ollama",
        prompt=prompt,
        vetbert_result=vetbert_result.value,
        history=history_str
    )
    print(f"Ollama output: {ollama_result.value}")
    return ollama_result.value

def gradio_chat(prompt, history):
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(chat(prompt, history))

interface = gr.ChatInterface(
    gradio_chat,
    title='MARD',
    description='A veterinary chatbot with RAG for disease prediction and recommendations.'
)

asyncio.run(load_knowledge_base())

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7868, share=False)
