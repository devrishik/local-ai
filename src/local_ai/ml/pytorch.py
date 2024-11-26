import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dsp.modules.mistral import Mistral

class LocalLanguageModel:
    """Wrapper for local language models to use with DSPy."""
    
    def __init__(self, model_name: str = "microsoft/phi-2", temperature: float = 0.7):
        """Initialize local model and tokenizer."""
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",  # Explicitly use CUDA, change to `auto` for CPU
            low_cpu_mem_usage=True,  # Optimize CPU memory usage during loading
            cache_dir="/c/workspace/models",
            token=True,
            # load_in_4bit=True
        )

        # Add required kwargs for DSPy compatibility
        self.kwargs = {
            "temperature": temperature,
            "max_tokens": 512,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2
        }
        self.model.kwargs = self.kwargs
        

    def __call__(self, prompt: str, **kwargs) -> list:
        """Make the class callable for DSPy compatibility."""
        response = self.generate(prompt, **kwargs)
        return [{"text": response}]
    
    def generate(self, prompt: str, max_length: int = 512) -> str:
        """Generate text using the local model."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class LocalMistral(LocalLanguageModel):
    def __init__(self, model_name: str = "mistralai/Mistral-1B-Instruct-v0.2"):
        super().__init__(model_name)
    
    def generate(self, prompt: str, max_length: int = 512) -> str:
        """Generate text using the local model."""
        # Format prompt according to Mistral's instruction format
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = torch.tensor(inputs, dtype=torch.int32).to(self.model.device)
        # Debugging: Check the type and content of inputs
        print("Inputs:", inputs)  # Check the structure of inputs
        print("Input IDs:", inputs['input_ids'])  # Check input_ids specifically
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    temperature=self.temperature,
                    max_length=max_length,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    # Mistral specific parameters
                    top_p=0.95,
                    repetition_penalty=1.1,
                    top_k=40
                )
        except Exception as e:
            print(f"Generation error: {e}")
            return str(e)
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the instruction prompt from the output
        return generated_text.split("[/INST]")[-1].strip()

class LocalMistral2(Mistral):
    pass

class LocalMinistral3b(LocalMistral):
    def __init__(self, model_name: str = "ministral/Ministral-3b-instruct"):
        super().__init__(model_name)

class LocalLlama3(LocalLanguageModel):
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
        super().__init__(model_name)
    
    def generate(self, prompt: str, max_length: int = 512) -> str:
        """Generate text using the local model."""
        # Format prompt according to TinyLlama's chat format
        formatted_prompt = f"<|system|>You are a helpful assistant.</s><|user|>{prompt}</s><|assistant|>"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                # TinyLlama specific parameters
                top_p=0.9,
                repetition_penalty=1.2,
                top_k=50,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("</s>")
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the system and user prompt
        response = generated_text.split("<|assistant|>")[-1].strip()
        return response
