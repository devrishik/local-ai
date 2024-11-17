from llama_cpp import Llama

class LocalLanguageModel:
    """Wrapper for local language models to use with DSPy using llama.cpp."""
    
    def __init__(self, 
                 model_path: str = "models/llama-2-7b-chat.gguf",
                 n_ctx: int = 2048,
                 n_gpu_layers: int = -1, # offload to gpu
                 temperature: float = 0.7):
        """Initialize llama.cpp model."""
        
        # Initialize llama.cpp model
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,  # Context window
            n_gpu_layers=n_gpu_layers,  # Number of layers to offload to GPU
            verbose=False
        )
        
        # Required kwargs for DSPy compatibility
        self.kwargs = {
            "temperature": temperature,
            "max_tokens": 512,
            "top_p": 0.9,
            "top_k": 50,
            "repeat_penalty": 1.1
        }
        
    def __call__(self, prompt: str, **kwargs) -> list:
        """Make the class callable for DSPy compatibility."""
        response = self.generate(prompt, **kwargs)
        return [{"text": response}]
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using llama.cpp."""
        formatted_prompt = f"[INST] {prompt} [/INST]"
        
        response = self.model(
            formatted_prompt,
            max_tokens=kwargs.get('max_tokens', 512),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
            top_k=kwargs.get('top_k', 50),
            repeat_penalty=kwargs.get('repeat_penalty', 1.1),
            echo=False  # Don't include prompt in response
        )
        
        # Extract generated text
        if isinstance(response, dict):
            return response.get('choices', [{}])[0].get('text', '').strip()
        elif isinstance(response, list) and response:
            return response[0]['choices'][0]['text'].strip()
        return ''