import dspy
from typing import List, Dict
import random


class QAModule(dspy.Module):
    """Module for question answering with reasoning."""
    
    def __init__(self):
        super().__init__()
        
        # Define the signatures for reasoning and answer generation
        self.reason = dspy.ChainOfThought("question -> reasoning")
        self.answer = dspy.Predict("question, reasoning -> answer")
    
    def forward(self, question: str) -> dict:
        """Generate reasoned answer for a question."""
        # First generate reasoning
        reasoning = self.reason(question=question).reasoning
        
        # Then generate answer based on reasoning
        answer = self.answer(question=question, reasoning=reasoning).answer
        
        return {
            "reasoning": reasoning,
            "answer": answer
        }


def create_training_data() -> List[Dict[str, str]]:
    """Create training data with questions, reasoning, and answers."""
    return [
        {
            "question": "Explain quantum computing",
            "reasoning": "To explain quantum computing, I should break it down into key concepts: 1) Quantum mechanics principles like superposition and entanglement, 2) Comparison with classical computing, 3) Practical applications",
            "answer": "Quantum computing uses quantum mechanics principles like superposition and entanglement to perform computations. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits that can exist in multiple states simultaneously."
        },
        {
            "question": "What is machine learning?",
            "reasoning": "To explain machine learning, I should cover: 1) Its relationship to AI, 2) The core concept of learning from data, 3) How it differs from traditional programming",
            "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicit programming. It uses statistical techniques to allow computers to 'learn' from data."
        }
    ]

def evaluate_reasoning(pred, gold):
    """Custom evaluation function for reasoning quality."""
    if not pred.get('reasoning'):
        return 0.0
            
    # Basic checks for reasoning quality
    reasoning = pred['reasoning'].lower()
    has_structure = any(word in reasoning for word in ['first', 'second', 'then', 'because', 'therefore'])
    has_depth = len(reasoning.split()) >= 20
    
    # Compare answer with gold standard
    answer_similarity = len(set(pred['answer'].split()) & set(gold['answer'].split())) / len(set(gold['answer'].split()))
    
    # Combine metrics
    score = (0.4 * has_structure + 0.3 * has_depth + 0.3 * answer_similarity)
    return score

def optimize_with_miprov2(num_rounds: int = 3):
    """
    Optimize prompts using MiProv2.
    
    Args:
        num_rounds: Number of optimization rounds
    """
    # Initialize language model
    # local_lm = LocalLanguageModel()
    # dspy.settings.configure(lm=local_ministral_3b.model)

    # from local_ai.ml.cpp import LocalLanguageModel
    # local_lm = LocalLanguageModel(
    #     model_path="C:/workspace/models/Ministral-3b-instruct-Q4_0.gguf")

    # local_lm = dspy.OllamaLocal(model='ministral')

    # from local_ai.ml.pytorch import LocalMinistral3b
    # local_lm = LocalMinistral3b()

    from local_ai.ml.pytorch import LocalMistral
    local_lm = LocalMistral()

    dspy.settings.configure(lm=local_lm)

    
    # Create training data
    train_data = create_training_data()
    
    # Initialize the module
    qa_module = QAModule()
    

    # Initialize MiProv2
    miprov2 = dspy.MIPROv2(
        metric=evaluate_reasoning,
        auto="light"
    )
    
    # Compile module with MiProv2
    compiled_module = miprov2.compile(
        qa_module,
        trainset=train_data,
        requires_permission_to_run=False
    )
    
    return compiled_module

def evaluate_module(optimizer, test_inputs: List[str]):
    """
    Evaluate the optimized module.
    
    Args:
        optimizer: Compiled DSPy module
        test_inputs: List of test questions
    """
    results = []
    for question in test_inputs:
        try:
            output = optimizer(question=question)
            results.append({
                "question": question,
                "reasoning": output["reasoning"],
                "answer": output["answer"],
                "status": "success"
            })
        except Exception as e:
            results.append({
                "question": question,
                "reasoning": None,
                "answer": str(e),
                "status": "error"
            })
    return results

if __name__ == "__main__":
    # Example usage
    test_questions = [
        "What is deep learning?",
        "Explain neural networks",
        "How does reinforcement learning work?"
    ]
    
    # Optimize module with MiProv2
    optimized_module = optimize_with_miprov2(num_rounds=3)
    
    # Evaluate results
    results = evaluate_module(optimized_module, test_questions)
    
    # Print results
    for result in results:
        print(f"\nQuestion: {result['question']}")
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Reasoning: {result['reasoning']}")
            print(f"Answer: {result['answer']}")
        else:
            print(f"Error: {result['answer']}")

# if __name__ == "__main__":
#     # Example usage with local model
#     test_inputs = [
#         """create a event scraper which web scrapes tripadvisor.com using crawl4ai.
#          The UI is a simple streamlit app which suggests events for any location based on the user's query.
#          Its suggestions include detailed steps on how to book that event""",
#     ]

#     # Optimize prompts with local model
#     optimized_module = optimize_prompts()
    
#     # Evaluate results
#     results = evaluate_prompts(optimized_module, test_inputs)
    
#     # Print results
#     for result in results:
#         print(f"\nInput: {result['input']}")
#         print(f"Status: {result['status']}")
#         print(f"Output: {result['output']}")
