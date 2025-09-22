#!/usr/bin/env python3
"""
Test script for Llama-3.2-1B-Instruct using Hugging Face Transformers
This script will download and run the Llama-3.2-1B-Instruct model using transformers library
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time


def main():
    """Main function to test Llama-3.2-1B-Instruct model"""
    print("Initializing Llama-3.2-1B-Instruct model...")
    print("This may take a few minutes for first-time download...\n")
    
    # Model name on Hugging Face
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Test prompt
        prompt = "Hello! Can you tell me a short story about a friendly robot?"
        
        print(f"\nPrompt: {prompt}")
        print("Generating response...\n")
        
        # Format the prompt for instruction model
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # Tokenize input
        inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
        
        # Move inputs to the same device as model
        if torch.cuda.is_available():
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        assistant_response = response.split("assistant<|end_header_id|>\n\n")[-1]
        
        generation_time = time.time() - start_time
        
        print("Response:")
        print("-" * 50)
        print(assistant_response)
        print("-" * 50)
        print(f"\nGeneration completed in {generation_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during model loading or generation: {e}")
        print("\nNote: If you get permission errors, you may need to:")
        print("1. Accept the license agreement on the Hugging Face model page")
        print("2. Login to Hugging Face using: huggingface-cli login")
        print("3. Or use a different model variant that doesn't require authentication")
    
    print("\nTest completed!")


if __name__ == "__main__":
    main()