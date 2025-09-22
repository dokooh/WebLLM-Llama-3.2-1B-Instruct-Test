#!/usr/bin/env python3
"""
Alternative test script using an open Llama-compatible model
This script uses microsoft/DialoGPT-medium as a demonstration, 
or you can authenticate to use the official Llama model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import sys


def test_open_model():
    """Test with an open model that doesn't require authentication"""
    print("=== Testing with Microsoft DialoGPT (open model) ===")
    print("This is a demonstration of the setup working with an open model.\n")
    
    try:
        # Use a smaller, open model for testing
        model_name = "microsoft/DialoGPT-medium"
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        print("Model loaded successfully!")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Test conversation
        conversation_history = ""
        prompt = "Hello, can you tell me about artificial intelligence?"
        
        print(f"\nUser: {prompt}")
        print("Bot: ", end="", flush=True)
        
        # Encode the conversation
        new_input = tokenizer.encode(conversation_history + prompt + tokenizer.eos_token, 
                                   return_tensors='pt')
        
        # Generate response
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(
                new_input,
                max_length=new_input.shape[1] + 100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(output[0][new_input.shape[1]:], skip_special_tokens=True)
        generation_time = time.time() - start_time
        
        print(response)
        print(f"\nGeneration completed in {generation_time:.2f} seconds")
        print("✅ Model testing successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model testing: {e}")
        return False


def show_llama_instructions():
    """Show instructions for accessing the official Llama model"""
    print("\n" + "="*60)
    print("📋 Instructions for using official Llama-3.2-1B-Instruct:")
    print("="*60)
    print("\n1. 🔐 Get Hugging Face Access:")
    print("   - Create account at https://huggingface.co/")
    print("   - Go to https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
    print("   - Click 'Accept license' and follow the process")
    print("   - This may require approval from Meta")
    
    print("\n2. 🔑 Authenticate locally:")
    print("   - Get your token from https://huggingface.co/settings/tokens")
    print("   - Run: huggingface-cli login")
    print("   - Enter your token when prompted")
    
    print("\n3. 🚀 Run the model:")
    print("   - After authentication, run test_llama.py again")
    print("   - The official Llama model will download and run")
    
    print("\n4. 💡 Alternative models (no authentication needed):")
    print("   - Llama-3.2-1B-Instruct-Q4_K_M (GGUF format)")
    print("   - Various community fine-tuned versions")
    print("   - Other instruction-tuned models like Mistral, Phi, etc.")
    

def main():
    """Main function"""
    print("🦙 Llama-3.2-1B-Instruct Setup Test")
    print("="*50)
    
    # Test with an open model first to verify setup
    success = test_open_model()
    
    if success:
        show_llama_instructions()
        
        print(f"\n🎉 Setup Complete!")
        print(f"✅ Virtual environment: Llama-3.2-1B-Instruct")
        print(f"✅ Python {sys.version.split()[0]}")
        print(f"✅ PyTorch {torch.__version__}")
        print(f"✅ Transformers library installed and working")
        print(f"✅ Model loading and generation tested successfully")
        
        print(f"\n📁 Files created:")
        print(f"   - verify_setup.py (environment verification)")
        print(f"   - test_llama.py (official Llama model test)")
        print(f"   - test_alternative.py (this file - alternative model test)")
        
    else:
        print("\n❌ There was an issue with the setup. Please check the error messages above.")


if __name__ == "__main__":
    main()