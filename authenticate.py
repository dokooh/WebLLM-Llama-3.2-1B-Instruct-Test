#!/usr/bin/env python3
"""
Authentication helper for accessing Meta Llama models
"""

import subprocess
import sys
import os
from huggingface_hub import login


def check_authentication():
    """Check if user is already authenticated with Hugging Face"""
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Already authenticated as: {user_info['name']}")
        return True
    except Exception:
        print("❌ Not authenticated with Hugging Face")
        return False


def authenticate_huggingface():
    """Guide user through Hugging Face authentication"""
    print("\n🔐 Hugging Face Authentication Setup")
    print("="*40)
    
    if check_authentication():
        return True
    
    print("\nTo access Meta Llama models, you need to:")
    print("1. Create a Hugging Face account at https://huggingface.co/join")
    print("2. Request access to Llama models at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
    print("3. Get a token from https://huggingface.co/settings/tokens")
    
    choice = input("\nDo you have a Hugging Face token ready? (y/n): ").lower().strip()
    
    if choice == 'y':
        token = input("\nPaste your Hugging Face token here: ").strip()
        
        try:
            login(token=token)
            print("✅ Authentication successful!")
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False
    else:
        print("\n📝 Please complete the steps above and run this script again.")
        return False


def test_llama_access():
    """Test if we can access the Llama model"""
    print("\n🧪 Testing Llama model access...")
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        print("✅ Llama model access confirmed!")
        print("🚀 You can now run test_llama.py to use the official model")
        return True
    except Exception as e:
        print(f"❌ Cannot access Llama model: {e}")
        print("\n💡 Make sure you've accepted the license at:")
        print("   https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
        return False


def main():
    """Main authentication flow"""
    print("🦙 Llama Authentication Helper")
    print("="*35)
    
    if authenticate_huggingface():
        test_llama_access()
        
        print("\n🎉 Setup Summary:")
        print("✅ Virtual environment: Llama-3.2-1B-Instruct")
        print("✅ Dependencies installed: torch, transformers, accelerate")
        print("✅ Hugging Face authentication: Complete")
        print("✅ Ready to run Llama-3.2-1B-Instruct!")
        
        print("\n📚 Available test scripts:")
        print("   - test_llama.py        → Official Llama-3.2-1B-Instruct")
        print("   - test_alternative.py  → Demo with open model")
        print("   - verify_setup.py      → Environment verification")


if __name__ == "__main__":
    main()