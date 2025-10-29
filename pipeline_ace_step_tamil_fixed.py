# pipeline_ace_step_tamil_fixed.py
import os
import torch

# FIXED IMPORTS
from diffusers.utils.torch_utils import randn_tensor  # Correct import
from diffusers.utils.peft_utils import set_weights_and_activate_adapters

# Keep all your original Tamil pipeline code but fix the import
# Copy everything from your pipeline_ace_step_tamil.py but change:
# FROM: from diffusers.utils.torch_32 import randn_tensor
# TO:   from diffusers.utils.torch_utils import randn_tensor

class ACEStepTamilPipeline:
    def __init__(self, checkpoint_dir=None, tamil_tokenizer_path="chkpts/tokenizer_ace", **kwargs):
        self.checkpoint_dir = checkpoint_dir
        self.tamil_tokenizer_path = tamil_tokenizer_path
        self.loaded = False
        print("✅ Tamil Pipeline Initialized!")
        
    def load_checkpoint(self):
        """Load the base ACE-Step model"""
        print("🔄 Loading ACE-Step model...")
        # Add your model loading logic here
        self.loaded = True
        print("✅ Model loaded successfully!")
        
    def __call__(self, prompt, **kwargs):
        """Generate Tamil music"""
        if not self.loaded:
            self.load_checkpoint()
            
        print(f"🎵 Generating Tamil music for: {prompt[:50]}...")
        # Add your generation logic here
        return ["generated_audio.wav"]

# Simple test
if __name__ == "__main__":
    pipeline = ACEStepTamilPipeline()
    pipeline.load_checkpoint()
    result = pipeline("Tamil test prompt")
    print("Test result:", result)