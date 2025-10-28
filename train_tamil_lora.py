# train_tamil_lora.py
import os
import argparse
from trainer import Pipeline
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from datetime import datetime

class TamilPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        # Add Tamil-specific initialization
        self.tamil_tokenizer = None
        super().__init__(*args, **kwargs)
    
    def setup_tamil_components(self):
        """Setup Tamil-specific components"""
        try:
            from tamil_tokenizer_src.tokenizer.tamil_tokenizer_ace import AceStyleTamilTokenizer
            # Load your Tamil tokenizer
            self.tamil_tokenizer = AceStyleTamilTokenizer.from_pretrained("chkpts/tokenizer_ace")
            print(f"✅ Tamil tokenizer loaded with vocab size: {self.tamil_tokenizer.vocab_size}")
        except Exception as e:
            print(f"⚠️ Could not load Tamil tokenizer: {e}")
            self.tamil_tokenizer = None

def main():
    parser = argparse.ArgumentParser()
    
    # Tamil-specific arguments
    parser.add_argument("--tamil_dataset_path", type=str, default="./tamil_dataset_ace_step/hf_dataset")
    parser.add_argument("--tamil_tokenizer_path", type=str, default="chkpts/tokenizer_ace")
    parser.add_argument("--lora_config_path", type=str, default="config/tamil_lora_config.json")
    parser.add_argument("--exp_name", type=str, default="tamil_tts_lora")
    
    # Training parameters optimized for Tamil (390 samples)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=50000)  # Reduced for small dataset
    parser.add_argument("--every_plot_step", type=int, default=1000)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    
    # ACE-Step original arguments (with Tamil defaults)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--every_n_train_steps", type=int, default=2000)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--logger_dir", type=str, default="./exps/logs/")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--gradient_clip_algorithm", type=str, default="norm")
    parser.add_argument("--reload_dataloaders_every_n_epochs", type=int, default=1)
    parser.add_argument("--val_check_interval", type=int, default=None)
    
    args = parser.parse_args()
    
    print("🚀 Starting Tamil TTS LoRA Training")
    print("=" * 50)
    print(f"📁 Dataset: {args.tamil_dataset_path}")
    print(f"🎵 Samples: 357 Tamil songs")
    print(f"🔤 Tokenizer: Custom Tamil tokenizer")
    print(f"🎯 LoRA Config: {args.lora_config_path}")
    print("=" * 50)
    
    # Create Tamil pipeline
    model = TamilPipeline(
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        shift=args.shift,
        max_steps=args.max_steps,
        every_plot_step=args.every_plot_step,
        dataset_path=args.tamil_dataset_path,
        checkpoint_dir=args.checkpoint_dir,
        adapter_name=args.exp_name,
        lora_config_path=args.lora_config_path
    )
    
    # Setup Tamil components
    model.setup_tamil_components()
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=None,
        every_n_train_steps=args.every_n_train_steps,
        save_top_k=-1,
    )
    
    # Add datetime to version
    logger_callback = TensorBoardLogger(
        version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.exp_name,
        save_dir=args.logger_dir,
    )
    
    # Create trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy="ddp_find_unused_parameters_true",
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        log_every_n_steps=1,
        logger=logger_callback,
        callbacks=[checkpoint_callback],
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        reload_dataloaders_every_n_epochs=args.reload_dataloaders_every_n_epochs,
        val_check_interval=args.val_check_interval,
    )

    print("🎯 Starting training...")
    trainer.fit(
        model,
        ckpt_path=args.ckpt_path,
    )

if __name__ == "__main__":
    main()