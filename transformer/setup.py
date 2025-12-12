
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader
import pandas as pd
import numpy as n
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime


from config import Config
from beamSearch import BeamSearch
from transformer import Transformer
from bpeTokenizer import BPETokenizer
from bleu import BLEUScore
from summarizer import SummarizationDataset



# GET CONFIG
config = Config()

# CHECKPOINT LOADING AND SAVING
def save_checkpoint(epoch, model, optimizer, scheduler, scaler, val_loss, train_losses, val_losses, val_bleu_scores, config):
    """Save training checkpoint with atomic write to prevent corruption"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'val_loss': val_loss,
        'train_losses': train_losses, 
        'val_losses': val_losses, 
        'val_bleu_scores': val_bleu_scores,
        'config': {
            'n_encoder_layers': config.n_encoder_layers,
            'n_decoder_layers': config.n_decoder_layers,
            'd_model': config.d_model,
            'n_heads': config.n_heads,
            'd_ff': config.d_ff,
            'vocab_size': config.vocab_size,
            'max_seq_len': config.max_seq_len,
            'dropout': config.dropout
        }
    }
    
    temp_file = config.model_file + '.tmp'
    try:
        torch.save(checkpoint, temp_file)
        if os.path.exists(config.model_file):
            os.replace(temp_file, config.model_file)
        else:
            os.rename(temp_file, config.model_file)
        print(f"[Success] Checkpoint saved to {config.model_file}")
    except Exception as e:
        print(f"[Error] Error saving checkpoint: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)

def load_checkpoint(model, optimizer, scheduler, scaler, config):
    """Load training checkpoint if it exists"""
    if not os.path.exists(config.model_file):
        print("No checkpoint found. Starting training from scratch.")
        return 0, float('inf'), [], [], []
    
    print(f"Loading checkpoint from {config.model_file}...")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(config.model_file, map_location=config.device, weights_only=False)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state loaded")
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded")
        
        # Load scheduler state if available
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded")
        
        # Load scaler state if available
        if scaler and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("Scaler state loaded")
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        # Load loss histories (with defaults for backward compatibility)
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        val_bleu_scores = checkpoint.get('val_bleu_scores', [])
        
        print(f"[Success] Resuming from epoch {start_epoch}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Loaded {len(train_losses)} training loss records")
        
        return start_epoch, best_val_loss, train_losses, val_losses, val_bleu_scores
        
    except (RuntimeError, EOFError, pickle.UnpicklingError, Exception) as e:
        print(f"✗ Error loading checkpoint: {e}")
        print("✗ Checkpoint file appears to be corrupted.")
        
        # Create backup of corrupted file
        backup_file = config.model_file + '.corrupted'
        print(f"Creating backup at: {backup_file}")
        try:
            os.rename(config.model_file, backup_file)
            print("[Success] Corrupted checkpoint backed up.")
        except:
            print("[Error] Could not backup corrupted file. Deleting it.")
            os.remove(config.model_file)
        
        print("Starting training from scratch.")
        return 0, float('inf'), [], [], []

def collate_fn(batch):
    """Collate function with padding"""
    articles = [item['article'] for item in batch]
    summaries = [item['summary'] for item in batch]

    # Pad sequences
    article_lens = [len(a) for a in articles]
    summary_lens = [len(s) for s in summaries]

    max_article_len = max(article_lens)
    max_summary_len = max(summary_lens)

    padded_articles = torch.zeros(len(batch), max_article_len, dtype=torch.long)
    padded_summaries = torch.zeros(len(batch), max_summary_len, dtype=torch.long)

    for i, (article, summary) in enumerate(zip(articles, summaries)):
        padded_articles[i, :len(article)] = article
        padded_summaries[i, :len(summary)] = summary

    return {
        'article': padded_articles,
        'summary': padded_summaries,
        'article_mask': (padded_articles != 0).unsqueeze(1).unsqueeze(2),
        'summary_mask': (padded_summaries != 0).unsqueeze(1).unsqueeze(2)
    }



def validate(model, dataloader, criterion, device, config, tokenizer=None, compute_bleu=False):
    """Validate model (faster version) with optional BLEU computation"""
    model.eval()
    total_loss = 0
    num_batches = 0

    references = []
    hypotheses = []
    max_bleu_batches = 20  

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating', dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar):
            src = batch['article'].to(device, non_blocking=True)
            tgt = batch['summary'].to(device, non_blocking=True)
            src_mask, tgt_mask = create_masks(src, tgt[:, :-1])
            if config.device == 'cuda':
                with torch.cuda.amp.autocast():
                    output = model(src, tgt[:, :-1], src_mask, tgt_mask)
                    output = output.reshape(-1, output.size(-1))
                    tgt_out = tgt[:, 1:].reshape(-1)
                    loss = criterion(output, tgt_out)
            else:
                output = model(src, tgt[:, :-1], src_mask, tgt_mask)
                output = output.reshape(-1, output.size(-1))
                tgt_out = tgt[:, 1:].reshape(-1)
                loss = criterion(output, tgt_out)

            total_loss += loss.item()
            num_batches += 1
            
            # Compute BLEU on subset
            if compute_bleu and tokenizer and batch_idx < max_bleu_batches:
                # Simple greedy decoding for BLEU (faster than beam search)
                predictions = output.argmax(dim=-1).view(src.size(0), -1)
                
                for i in range(src.size(0)):
                    ref_ids = tgt[i].cpu().tolist()
                    ref_ids = [id for id in ref_ids if id not in [
                        tokenizer.special_tokens['<PAD>'],
                        tokenizer.special_tokens['<SOS>'],
                        tokenizer.special_tokens['<EOS>']
                    ]]
                    references.append(ref_ids)
                    
                    # Hypothesis
                    hyp_ids = predictions[i].cpu().tolist()
                    hyp_ids = [id for id in hyp_ids if id not in [
                        tokenizer.special_tokens['<PAD>'],
                        tokenizer.special_tokens['<SOS>'],
                        tokenizer.special_tokens['<EOS>']
                    ]]
                    hypotheses.append(hyp_ids)

    avg_loss = total_loss / num_batches

    # Compute BLEU if requested
    bleu_score = None
    if compute_bleu and tokenizer and references:
        bleu_scorer = BLEUScore()
        bleu_results = bleu_scorer.compute(references, hypotheses)
        bleu_score = bleu_results['bleu']

    return avg_loss, bleu_score

# TRAINING
def create_masks(src, tgt, pad_id=0):
    """Create masks for source and target"""
    src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
    tgt_pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)
    tgt_mask = tgt_mask & tgt_pad_mask

    return src_mask, tgt_mask

def train_epoch(model, dataloader, optimizer, criterion, device, config, scaler=None):
    """Train for one epoch with mixed precision and gradient accumulation"""
    model.train()
    total_loss = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(dataloader, desc='Training', dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        src = batch['article'].to(device, non_blocking=True)
        tgt = batch['summary'].to(device, non_blocking=True)
        src_mask, tgt_mask = create_masks(src, tgt[:, :-1])

        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(src, tgt[:, :-1], src_mask, tgt_mask)
                output = output.reshape(-1, output.size(-1))
                tgt_out = tgt[:, 1:].reshape(-1)
                loss = criterion(output, tgt_out) / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            output = model(src, tgt[:, :-1], src_mask, tgt_mask)
            output = output.reshape(-1, output.size(-1))
            tgt_out = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt_out) / config.gradient_accumulation_steps

            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * config.gradient_accumulation_steps

        # Update progress bar less frequently
        if batch_idx % config.log_interval == 0:
            pbar.set_postfix({
                'loss': f'{loss.item() * config.gradient_accumulation_steps:.4f}',
                'avg': f'{total_loss/(batch_idx+1):.4f}'
            }, refresh=False)

    return total_loss / len(dataloader)

# VISUALIZATION
def plot_training_curves(train_losses, val_losses, val_bleu_scores, save_path='training_curves.png'):

    # Filter out None values for validation metrics
    val_epochs = [i for i, loss in enumerate(val_losses) if loss is not None]
    val_losses_filtered = [loss for loss in val_losses if loss is not None]
    # val_bleu_filtered = [bleu for bleu in val_bleu_scores if bleu is not None]
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    
    # Plot 1: Training Loss
    axes[0].plot(range(len(train_losses)), train_losses, 'b-', linewidth=2, label='Training Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Training vs Validation Loss
    axes[1].plot(range(len(train_losses)), train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.7)
    if val_losses_filtered:
        axes[1].plot(val_epochs, val_losses_filtered, 'r-', linewidth=2, marker='o', 
                     markersize=6, label='Validation Loss')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # # Plot 3: BLEU Score
    # if val_bleu_filtered:
    #     axes[2].plot(val_epochs, val_bleu_filtered, 'g-', linewidth=2, marker='s', 
    #                  markersize=6, label='Validation BLEU')
    #     axes[2].set_xlabel('Epoch', fontsize=12)
    #     axes[2].set_ylabel('BLEU Score', fontsize=12)
    #     axes[2].set_title('BLEU Score over Epochs', fontsize=14, fontweight='bold')
    #     axes[2].grid(True, alpha=0.3)
    #     axes[2].legend()
        
    #     # Add best BLEU annotation
    #     if val_bleu_filtered:
    #         best_bleu = max(val_bleu_filtered)
    #         best_epoch = val_epochs[val_bleu_filtered.index(best_bleu)]
    #         axes[2].axhline(y=best_bleu, color='r', linestyle='--', alpha=0.5, linewidth=1)
    #         axes[2].text(0.02, 0.98, f'Best BLEU: {best_bleu:.4f} (Epoch {best_epoch+1})', 
    #                     transform=axes[2].transAxes, verticalalignment='top',
    #                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # else:
    #     axes[2].text(0.5, 0.5, 'No BLEU scores available', 
    #                 ha='center', va='center', transform=axes[2].transAxes, fontsize=14)
    #     axes[2].set_title('BLEU Score over Epochs', fontsize=14, fontweight='bold')
    
    # Add overall statistics
    fig.suptitle(f'Training Progress - {len(train_losses)} Epochs', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.99, 0.01, f'Generated: {timestamp}', ha='right', fontsize=8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")    
    plt.close()
    

# MAIN EXECUTION

print("=" * 80)
print("TRANSFORMER TEXT SUMMARIZATION")
print("=" * 80)
print(f"Device: {config.device}")
print(f"Model Depth: {config.n_encoder_layers} encoder layers, {config.n_decoder_layers} decoder layers")
print()

# Load or train tokenizer
print("Step 1: Loading/Training Tokenizer")
print("-" * 80)

tokenizer = None
if os.path.exists(config.tokenizer_file):
    try:
        print(f"Loading tokenizer from {config.tokenizer_file}")
        tokenizer = BPETokenizer.load(config.tokenizer_file)
        print("Tokenizer loaded successfully!")
    except (KeyError, EOFError, pickle.UnpicklingError) as e:
        print(f"Error loading tokenizer: {e}")
        print("Deleting corrupted tokenizer file and training new one...")
        os.remove(config.tokenizer_file)
        tokenizer = None

if tokenizer is None:
    print("Training new tokenizer...")
    # Load training data
    train_df = pd.read_csv(config.train_file)

    texts = train_df['article'].tolist() + train_df['highlights'].tolist()

    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    tokenizer.train(texts)
    tokenizer.save(config.tokenizer_file)
    print(f"Tokenizer saved to {config.tokenizer_file}")

print(f"Vocabulary size: {len(tokenizer.vocab)}")

# Save vocabulary
print("Saving vocabulary...")
with open(config.vocab_file, 'wb') as f:
    pickle.dump(tokenizer.vocab, f)
print(f"Vocabulary saved to {config.vocab_file}")
print()

# Load datasets
print("Step 2: Loading Datasets")
print("-" * 80)

train_df = pd.read_csv(config.train_file)
val_df = pd.read_csv(config.val_file)
test_df = pd.read_csv(config.test_file)

# Sampling
if len(train_df) > config.max_train_samples:
    print(f"Sampling {config.max_train_samples} from {len(train_df)} training samples...")
    train_df = train_df.sample(n=config.max_train_samples, random_state=42)

if len(val_df) > config.max_val_samples:
    print(f"Sampling {config.max_val_samples} from {len(val_df)} validation samples...")
    val_df = val_df.sample(n=config.max_val_samples, random_state=42)

if len(test_df) > config.max_test_samples:
    print(f"Sampling {config.max_test_samples} from {len(test_df)} test samples...")
    test_df = test_df.sample(n=config.max_test_samples, random_state=42)

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")
print()

train_dataset = SummarizationDataset(train_df, tokenizer, config.max_seq_len, config.max_summary_len)
val_dataset = SummarizationDataset(val_df, tokenizer, config.max_seq_len, config.max_summary_len)
test_dataset = SummarizationDataset(test_df, tokenizer, config.max_seq_len, config.max_summary_len)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=config.num_workers,
    pin_memory=config.pin_memory,
    persistent_workers=True if config.num_workers > 0 else False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=config.num_workers,
    pin_memory=config.pin_memory,
    persistent_workers=True if config.num_workers > 0 else False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=config.num_workers,
    pin_memory=config.pin_memory,
    persistent_workers=True if config.num_workers > 0 else False
)

# Create model
print("Step 3: Creating Model")
print("-" * 80)

model = Transformer(
    vocab_size=len(tokenizer.vocab),
    d_model=config.d_model,
    n_heads=config.n_heads,
    n_encoder_layers=config.n_encoder_layers,
    n_decoder_layers=config.n_decoder_layers,
    d_ff=config.d_ff,
    max_seq_len=config.max_seq_len,
    dropout=config.dropout
).to(config.device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Architecture: {config.n_encoder_layers}×Encoder + {config.n_decoder_layers}×Decoder")

if config.compile_model and hasattr(torch, 'compile'):
    print("Compiling model with torch.compile()...")
    try:
        model = torch.compile(model, mode='reduce-overhead')
        print("Model compiled successfully!")
    except Exception as e:
        print(f"Could not compile model: {e}")

print()

# Training setup
criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer.special_tokens['<PAD>'],
    label_smoothing=config.label_smoothing
)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=0.01,
    fused=True if config.device == 'cuda' else False
)

# Learning rate scheduler for better convergence with deeper network
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=config.learning_rate,
    epochs=config.n_epochs,
    steps_per_epoch=len(train_loader) // config.gradient_accumulation_steps,
    pct_start=0.1
)

# Mixed precision training scaler
if config.device == 'cuda':
    try:
        scaler = torch.amp.GradScaler('cuda')
    except (AttributeError, TypeError):
        # Fall back to old API
        scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None

# Load checkpoint if exists
print("Step 4: Checking for Existing Checkpoint")
print("-" * 80)
start_epoch, best_val_loss, train_losses, val_losses, val_bleu_scores = load_checkpoint(
    model, optimizer, scheduler, scaler, config
)
print()

# Training loop
print("Step 5: Training the Network")
print("-" * 80)
print(f"Total epochs: {config.n_epochs}")
print(f"Starting from epoch: {start_epoch + 1}")
print(f"Total batches per epoch: {len(train_loader)}")
print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
print()

for epoch in range(start_epoch, config.n_epochs):
    print(f"\nEpoch {epoch + 1}/{config.n_epochs}")
    epoch_start_time = torch.cuda.Event(enable_timing=True) if config.device == 'cuda' else None
    epoch_end_time = torch.cuda.Event(enable_timing=True) if config.device == 'cuda' else None

    if epoch_start_time:
        epoch_start_time.record()

    train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device, config, scaler)
    train_losses.append(train_loss)  

    if epoch_end_time:
        epoch_end_time.record()
        torch.cuda.synchronize()
        elapsed_time = epoch_start_time.elapsed_time(epoch_end_time) / 1000.0
        print(f"Epoch time: {elapsed_time:.2f}s")

    # Validate only every N epochs
    if (epoch + 1) % config.validate_every_n_epochs == 0:
        val_loss, bleu_score = validate(model, val_loader, criterion, config.device, config, 
                                       tokenizer=tokenizer, compute_bleu=True)
        
        val_losses.append(val_loss)  
        val_bleu_scores.append(bleu_score)  
        
        if bleu_score is not None:
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        else:
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(epoch, model, optimizer, scheduler, scaler, val_loss, 
                          train_losses, val_losses, val_bleu_scores, config)
    else:
        val_losses.append(None)  
        val_bleu_scores.append(None) 
        print(f"Train Loss: {train_loss:.4f}")

print()
print("Training completed!")
print()

# Plot training curves
print("Step 6: Plotting Training Curves")
print("-" * 80)
plot_training_curves(train_losses, val_losses, val_bleu_scores, save_path='training_curves.png')
print()

# Load best model
checkpoint = torch.load(config.model_file, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluation
print("Step 7: Final Evaluation")
print("-" * 80)

beam_search = BeamSearch(model, tokenizer, beam_size=config.beam_size,
                        max_len=config.max_summary_len, length_penalty=config.length_penalty)
bleu_scorer = BLEUScore()

references = []
hypotheses = []
examples = []

model.eval()
batch_count = 0
print(f"Generating summaries (max {config.max_generation_batches} batches)...")

for i, batch in enumerate(tqdm(test_loader, desc='Generating summaries')):
    if batch_count >= config.max_generation_batches:
        break

    src = batch['article'].to(config.device)
    tgt = batch['summary'].to(config.device)
    src_mask, _ = create_masks(src, tgt)

    # Generate summaries
    with torch.no_grad():
        generated = beam_search.generate(src, src_mask)

    for j in range(len(generated)):
        ref_ids = tgt[j].cpu().tolist()
        ref_ids = [id for id in ref_ids if id not in [tokenizer.special_tokens['<PAD>'],
                                                       tokenizer.special_tokens['<SOS>'],
                                                       tokenizer.special_tokens['<EOS>']]]
        references.append(ref_ids)

        hyp_ids = [id for id in generated[j] if id not in [tokenizer.special_tokens['<PAD>'],
                                                            tokenizer.special_tokens['<SOS>'],
                                                            tokenizer.special_tokens['<EOS>']]]
        hypotheses.append(hyp_ids)

        if len(examples) < config.top_k_summaries:
            article_ids = src[j].cpu().tolist()
            article_ids = [id for id in article_ids if id not in [tokenizer.special_tokens['<PAD>']]]

            examples.append({
                'article': tokenizer.decode(article_ids),
                'reference': tokenizer.decode(ref_ids),
                'generated': tokenizer.decode(hyp_ids)
            })

    batch_count += 1

# Display top examples
print(f"Step 8: Top {config.top_k_summaries} Generated Summaries")
print("=" * 80)

for i, example in enumerate(examples, 1):
    print(f"\nExample {i}:")
    print("-" * 80)
    print(f"Article (truncated): {example['article'][:200]}...")
    print(f"\nReference Summary: {example['reference']}")
    print(f"\nGenerated Summary: {example['generated']}")
    print()

print("=" * 80)
print("DONE!")
print("=" * 80)