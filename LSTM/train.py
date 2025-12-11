import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import pickle
import os
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import ConfigLSTM
from lstm_internals import LSTMEncoder, LSTMDecoder
from lstm import Seq2Seq
from beam_search_lstm import BeamSearchLSTM
from transformer.bpeTokenizer import BPETokenizer
from transformer.bleu import BLEUScore
from transformer.summarizer import SummarizationDataset
import matplotlib.pyplot as plt
from datetime import datetime

config = ConfigLSTM()

# ============================================================================
# COLLATE FUNCTION
# ============================================================================
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
        'article_lens': torch.tensor(article_lens, dtype=torch.long),
        'summary_lens': torch.tensor(summary_lens, dtype=torch.long)
    }


# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device, config, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(dataloader, desc='Training', dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        src = batch['article'].to(device, non_blocking=True)
        tgt = batch['summary'].to(device, non_blocking=True)
        src_lens = batch['article_lens'].to(device, non_blocking=True)

        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(src, src_lens, tgt, teacher_forcing_ratio=0.5)
                output = output[:, 1:].reshape(-1, output.size(-1))
                tgt = tgt[:, 1:].reshape(-1)
                loss = criterion(output, tgt) / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            output = model(src, src_lens, tgt, teacher_forcing_ratio=0.5)
            output = output[:, 1:].reshape(-1, output.size(-1))
            tgt = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt) / config.gradient_accumulation_steps

            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * config.gradient_accumulation_steps

        if batch_idx % config.log_interval == 0:
            pbar.set_postfix({
                'loss': f'{loss.item() * config.gradient_accumulation_steps:.4f}',
                'avg': f'{total_loss/(batch_idx+1):.4f}'
            }, refresh=False)

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, config):
    """Validate model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating', dynamic_ncols=True)
        for batch in pbar:
            src = batch['article'].to(device, non_blocking=True)
            tgt = batch['summary'].to(device, non_blocking=True)
            src_lens = batch['article_lens'].to(device, non_blocking=True)

            if config.device == 'cuda':
                with torch.cuda.amp.autocast():
                    output = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)
                    output = output[:, 1:].reshape(-1, output.size(-1))
                    tgt = tgt[:, 1:].reshape(-1)
                    loss = criterion(output, tgt)
            else:
                output = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)
                output = output[:, 1:].reshape(-1, output.size(-1))
                tgt = tgt[:, 1:].reshape(-1)
                loss = criterion(output, tgt)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_training_curves(train_losses, val_losses, save_path='training_curves_lstm.png'):
    """
    Plot training and validation curves
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch (with None for non-validation epochs)
        save_path: Path to save the plot
    """
    # Filter out None values for validation metrics
    val_epochs = [i for i, loss in enumerate(val_losses) if loss is not None]
    val_losses_filtered = [loss for loss in val_losses if loss is not None]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Training Loss
    axes[0].plot(range(len(train_losses)), train_losses, 'b-', linewidth=2, label='Training Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Add best training loss annotation
    if train_losses:
        best_train_loss = min(train_losses)
        best_train_epoch = train_losses.index(best_train_loss)
        axes[0].axhline(y=best_train_loss, color='r', linestyle='--', alpha=0.5, linewidth=1)
        axes[0].text(0.02, 0.98, f'Best Loss: {best_train_loss:.4f} (Epoch {best_train_epoch+1})', 
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Training vs Validation Loss
    axes[1].plot(range(len(train_losses)), train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.7)
    if val_losses_filtered:
        axes[1].plot(val_epochs, val_losses_filtered, 'r-', linewidth=2, marker='o', 
                     markersize=6, label='Validation Loss')
        
        # Add best validation loss annotation
        best_val_loss = min(val_losses_filtered)
        best_val_epoch = val_epochs[val_losses_filtered.index(best_val_loss)]
        axes[1].axhline(y=best_val_loss, color='g', linestyle='--', alpha=0.5, linewidth=1)
        axes[1].text(0.02, 0.98, f'Best Val Loss: {best_val_loss:.4f} (Epoch {best_val_epoch+1})', 
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Add overall statistics
    fig.suptitle(f'LSTM Training Progress - {len(train_losses)} Epochs', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.99, 0.01, f'Generated: {timestamp}', ha='right', fontsize=8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to {save_path}")
    
    # Also display if in notebook
    try:
        from IPython.display import display
        display(fig)
        plt.show()
    except ImportError:
        pass


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================
def save_checkpoint(epoch, model, optimizer, scheduler, scaler, val_loss, train_losses, val_losses, config):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'val_loss': val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    torch.save(checkpoint, config.model_file)
    print(f"✓ Best model saved to {config.model_file} (Val Loss: {val_loss:.4f})")


def load_checkpoint(model, optimizer, scheduler, scaler, config):
    """Load checkpoint if exists"""
    if os.path.exists(config.model_file):
        print(f"Found existing checkpoint: {config.model_file}")
        checkpoint = torch.load(config.model_file, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if scaler and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        
        print(f"Resuming from epoch {start_epoch} (Best Val Loss: {best_val_loss:.4f})")
        return start_epoch, best_val_loss, train_losses, val_losses
    else:
        print("No existing checkpoint found. Starting from scratch.")
        return 0, float('inf'), [], []


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print("LSTM SUMMARIZATION TRAINING")
    print("=" * 80)
    print()

    # Print device information
    print("Device Information:")
    print("-" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device count: {torch.cuda.device_count()}")
    print(f"Using device: {config.device}")
    print()

    # FIX: Convert device string to torch.device object
    device = torch.device(config.device)
    print(f"Device object: {device}")
    print()

    # Load tokenizer
    print("Step 1: Loading BPE Tokenizer")
    print("-" * 80)
    tokenizer = BPETokenizer.load(config.tokenizer_file)
    print(f"Tokenizer loaded from {config.tokenizer_file}")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print()

    # Load data
    print("Step 2: Loading Data")
    print("-" * 80)

    train_data = pd.read_csv(config.train_file, nrows=config.max_train_samples)
    val_data = pd.read_csv(config.val_file, nrows=config.max_val_samples)
    test_data = pd.read_csv(config.test_file, nrows=config.max_test_samples)

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    print()

    train_dataset = SummarizationDataset(
        train_data,
        tokenizer,
        max_seq_len=config.max_seq_len,
        max_summary_len=config.max_summary_len
    )

    val_dataset = SummarizationDataset(
        val_data,
        tokenizer,
        max_seq_len=config.max_seq_len,
        max_summary_len=config.max_summary_len
    )

    test_dataset = SummarizationDataset(
        test_data,
        tokenizer,
        max_seq_len=config.max_seq_len,
        max_summary_len=config.max_summary_len
    )

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
    print("Step 3: Creating LSTM Model")
    print("-" * 80)

    encoder = LSTMEncoder(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional
    )

    decoder = LSTMDecoder(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
        bidirectional_encoder=config.bidirectional
    )

    # FIX: Pass device object instead of string
    model = Seq2Seq(encoder, decoder, device).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Verify model is on correct device
    print(f"Model is on device: {next(model.parameters()).device}")
    print()

    # Training setup
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.special_tokens['<PAD>'],
        label_smoothing=config.label_smoothing
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )

    # Mixed precision training scaler
    if config.device == 'cuda':
        try:
            scaler = torch.amp.GradScaler('cuda')
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    print("Step 4: Checking for Existing Checkpoint")
    print("-" * 80)
    start_epoch, best_val_loss, train_losses, val_losses = load_checkpoint(
        model, optimizer, scheduler, scaler, config
    )
    print()

    # Training loop
    print("Step 5: Training LSTM Model")
    print("-" * 80)
    print(f"Total epochs: {config.n_epochs}")
    print(f"Starting from epoch: {start_epoch + 1}")
    print()

    for epoch in range(start_epoch, config.n_epochs):
        print(f"\nEpoch {epoch + 1}/{config.n_epochs}")

        # FIX: Pass device object instead of accessing config.device
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config, scaler)
        train_losses.append(train_loss)

        if (epoch + 1) % config.validate_every_n_epochs == 0:
            val_loss = validate(model, val_loader, criterion, device, config)
            val_losses.append(val_loss)
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if scheduler:
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(epoch, model, optimizer, scheduler, scaler, val_loss, 
                              train_losses, val_losses, config)
        else:
            val_losses.append(None)
            print(f"Train Loss: {train_loss:.4f}")

    print()
    print("Training completed!")
    print()

    # Plot training curves
    print("Step 6: Plotting Training Curves")
    print("-" * 80)
    plot_training_curves(train_losses, val_losses, save_path='training_curves_lstm.png')
    print()

    # Save vocabulary
    print("Step 7: Saving Vocabulary")
    print("-" * 80)
    with open(config.vocab_file, 'wb') as f:
        pickle.dump(tokenizer.vocab, f)
    print(f"Vocabulary saved to {config.vocab_file}")
    print()

    # Load best model
    checkpoint = torch.load(config.model_file, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluation with BLEU
    print("Step 8: Final Evaluation")
    print("-" * 80)


    beam_search = BeamSearchLSTM(
        model, tokenizer,
        beam_size=config.beam_size,
        max_len=config.max_summary_len,
        length_penalty=config.length_penalty
    )
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

        # FIX: Use device object
        src = batch['article'].to(device)
        tgt = batch['summary'].to(device)
        src_lens = batch['article_lens'].to(device)

        # Generate summaries
        with torch.no_grad():
            generated = beam_search.generate(src, src_lens)

        for j in range(len(generated)):
            # Reference
            ref_ids = tgt[j].cpu().tolist()
            ref_ids = [id for id in ref_ids if id not in [tokenizer.special_tokens['<PAD>'],
                                                           tokenizer.special_tokens['<SOS>'],
                                                           tokenizer.special_tokens['<EOS>']]]
            references.append(ref_ids)

            # Hypothesis
            hyp_ids = [id for id in generated[j] if id not in [tokenizer.special_tokens['<PAD>'],
                                                                tokenizer.special_tokens['<SOS>'],
                                                                tokenizer.special_tokens['<EOS>']]]
            hypotheses.append(hyp_ids)

            # Save examples
            if len(examples) < config.top_k_summaries:
                article_ids = src[j].cpu().tolist()
                article_ids = [id for id in article_ids if id not in [tokenizer.special_tokens['<PAD>']]]

                examples.append({
                    'article': tokenizer.decode(article_ids),
                    'reference': tokenizer.decode(ref_ids),
                    'generated': tokenizer.decode(hyp_ids)
                })

        batch_count += 1

    # Compute BLEU
    bleu_results = bleu_scorer.compute(references, hypotheses)

    print(f"\nBLEU Score: {bleu_results['bleu']:.4f}")
    print(f"Brevity Penalty: {bleu_results['brevity_penalty']:.4f}")
    print(f"Length Ratio: {bleu_results['length_ratio']:.4f}")
    for i, prec in enumerate(bleu_results['precisions'], 1):
        print(f"Precision-{i}: {prec:.4f}")
    print()

    # Display top examples
    print(f"Step 9: Top {config.top_k_summaries} Generated Summaries")
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


if __name__ == "__main__":
    main()