# Transformer Text Summarization

A PyTorch implementation of a Transformer model for abstractive text summarization, featuring custom BPE tokenization, beam search decoding, and BLEU score evaluation.

## 📋 Overview

This project implements a sequence-to-sequence Transformer model for generating concise summaries from longer articles. The implementation includes:

- **Custom BPE Tokenizer**: Efficient subword tokenization with configurable vocabulary size
- **Transformer Architecture**: Configurable encoder-decoder layers with multi-head attention
- **Beam Search Decoding**: Generate high-quality summaries with length penalty
- **BLEU Score Evaluation**: Automatic evaluation metrics for summary quality
- **Gradient Accumulation**: Train with larger effective batch sizes

## 🏗️ Architecture

### Model Components

- **Encoder**: Multi-layer transformer encoder with self-attention
- **Decoder**: Multi-layer transformer decoder with masked self-attention and cross-attention
- **Positional Encoding**: Sinusoidal position embeddings
- **Multi-Head Attention**: Scaled dot-product attention with multiple heads
- **Feed-Forward Networks**: Position-wise fully connected layers

### Default Configuration

```
Model Dimension (d_model):     512
Attention Heads:               8
Encoder Layers:                8
Decoder Layers:                8
Feed-Forward Dimension:        2048
Vocabulary Size:               50,000
Max Sequence Length:           512
Max Summary Length:            64
```

## 📁 Project Structure

```
.
├── beamSearch.py           # Beam search implementation for generation
├── bleu.py                 # BLEU score calculation
├── bpeTokenizer.py         # Byte Pair Encoding tokenizer
├── config.py               # Hyperparameters and configuration
├── setup.py                # Main training and evaluation script
├── summarizer.py           # Dataset class for summarization
├── transformer.py          # Main Transformer model
├── transformerInternals.py # Attention, encoder, decoder components
└── data/
    ├── train.csv           # Training data
    ├── validation.csv      # Validation data
    └── test.csv            # Test data
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch pandas numpy tqdm matplotlib
```
### Dataset

This project uses the **CNN/DailyMail Dataset** for text summarization:

📦 **Dataset Source**: [Newspaper Text Summarization - CNN/DailyMail](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)

Download the dataset from Kaggle and place the CSV files in the `data/` directory:
- `train.csv`
- `validation.csv`
- `test.csv`

### Data Format

Your CSV files should contain two columns:
- `article`: The input text to summarize
- `highlights`: The reference summary

Example:
```csv
article,highlights
"Long article text here...","Short summary here"
```

### Training

1. **Configure hyperparameters** in `config.py`:
   ```python
   # Adjust these based on your needs
   batch_size = 24
   learning_rate = 0.0005
   n_epochs = 50
   max_train_samples = 287000  # Set to None for full dataset
   ```

2. **Run training**:
   ```bash
   python setup.py
   ```

The script will automatically:
- Train/load a BPE tokenizer
- Load and preprocess datasets
- Train the model with checkpointing
- Generate training curves
- Evaluate on test set

### Resuming Training

The script automatically saves checkpoints. If training is interrupted, simply run `setup.py` again to resume from the last checkpoint.

## ⚙️ Configuration Options

### Model Architecture

```python
d_model = 512              # Model dimension
n_heads = 8                # Number of attention heads
n_encoder_layers = 8       # Encoder depth
n_decoder_layers = 8       # Decoder depth
d_ff = 2048               # Feed-forward dimension
dropout = 0.1             # Dropout rate
```

### Training Hyperparameters

```python
batch_size = 24                    # Batch size
learning_rate = 0.0005            # Initial learning rate
n_epochs = 5                      # Training epochs
gradient_clip = 0.5               # Gradient clipping threshold
label_smoothing = 0.1             # Label smoothing factor
gradient_accumulation_steps = 2   # Accumulation steps
```

### Generation Parameters

```python
beam_size = 3              # Beam width for beam search
length_penalty = 0.6       # Length penalty coefficient
max_summary_len = 64       # Maximum summary length
```

### Performance Optimization

```python
compile_model = False              # Use torch.compile() (PyTorch 2.0+)
use_flash_attention = False        # Flash Attention (requires installation)
num_workers = 0                    # DataLoader workers (Increase for Cuda)
pin_memory = True                  # Pin memory for CUDA
```

## 📈 Monitoring Training

The script provides real-time monitoring:
- Training loss per batch
- Validation loss
- Training time per epoch
- Automatic checkpoint saving on improvement

Training curves are automatically saved to `training_curves.png`.

## 🔧 Troubleshooting

### Out of Memory
- Reduce `batch_size` in `config.py`
- Reduce `max_seq_len` or `max_summary_len`
- Increase `gradient_accumulation_steps`
- Reduce model size (layers, d_model, d_ff)

### Slow Training
- Enable `compile_model` (PyTorch 2.0+)
- Increase `num_workers` for DataLoader
- Use smaller sample sizes during development
- Consider using a smaller model

### Corrupted Checkpoint
The script automatically handles corrupted checkpoints by:
1. Creating a backup of the corrupted file
2. Starting training from scratch
3. Logging the error

## 📝 Citation

If you use this implementation, please cite the original Transformer paper:

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Note**: This implementation is designed for educational purposes and research. For production use, consider using established libraries like Hugging Face Transformers.
