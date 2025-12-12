# Text Summarization with Neural Networks

PyTorch implementations of Transformer and LSTM models for abstractive text summarization, featuring custom BPE tokenization, beam search decoding, and BLEU score evaluation.

## Overview

This project implements sequence-to-sequence models for generating concise summaries from longer articles. The repository includes two architectures:

- **Transformer**: Transformer with attention-based architecture
- **LSTM**: Recurrent neural network with attention mechanism

Both implementations include:

- **Custom BPE Tokenizer**: Efficient subword tokenization with configurable vocabulary size
- **Beam Search Decoding**: Generate high-quality summaries with length penalty
- **BLEU Score Evaluation**: Automatic evaluation metrics for summary quality
- **Gradient Accumulation**: Train with larger effective batch sizes

## Project Structure

```
project_root/
├── LSTM/
│   ├── beam_search_lstm.py       # Beam search for LSTM
│   ├── config.py                 # LSTM configuration
│   ├── lstm_internals.py         # Encoder, Decoder, Attention
│   ├── lstm.py                   # Seq2Seq model
│   └── train.py                  # Training script
│
├── transformer/
│   ├── beamSearch.py             # Beam search for Transformer
│   ├── bleu.py                   # BLEU score calculation
│   ├── bpeTokenizer.py           # BPE tokenizer
│   ├── config.py                 # Transformer configuration
│   ├── FinalTransformer.ipynb    # Training notebook
│   ├── setup.py                  # Setup and training script
│   ├── summarizer.py             # Summarization utilities
│   ├── transformer.py            # Transformer model
│   └── transformerInternals.py   # Transformer components
│
├── README.md
└── .gitignore
```

## Getting Started

### Prerequisites

```bash
pip install torch pandas numpy tqdm matplotlib
```

### Dataset

This project uses the **CNN/DailyMail Dataset** for text summarization:

**Dataset Source**: [Newspaper Text Summarization - CNN/DailyMail](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)

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

---

## Transformer Model

### Architecture

The Transformer implementation uses the encoder-decoder architecture from "Attention is All You Need":

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

### Training

1. **Configure hyperparameters** in `transformer/config.py`:
   ```python
   # Adjust these based on your needs
   batch_size = 24
   learning_rate = 0.0005
   n_epochs = 50
   max_train_samples = 287000  # Set to None for full dataset
   ```

2. **Run training**:
   ```bash
   cd transformer
   python setup.py
   ```

The script will automatically:
- Train/load a BPE tokenizer
- Load and preprocess datasets
- Train the model with checkpointing
- Generate training curves
- Evaluate on test set

### Configuration Options

#### Model Architecture

```python
d_model = 512              # Model dimension
n_heads = 8                # Number of attention heads
n_encoder_layers = 8       # Encoder depth
n_decoder_layers = 8       # Decoder depth
d_ff = 2048               # Feed-forward dimension
dropout = 0.1             # Dropout rate
```

#### Training Hyperparameters

```python
batch_size = 24                    # Batch size
learning_rate = 0.0005            # Initial learning rate
n_epochs = 5                      # Training epochs
gradient_clip = 0.5               # Gradient clipping threshold
label_smoothing = 0.1             # Label smoothing factor
gradient_accumulation_steps = 2   # Accumulation steps
```

#### Generation Parameters

```python
beam_size = 3              # Beam width for beam search
length_penalty = 0.6       # Length penalty coefficient
max_summary_len = 64       # Maximum summary length
```

#### Performance Optimization

```python
compile_model = False              # Use torch.compile() (PyTorch 2.0+)
use_flash_attention = False        # Flash Attention (requires installation)
num_workers = 0                    # DataLoader workers (Increase for CUDA)
pin_memory = True                  # Pin memory for CUDA
```

---

## LSTM Model

### Architecture

The LSTM implementation uses a sequence-to-sequence architecture with attention:

- **Bidirectional LSTM Encoder**: Captures context from both directions
- **LSTM Decoder with Attention**: Generates summaries with attention over encoder outputs
- **Bahdanau Attention**: Additive attention mechanism for alignment
- **Teacher Forcing**: Configurable ratio for training stability

### Default Configuration

```
Embedding Dimension:        256
Hidden Dimension:           256
Encoder Layers:             2
Decoder Layers:             2
Bidirectional Encoder:      True
Vocabulary Size:            50,000
Max Sequence Length:        256
Max Summary Length:         64
```

### Training

1. **Configure hyperparameters** in `LSTM/config.py`:
   ```python
   # Adjust these based on your needs
   batch_size = 64
   learning_rate = 0.001
   n_epochs = 3
   max_train_samples = 75000  # Set to None for full dataset
   ```

2. **Run training**:
   ```bash
   cd LSTM
   python train.py
   ```

The script will automatically:
- Load the pre-trained BPE tokenizer
- Load and preprocess datasets
- Train the model with checkpointing
- Generate training curves
- Evaluate on test set

### Configuration Options

#### Model Architecture

```python
embedding_dim = 256        # Embedding dimension
hidden_dim = 256          # Hidden state dimension
n_layers = 2              # Number of LSTM layers
dropout = 0.3             # Dropout rate
bidirectional = True      # Bidirectional encoder
```

#### Training Hyperparameters

```python
batch_size = 64                    # Batch size
learning_rate = 0.001             # Initial learning rate
n_epochs = 3                      # Training epochs
gradient_clip = 1.0               # Gradient clipping threshold
label_smoothing = 0.1             # Label smoothing factor
gradient_accumulation_steps = 1   # Accumulation steps
```

#### Generation Parameters

```python
beam_size = 5              # Beam width for beam search
length_penalty = 0.6       # Length penalty coefficient
max_summary_len = 64       # Maximum summary length
```

#### Performance Optimization

```python
num_workers = 7                    # DataLoader workers (Increase for CUDA)
pin_memory = True                  # Pin memory for CUDA
log_interval = 50                  # Progress logging frequency
```

---

## Resuming Training

Both implementations automatically save checkpoints. If training is interrupted, simply run the training script again to resume from the last checkpoint.

## Monitoring Training

Both scripts provide real-time monitoring:
- Training loss per batch
- Validation loss
- Training time per epoch
- Automatic checkpoint saving on improvement

Training curves are automatically saved to `training_curves.png`.

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in `config.py`
- Reduce `max_seq_len` or `max_summary_len`
- Increase `gradient_accumulation_steps`
- Reduce model size (layers, hidden_dim, d_model, d_ff)

### Slow Training
- Enable `compile_model` for Transformer (PyTorch 2.0+)
- Increase `num_workers` for DataLoader
- Use smaller sample sizes during development
- Consider using a smaller model

### Corrupted Checkpoint
Both scripts automatically handle corrupted checkpoints by:
1. Creating a backup of the corrupted file
2. Starting training from scratch
3. Logging the error

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on the repository.

---

**Note**: These implementations are designed for educational purposes and research. For production use, consider using established libraries like Hugging Face Transformers.