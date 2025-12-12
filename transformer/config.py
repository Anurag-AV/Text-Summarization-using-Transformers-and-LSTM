
import torch

# HYPERPARAMETERS
class Config:
    # Data parameters
    train_file = '../data/train.csv'
    val_file = '../data/validation.csv'
    test_file = '../data/test.csv'

    # BPE Tokenizer parameters
    vocab_size = 50000
    tokenizer_file = 'bpe_tokenizer_all.pkl'

    # Model parameters
    d_model = 512
    n_heads = 8
    n_encoder_layers = 8  
    n_decoder_layers = 8 
    d_ff = 2048
    dropout = 0.1
    max_seq_len = 512
    max_summary_len = 64

    # Training parameters
    batch_size = 24  
    learning_rate = 0.0005  
    n_epochs = 50
    warmup_steps = 4000  
    gradient_clip = 0.5  
    label_smoothing = 0.1
    gradient_accumulation_steps = 2  

    # Data sampling 
    max_train_samples = 287000
    max_val_samples = 5000
    max_test_samples = 5000

    # Generation parameters
    beam_size = 5
    length_penalty = 0.6
    top_k_summaries = 5
    max_generation_batches = 100

    # Model saving
    model_file = 'transformer_summarization_deep.pt'
    vocab_file = 'vocabulary.pkl'

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Performance optimizations
    num_workers = 0
    pin_memory = True if torch.cuda.is_available() else False
    compile_model = False
    use_flash_attention = False

    # Progress settings
    log_interval = 50
    validate_every_n_epochs = 1

