import torch

class ConfigLSTM:
    # Data parameters
    train_file = 'C:/Users/hp/Downloads/archive (1)/cnn_dailymail/train.csv'
    val_file = 'C:/Users/hp/Downloads/archive (1)/cnn_dailymail/validation.csv'
    test_file = 'C:/Users/hp/Downloads/archive (1)/cnn_dailymail/test.csv'

    # BPE Tokenizer parameters
    vocab_size = 50000
    tokenizer_file = '../transformer/bpe_tokenizer_all.pkl'

    # LSTM Model parameters
    embedding_dim = 256
    hidden_dim = 256
    n_layers = 2
    dropout = 0.3
    bidirectional = True
    max_seq_len = 256
    max_summary_len = 64

    # Training parameters
    batch_size = 64
    learning_rate = 0.001
    n_epochs = 15
    gradient_clip = 1.0
    label_smoothing = 0.1
    gradient_accumulation_steps = 1

    # Data sampling for faster training
    max_train_samples = 50000
    max_val_samples = 5000
    max_test_samples = 5000

    # Generation parameters
    beam_size = 5
    length_penalty = 0.6
    top_k_summaries = 5
    max_generation_batches = 100

    # Model saving
    model_file = 'lstm_summarization.pt'
    vocab_file = 'vocabulary_lstm.pkl'

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Performance optimizations
    num_workers = 7
    pin_memory = True if torch.cuda.is_available() else False

    # Progress settings
    log_interval = 50
    validate_every_n_epochs = 1