import pickle
from collections import defaultdict, Counter
from tqdm import tqdm
import re
# BPE TOKENIZER 
class BPETokenizer:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.word_freqs = {}
        self.splits = {}
        self.merges = {}
        self.vocab = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3
        }

    def train(self, texts, progress=True, max_samples=1000000):
        print("Training BPE tokenizer (optimized)...")

        if len(texts) > max_samples:
            print(f"Sampling {max_samples} texts from {len(texts)} for faster training...")
            import random
            texts = random.sample(texts, max_samples)

        print("Computing word frequencies...")
        word_counter = Counter()
        for text in tqdm(texts, disable=not progress):
            words = self._pre_tokenize(text)
            word_counter.update(words)

        max_words = 50000
        if len(word_counter) > max_words:
            print(f"Keeping top {max_words} most frequent words...")
            self.word_freqs = dict(word_counter.most_common(max_words))
        else:
            self.word_freqs = dict(word_counter)

        print("Initializing character-level splits...")
        alphabet = set()
        for word in self.word_freqs.keys():
            alphabet.update(word)

        self.vocab = self.special_tokens.copy()
        for char in sorted(alphabet):
            self.vocab[char] = len(self.vocab)
        self.splits = {word: tuple(word) for word in self.word_freqs.keys()}

        num_merges = self.vocab_size - len(self.vocab)
        print(f"Learning {num_merges} merges...")

        # Cache pair positions for faster updates
        self.pair_cache = {}
        self._build_pair_cache()

        for i in tqdm(range(num_merges), disable=not progress):
            pair_freqs = self._compute_pair_freqs_cached()
            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)
            merged = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged
            self.vocab[merged] = len(self.vocab)
            self._update_splits_cached(best_pair, merged)

        print(f"Vocabulary size: {len(self.vocab)}")

        # Clear cache to save memory
        self.pair_cache = None

    def _pre_tokenize(self, text):
        """Pre-tokenize text into words"""
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s\.,!?;:\'\"-]', '', text)
        return text.split()

    def _build_pair_cache(self):
        """Build cache of which words contain which pairs"""
        self.pair_cache = defaultdict(set)
        for word in self.word_freqs.keys():
            split = self.splits[word]
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                self.pair_cache[pair].add(word)

    def _compute_pair_freqs_cached(self):
        """Compute frequency of character pairs using cache"""
        pair_freqs = defaultdict(int)
        for pair, words in self.pair_cache.items():
            for word in words:
                split = self.splits[word]
                count = sum(1 for i in range(len(split) - 1)
                           if split[i] == pair[0] and split[i + 1] == pair[1])
                pair_freqs[pair] += count * self.word_freqs[word]
        return pair_freqs

    def _update_splits_cached(self, pair, merged):
        """Update splits only for words containing the pair"""
        words_to_update = self.pair_cache[pair].copy()
        del self.pair_cache[pair]

        for word in words_to_update:
            old_split = self.splits[word]
            new_split = []
            i = 0

            while i < len(old_split):
                if i < len(old_split) - 1 and old_split[i] == pair[0] and old_split[i + 1] == pair[1]:
                    new_split.append(merged)
                    i += 2
                else:
                    new_split.append(old_split[i])
                    i += 1

            self.splits[word] = tuple(new_split)
            for i in range(len(new_split) - 1):
                new_pair = (new_split[i], new_split[i + 1])
                self.pair_cache[new_pair].add(word)

    def _merge_pair(self, split, pair):
        """Merge a pair in a split"""
        new_split = []
        i = 0
        while i < len(split):
            if i < len(split) - 1 and split[i] == pair[0] and split[i + 1] == pair[1]:
                new_split.append(self.merges[pair])
                i += 2
            else:
                new_split.append(split[i])
                i += 1
        return new_split

    def tokenize(self, text):
        """Tokenize text using learned BPE (optimized)"""
        words = self._pre_tokenize(text)
        tokens = []

        merge_priority = {pair: idx for idx, pair in enumerate(self.merges.keys())}

        for word in words:
            split = list(word)
            while len(split) > 1:
                pairs = [(i, (split[i], split[i + 1]))
                        for i in range(len(split) - 1)]

                valid_merges = [(i, pair) for i, pair in pairs if pair in merge_priority]

                if not valid_merges:
                    break
                min_pos, min_pair = min(valid_merges, key=lambda x: merge_priority[x[1]])
                split = split[:min_pos] + [self.merges[min_pair]] + split[min_pos + 2:]

            tokens.extend(split)

        return tokens

    def encode(self, text, add_special_tokens=True):
        """Encode text to token IDs"""
        tokens = self.tokenize(text)
        ids = []

        if add_special_tokens:
            ids.append(self.special_tokens['<SOS>'])

        for token in tokens:
            ids.append(self.vocab.get(token, self.special_tokens['<UNK>']))

        if add_special_tokens:
            ids.append(self.special_tokens['<EOS>'])

        return ids

    def decode(self, ids, skip_special_tokens=True):
        """Decode token IDs to text"""
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = []

        for id in ids:
            token = id_to_token.get(id, '<UNK>')
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return ' '.join(tokens)

    def save(self, filepath):
        """Save tokenizer"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'word_freqs': self.word_freqs,
                'splits': self.splits,
                'merges': self.merges,
                'vocab': self.vocab,
                'special_tokens': self.special_tokens
            }, f)

    @classmethod
    def load(cls, filepath):
        """Load tokenizer"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        tokenizer = cls(data.get('vocab_size', 10000))
        tokenizer.word_freqs = data.get('word_freqs', {})
        tokenizer.splits = data.get('splits', {})
        tokenizer.merges = data.get('merges', {})
        tokenizer.vocab = data.get('vocab', {})
        tokenizer.special_tokens = data.get('special_tokens', {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3
        })

        return tokenizer
