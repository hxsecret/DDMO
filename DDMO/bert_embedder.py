"""
BERT-based syscall embedding for DDMO.

Replaces the TF-IDF + LabelEncoder vectorizer (DualSyscallVectorizer) with the
128-dimensional BERT continued-pretrained on syscall logs
(bert-log-pretrained-128d-with-nsp). Each syscall event is serialized to the
canonical text "<syscall>(<args>)" and encoded with the BERT [CLS] token,
yielding a 128-dim contextual embedding (matching the paper's "pre-trained
BERT-based" embedding).
"""

import torch
from transformers import AutoTokenizer, AutoModel


class BertSyscallVectorizer:
    def __init__(self, model_path="./bert-log-pretrained-128d-with-nsp",
                 device=None, max_length=512):
        self.model_path = model_path
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.embed_dim = self.model.config.hidden_size  # 128

    @staticmethod
    def _event_text(event):
        """Canonical serialization of a parsed syscall event for embedding."""
        return f"{event['syscall']}({event['args']})"

    def encode_texts(self, texts):
        """Encode a list of syscall texts -> (N, embed_dim) [CLS] vectors."""
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        return out.last_hidden_state[:, 0, :].cpu()  # (N, embed_dim)

    def transform_direct(self, event):
        """Single syscall event -> (embed_dim,) torch.FloatTensor."""
        return self.encode_texts([self._event_text(event)])[0]

    def transform_sequence(self, sequence):
        """List of syscall events -> (seq_len, embed_dim) torch.FloatTensor."""
        texts = [self._event_text(e) for e in sequence]
        return self.encode_texts(texts)