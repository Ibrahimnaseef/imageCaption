# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):         #Defines a neural module for attention.
    """
    Additive (Bahdanau) attention over image regions
    """
#feat_dim = size of image feature (2048)
#hidden_dim = LSTM hidden size (512)
#attn_dim = intermediate size (256)
    def __init__(self, feat_dim, hidden_dim, attn_dim):
        super().__init__()                                  #Initializes parent nn.Module
        self.feat_attn = nn.Linear(feat_dim, attn_dim)
        self.hidden_attn = nn.Linear(hidden_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1)

    def forward(self, features, hidden):
        """
        features: (B, N, feat_dim)
        hidden:   (B, hidden_dim)
        """
        feat_proj = self.feat_attn(features)                 # (B, N, A) Project features
        hidden_proj = self.hidden_attn(hidden).unsqueeze(1)  # (B, 1, A) Project hidden state

        energy = torch.tanh(feat_proj + hidden_proj)        #This is additive (Bahdanau) attention
        scores = self.score(energy).squeeze(-1)              # (B, N) Convert to scalar scores

        alpha = F.softmax(scores, dim=1)                     # (B, N) Softmax → attention weights
        context = (features * alpha.unsqueeze(-1)).sum(dim=1) #Weighted sum (context vector)
        """context: what the model “looks at”
        alpha: attention map (useful for visualization)"""
        return context, alpha


class CaptionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        feat_dim=2048,
        embed_dim=256,
        hidden_dim=512,
        attn_dim=256,
        pad_idx=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(  #word index → dense vector
            vocab_size, embed_dim, padding_idx=pad_idx)

        self.feat_proj = nn.Linear(feat_dim, hidden_dim)    #Feature projection

        self.attention = Attention(feat_dim, hidden_dim, attn_dim) 

        self.lstm = nn.LSTMCell(embed_dim + feat_dim, hidden_dim)   #LSTM

        self.fc = nn.Linear(hidden_dim, vocab_size)     #Converts hidden → word probabilities

    def forward(self, features, captions):
        """
        features: (B, 36, 2048)
        captions: (B, T)
        """
        B, N, D = features.shape
        T = captions.size(1)

        embeddings = self.embedding(captions)   # (B, T, E) Embed captions

        h = self.feat_proj(features.mean(dim=1))    #Initialize hidden state
        c = torch.zeros_like(h)

        outputs = []

        for t in range(T):
            context, _ = self.attention(features, h)    #Attention
            lstm_input = torch.cat([embeddings[:, t], context], dim=1)  #Prepare LSTM input

            h, c = self.lstm(lstm_input, (h, c))    #LSTM update
            logits = self.fc(h)                 #Predict next word

            outputs.append(logits)      #Store output

        outputs = torch.stack(outputs, dim=1)   # (B, T, vocab)     Final Output

        return outputs
