"""
Contains PyTorch model code to instantiate a SkipGramModelWithNegSampling model.
"""
import torch
import torch.autograd as autograd
import torch.nn as nn

class SkipGramModelWithNegSampling(nn.Module):
    """
    """
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModelWithNegSampling, self).__init__()
        self.embeddings_in = nn.Embedding(vocab_size, embedding_dim) # center
        self.embeddings_out = nn.Embedding(vocab_size, embedding_dim) # context
        
        # никакая логсигмоида нам не нужна! это все заложено в лоссе
        torch.nn.init.xavier_uniform_(self.embeddings_in.weight)
        torch.nn.init.xavier_uniform_(self.embeddings_out.weight)
        
    def forward(self, center_words, pos_context_words, neg_context_words):
        # center_words — входные слова
        # pos_context_words — таргет, т.е. правильный контекст (реально существующий для входного слова)
        # neg_context_words — отрицательные примеры — то что не должно быть в контексте

        v_in = self.embeddings_in(center_words) 
        v_out = self.embeddings_out(pos_context_words)
        v_neg = self.embeddings_out(neg_context_words)
        
        pos_scores = (torch.sum(v_in * v_out, dim=1))
        neg_scores = (torch.bmm(v_neg, v_in.unsqueeze(2)).squeeze(2)) #.sum(1) # bmm - батчевое (по 2D-матричное) перемножение матриц
        return pos_scores, neg_scores
