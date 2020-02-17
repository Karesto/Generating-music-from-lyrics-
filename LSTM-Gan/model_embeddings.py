import lyrics
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

class CBOW(nn.Module):

    def __init__(self, context_size=2, embedding_size=100, vocab_size=None):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(embedding_size, vocab_size)

    def forward(self, inputs):
        lookup_embeds = self.embeddings(inputs)
        embeds = lookup_embeds.sum(dim=0)
        out = self.linear1(embeds)
       # out = F.log_softmax(out)
        return out


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


class Syll_Embeddings(nn.Module):
    def __init__(self,vocab,embed_size= 10):
        super(Syll_Embeddings, self).__init__()
        pad_token_idx = vocab['<pad>']
        self.model = nn.Embedding(len(vocab),embed_size,padding_idx = pad_token_idx) 
class Word_Embeddings(nn.Module):
    def __init__(self,vocab,embed_size=10):
        super(Word_Embeddings, self).__init__()
        pad_token_idx = vocab['<pad>']
        self.model = nn.Embedding(len(vocab),embed_size,padding_idx = pad_token_idx) 
class Doc_Embeddings(nn.Module):
    def __init__(self,vocab,embed_size=10):
        super(Doc_Embeddings, self).__init__()
        
        pad_token_idx = vocab['<pad>']
        self.model = nn.Embedding(len(vocab),embed_size,padding_idx = pad_token_idx) 
        
        
        
        
        
