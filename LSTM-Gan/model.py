import torch
import torch.nn as nn
import torch.nn.functional as F
import lyrics
import numpy as np
from model_embeddings import Syll_Embeddings,Word_Embeddings,Doc_Embeddings
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence



 # Les données sont de la forme : List[a] where a = [start of the note, The duration of the note,frequency of the note,str(word),str(syll)]
    
class SingGenerator(nn.Module):
    def __init__(self,embed_size,num_embeddings,hidden_size,num_layers):
        super(SingGenerator, self).__init__()
        self.lstm=nn.LSTM(hidden_size,hidden_size,bias = True,bidirectional = True,num_layers=num_layers,dropout = 0.25,batch_first=True)
        self.proj = nn.Linear(hidden_size*2,3)
        self.proj1 = nn.Linear(embed_size*num_embeddings*2,hidden_size)
    def forward(self,X):
        X = self.proj1(X)
        X = F.relu(X)
        X,_= self.lstm(X)
        X = self.proj(X)
        
        return X
                        
class SingDiscriminator(nn.Module):
    def __init__(self,embed_size,num_embeddings,hidden_size,num_layers):
        super(SingDiscriminator, self).__init__()
        self.lstm=nn.LSTM(3+embed_size*num_embeddings,hidden_size,bias = True,bidirectional = True,num_layers=num_layers,dropout = 0.25,batch_first=True)
        self.proj = nn.Linear(hidden_size*2,1)
    def forward(self,X):
        X,_= self.lstm(X)
        X = self.proj(X)
        X = torch.sigmoid(X)
        return X
                        
                          
                          
class SingGan(nn.Module):
    def __init__(self,hidden_size,syll_emb,word_emb,embed_size=10,num_layers=2,noise_size = 20,num_embeddings=2):
        super(SingGan, self).__init__()
        
        self.syll_embedding = syll_emb
        self.word_embedding = word_emb
        #self.doc_embedding = Doc_Embeddings(embed_size=embed_size, vocab=vocab)
        
      
        
        self.generator = SingGenerator(embed_size,num_embeddings,hidden_size,num_layers)
       
        
        self.discriminator = SingDiscriminator(embed_size,num_embeddings,hidden_size,num_layers)
      
    
    def generate(self,syll_input_padded,word_input_padded,num_embeddings=2,noise_size=20):
        
        
        
        if torch.cuda.is_available():
            device = 'cuda'
        else :
            device = 'cpu'
        
        
        enc_hiddens, dec_init_state = None, None
        syll_embeddings = syll_input_padded 
        word_embeddings = word_input_padded 
        
        embeddings = torch.cat((syll_embeddings,word_embeddings),dim=2)
        X = torch.cat((torch.FloatTensor(embeddings.size()).uniform_().to(device),embeddings),dim=2)
        
        
        
        midi_generated = self.generator(X)
 
        
        return midi_generated
    
    
    def discriminate(self, midi_vectors,syll,word):
        
        midi_lengths = [len(s) for s in midi_vectors]
        
        enc_hiddens, dec_init_state = None, None
        syll_embeddings = syll
        word_embeddings = word 
        
        embeddings = torch.cat((syll_embeddings,word_embeddings),dim=2)
        
        X = torch.cat((midi_vectors,embeddings),dim=2)
        
        proba_true = self.discriminator(X)
       
        
        
        return proba_true
    
       
        
