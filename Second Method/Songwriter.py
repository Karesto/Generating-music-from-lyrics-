# Libraries Included:
# Numpy, Scipy, Scikit, Pandas

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv

#This is songwriter based on 1809.04318.pdf
#The input is a series of sylables

######################################################
'''
Todo :
    *The rest lul



Tocheck:
    *Is the bidirectionnal RNN working fine ?


'''






class x_encoder(nn.Module):
    '''
    Lyric encoder
    '''
    def __init__(self,ic, hs, nl):
        super(x_encoder, self).__init__()
        self.ic = ic
        self.hs = hs
        self.nl = nl
        self.encoder = torch.nn.GRU(ic, hs, nl, batch_first=True, bidirectional=True)
    def forward(self,x,h):
        '''
        x.shape(batch_size,time_step,input_size)
        time_step : Length of sentence
        input_size: Syllable length
        '''
        r_out, (h_n, h_c) = self.encoder(x, h)
        return r_out[:, -1, :]

class m_encoder(nn.Module):
    '''
    Melody Encoder
    '''
    def __init__(self,ic, hs, nl):
        super(m_encoder, self).__init__()
        self.ic = ic
        self.hs = hs
        self.nl = nl
        self.encoder = torch.nn.GRU(ic, hs, nl, batch_first=True, bidirectional=True)

    def forward(self,x,h):
        '''
        x.shape(batch_size,time_step,input_size)
        time_step : Length of sentence
        input_size: Syllable length
        '''
        r_out, (h_n, h_c) = self.encoder(x, h)
        return r_out[:, -1, :]


class m_decoder(nn.Module):
    '''
    Melody decoder
    '''
    def __init__(self, ic, hs, nl):
        super(m_decoder, self).__init__()
        self.ic = ic
        self.hs = hs
        self.nl = nl
        self.decoder = torch.nn.GRU(ic, hs, nl, batch_first=True, bidirectional=True)
    def forward(self, x,h):
        '''
        x.shape(batch_size,time_step,input_size)
        time_step : Length of sentence
        input_size: Syllable length
        '''
        r_out, (h_n, h_c) = self.decoder(x, h)
        return r_out[:, -1, :]

class MC2(nn.Module):
    ''' E = MC2 '''
    def __init__(self, ic, hs):
        super(MC2, self).__init__()
        self.v = nn.Linear(ic[0],hs[0], bias = False)
        self.w = nn.Linear(ic[1],hs[1], bias = False)
        self.u = nn.Linear(ic[2],hs[2], bias = False)

    def forward(self, s, h):
        return self.v(torch.tanh(self.w(s) + self.u(h)))


class g(nn.Module):
   '''
    Since we have no information on this, we make it like this
   '''
   def __init__(self,ic,hs):
        super(g , self).__init__()
        self.lin = nn.Linear(ic, hs, bias = True)

   def forward(self,x):
        return torch.nn.Sigmoid(self.lin(x))





def Songwriter(X,n):
    # X : Liste des Syllabes


    i = 0
    spit = out_shape_de_pit * torch.tensor([0])
    sdur = out_shape_de_dur * torch.tensor([0])
    srest = out_shape_de_rest * torch.tensor([0])
    c = 2*out_shape * torch.tensor([0])
    y = torch.tensor([0, 0, 0])
    m = torch.tensor(tCon * [0, 0, 0])

    durations = [0.25, 0.5, 1, 2, 4, 8, 16, 32]

    coord = torch.tensor([[[(pit, dur, rest) for pit in range(128)] for dur in durations] for rest in [0]+durations])
    p = torch.tensor([[[0 for pit in range(128)] for dur in durations] for rest in [0]+durations])
    s = torch.tensor([[[(out_shape_de_pit * torch.tensor([0]),
                        out_shape_de_dur * torch.tensor([0]),
                        out_shape_de_rest * torch.tensor([0])) for pit in range(128)] for dur in durations] for rest in [0]+durations])
    j = 0

    hlrc = enc_x.forward(X.view(n, -1, out_embed),None)

    for i in range(n):
         hcon = torch.cat([enc_m_pit(m.view(7, -1, 1).float(),None), enc_m_dur(m.view(7, -1, 1).float(),None)])
         e = torch.tensor([torch.exp(mc2.forward(torch.cat([spit, sdur, srest]).view(3, -1).long(), hcon[k].view(2*out_shape_m, -1).long())) for k in range(tCon)])
         se = torch.sum(e)
         c2 = torch.tensor([e[k]*hcon[k]/se + hlrc[i] for k in range(tCon)])
         for pit in range(128):
             for dur in durations:
                 for rest in [0]+durations:
                     _,s[pit][dur][rest][0] = dec_m_pit.forward(torch.cat(c, y[-1][0], hlrc[i]).view(1+4*out_shape, -1, 1), spit.view(1, -1, 1))
                     _,s[pit][dur][rest][1] = dec_m_dur.forward(torch.cat(c, y[-1][1], pit, s[pit][dur][rest][0]).view(2*out_shape+3, -1, 1), sdur.view(1, -1, 1))
                     _,s[pit][dur][rest][2] = dec_m_rest.forward(torch.cat(c, y[-1][2], pit, dur, s[pit][dur][rest][1]).view(2*out_shape+4, -1, 1), srest.view(1, -1, 1))
                     _,p[pit][dur][rest][0] = gp(pit, s[pit][dur][rest][0], c2, y[-1])
                     _,p[pit][dur][rest][0] = gd(dur, s[pit][dur][rest][1], c2, y[-1], pit)
                     _,p[pit][dur][rest][0] = gr(rest, s[pit][dur][rest][2], c2, y[-1], pit, dur)
         y.append(torch.view(coord)[torch.argmax(p)])
         m = torch.cat(m[1:], y[-1])
         spit = s[ypit[-1][0]][ypit[-1][1]][ypit[-1][2]][0]
         sdur = s[ypit[-1][0]][ypit[-1][1]][ypit[-1][2]][1]
         srest = s[ypit[-1][0]][ypit[-1][1]][ypit[-1][2]][2]
         c = c2

    return y

def prm(l):
    # This is for parameters
    # Takes a list of NNs and computes list of parameters
    if l == []:
        return(l)
    else:
        return list(l[0].parameters()) + prm(l[1:])


def train(epochs,l):

    songs = np.load('syllable_level_npy_39/e104b4a16b5dfee63edd8becfb661bfd.npy',allow_pickle=True)
    params = prm(l)
    optimiser = torch.optim.Adam(params, lr = 0.001)
    for e in range(epochs):
        for song in songs[:,1:2,:]:
            res = [vocab_syll[x] for x in song[2]]

            res = torch.tensor(res, dtype=torch.long)
            y = Songwriter(res)
            loss = torch.abs(y-torch.tensor(song[0]))
            loss.backwards()
            optimizer.step()
        if e %150 == 0 : print("yes")

def train(epochs,l):

    songs = np.load('syllable_level_npy_39/e104b4a16b5dfee63edd8becfb661bfd.npy',allow_pickle=True)
    params = prm(l)
    optimiser = torch.optim.Adam(params, lr = 0.001)
    for e in range(epochs):
        for song in [songs[0]]:

            res = [embeds(torch.tensor(vocab_syll[x])) for x in song[2]]
            n = len(res)
            res = torch.cat(res)
            y = Songwriter(res,n)
            loss = torch.abs(y-torch.tensor(song[0]))
            loss.backwards()
            optimizer.step()
        if e %150 == 0 : print("yes")


#Vocabulary of syllables
vocab_syll={}
i=0
with open('syllabe_set.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        vocab_syll[row['syll']]=i
        i += 1


#Embeddings
in_embed  = i+1
out_embed = 128
embeds = nn.Embedding(in_embed, out_embed)

tCon = 7
#Setting up the lyric encoder
in_shape  = out_embed
out_shape = 32 #Param
enc_x = x_encoder(in_shape, out_shape, 1)

#Setting up the melody encoder
in_shape_m  = 1
out_shape_m = 32 #Param
enc_m_pit = m_encoder(in_shape_m, out_shape_m, 1)
enc_m_dur = m_encoder(in_shape_m, out_shape_m, 1)

#Setting up the melody decoder
in_shape_de_pit = 1+2*out_shape
out_shape_de_pit = 1
in_shape_de_dur = out_shape+3
out_shape_de_dur = 1
in_shape_de_rest = out_shape+4
out_shape_de_rest = 1
dec_m_pit = m_decoder(in_shape_de_pit, out_shape_de_pit, 1)
dec_m_dur = m_decoder(in_shape_de_dur, out_shape_de_dur, 1)
dec_m_rest = m_decoder(in_shape_de_rest, out_shape_de_rest, 1)

#Setting up MC2
mc2 = MC2([1, 3, out_shape_m], [1, 1, 1])









lmods = [enc_x,
enc_m_pit,
enc_m_dur,
dec_m_pit,
dec_m_dur,
dec_m_rest,
mc2,
embeds]


gpnn = g(5+2*out_shape, 1)
gdnn = g(6+2*out_shape, 1)
grnn = g(7+2*out_shape, 1)

def gp(ypit, spit, c, y):
    return gpnn(torch.cat((ypit,spit,c,y)))

def gd(ydur, sdur, c, y, ypit):
    return gdnn(torch.cat((ydur,sdur,c,y,ypit)))

def gr(ylab, slab, c, y, ypit, ydur):
    return grnn(torch.cat((ylab,slab,c,y,ypit,ydur)))


epochs = 1

train(epochs,lmods)
