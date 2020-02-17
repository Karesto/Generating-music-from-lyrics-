
import csv
import numpy as np

def to_input_vector(batch):
    res = [[[0,0,0] for note in song] for song in batch]
    for i,song in enumerate(batch):
        for j,note in enumerate(song):
            res[i][j][0] = note[0]
            res[i][j][1] = note[1]
            res[i][j][2] = note[2]
           
    return res

def to_input_vector_tune(batch):
    res = [[[0] for note in song] for song in batch]
    for i,song in enumerate(batch):
        for j,note in enumerate(song):
            res[i][j][0] = note[2]
           
    return res



def data_to_lyric_syll(data):
    lyr = []
    for i in range(len(data)):
        lyr += [[data[i][0][4]]]
        for j in range(1,len(data[i])):
            lyr[i] += [data[i][j][4]]
    return np.array(lyr)

def data_to_lyric_word(data):
    lyr = []
    for i in range(len(data)):
        lyr += [[data[i][0][3]]]
        for j in range(1,len(data[i])):
            lyr[i] += [data[i][j][3]]
    return np.array(lyr)


def to_input_size(data):
    data_adapted = []
    max_length = 0
    for syll_pars in data:
        input_adapted = []
        l = len(syll_pars[0][0])
        if l > max_length:
            max_length = l
        for j in range(l):
            input_adapted += [syll_pars[0][1][j]+[int(60*syll_pars[0][1][j][1]/syll_pars[0][0][j][1])]+[syll_pars[0][2][j]]]
        data_adapted += [input_adapted]
    data_adapted_padded = []
    for d in data_adapted:
        l = len(d)
        for i in range(max_length-l):
            d += [[0,0,0,0,'<pad>']]
        data_adapted_padded += [d]
    return np.array(data_adapted_padded)
        
def to_input_size_word(data):
    data_adapted = []
    max_length = 0
    max_length_syll = 0
    for word_pars in data:
        input_adapted = []
        l=0
        for word in range(len(word_pars[0][0])):
            l += len(word_pars[0][0][word])
            for j in range(len(word_pars[0][0][word])):
                input_adapted += [word_pars[0][1][word][j]+[int(60*word_pars[0][1][word][j][1]/word_pars[0][0][word][j][1])]+[word_pars[0][2][word][j]]+[word_pars[0][3][word][j]]]
        if l > max_length:
            max_length = l
        data_adapted += [input_adapted]
    data_adapted_padded = []
    for d in data_adapted:
        l = len(d)
        for i in range(max_length-l):
            d += [[0,0,0,0,'<pad>','<pad>']]
        data_adapted_padded += [d]
    return np.array(data_adapted_padded)
        
