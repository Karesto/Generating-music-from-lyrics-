

import json
import model_embeddings 
import lyrics
import numpy as np
import model
import torch
import word2vec

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
                input_adapted += [word_pars[0][0][word][0][0:3]+[word_pars[0][2][word][j]]+[word_pars[0][3][word][j]]]
        if l > max_length:
            max_length = l
        data_adapted += [input_adapted]
    data_adapted_padded = []
    for d in data_adapted:
        l = len(d)
        for i in range(max_length-l):
            d += [[0,0,0,'<pad>','<pad>']]
        data_adapted_padded += [d]
    return np.array(data_adapted_padded)
    
def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=25):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

if torch.cuda.is_available():

    device = 'cuda'
else :
    device = 'cpu'
    
with open('syll_emb.txt', 'r') as file:
  syll_emb = json.load(file)
  
with open('word_emb.txt', 'r') as file:
  word_emb = json.load(file)
  
print("embeddings loaded")
  
import csv
data_set=[]

i=0
j=0
with open('data_set_word_not_padded.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    song = []
    for row in reader:
        if row['id_song']!= i :
            i = row['id_song']
            data_set += [song]
            song = []
            j=0
        j = j+1
        if j < 23 :
            song += [[row['start'],row['duration'],row['frequency'],row['word'],row['syllabe']]]
	
          
real_samples=data_set[1:]
print(len(real_samples))
print("data set loaded")
singGan = model.SingGan(400,syll_emb,word_emb,num_layers=1).to(device)
singGan.load_state_dict(torch.load("singGan_2_decay.bin"))
batch_size = 24
lr_G = 0.1
lr_D = 0.01
nb_epochs = 500

optimizer_G = torch.optim.SGD(singGan.generator.parameters(),lr=lr_G)
optimizer_D = torch.optim.SGD(singGan.discriminator.parameters(),lr=lr_D)



loss_D_epoch = []
loss_G_epoch = []

loss_D_epoch=np.load("lossD_decay_layer.npy",allow_pickle=True)
loss_G_epoch=np.load("lossG_decay_layer.npy",allow_pickle=True)
length = 10
EPS = 1e-4
print("begin training")
for e in range(300,nb_epochs):
    

    exp_lr_scheduler(optimizer_G,e,init_lr=lr_G,lr_decay_epoch = 50)
    exp_lr_scheduler(optimizer_D,e,init_lr=lr_D,lr_decay_epoch = 50)
    
    #real_samples = np.array(data_set)
    loss_G = 0
    loss_D = 0
    for t in range(1,int(len(real_samples)/batch_size)):
            #improving D len(real_samples[t*batch_size+j])
        
        real_batch_temp = [[[(float(real_samples[(t-1)*batch_size:(t)*batch_size][j][i][k]))for k in range(3)] for i in range (length)] for j in range(batch_size)]
        real_batch = lyrics.to_input_vector(real_batch_temp)
        real_batch = torch.Tensor(real_batch).to(device)
        z_syll = torch.FloatTensor([[singGan.syll_embedding[song[i]] for i in range(length)] for song in lyrics.data_to_lyric_syll(real_samples[t*batch_size:(t+1)*batch_size])]).to(device)
        z_word = torch.FloatTensor([[singGan.word_embedding[song[i]] for i  in range(length)] for song in lyrics.data_to_lyric_word(real_samples[t*batch_size:(t+1)*batch_size])]).to(device)
            
        if e%1 == 0:
            
            fake_batch = singGan.generate(z_syll,z_word)
            
            D_scores_on_real = singGan.discriminate(real_batch,z_syll,z_word)
            D_scores_on_fake = singGan.discriminate(fake_batch,z_syll,z_word)
            
            loss = torch.mean(- torch.log(torch.clamp( 1-D_scores_on_real,EPS,1))  - torch.log(torch.clamp(D_scores_on_fake,EPS,1)))
            optimizer_D.zero_grad()
            loss.backward()
            optimizer_D.step()

            loss_D += loss
                    
            # improving G

        fake_batch = singGan.generate(z_syll,z_word)
        
        D_scores_on_fake = singGan.discriminate(fake_batch,z_syll,z_word)
            
        loss = -torch.mean(torch.log(torch.clamp(1-D_scores_on_fake,EPS,1)))
        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()
        loss_G += loss
    if False:
              lr=lr*0.1
              optimizer_G = torch.optim.Adam(singGan.generator.parameters(),lr=lr)
              optimizer_D = torch.optim.Adam(singGan.discriminator.parameters(),lr=lr)

    if e%10 == 0:
        torch.save(singGan.state_dict(), './singGan_2_decay.bin')
        z_syll = torch.FloatTensor([[singGan.syll_embedding[song[i]] for i in range(length)] for song in lyrics.data_to_lyric_syll(real_samples[0:1])]).to(device)
        z_word = torch.FloatTensor([[singGan.word_embedding[song[i]] for i  in range(length)] for song in lyrics.data_to_lyric_word(real_samples[0:1])]).to(device)
        
        fake_batch= singGan.generate(z_syll,z_word)
        
        np.save("dl_test_epoch_"+str(e),torch.Tensor.cpu(fake_batch).detach())
        
    loss_D_epoch += [loss_D]
    loss_G_epoch += [loss_G]
    print("epoch :",e," loss D : ",loss_D," loss_G : ",loss_G)
    if e%100==0:
       np.save("lossD_decay_layer",loss_D_epoch)
       np.save("lossG_decay_layer",loss_G_epoch)

