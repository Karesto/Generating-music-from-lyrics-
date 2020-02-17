import numpy as np
from midiutil import MIDIFile

def in_beats(n,tempo,with_zeros=True):
    beats = [0.25,0.5,1,2,4,8,16,32]
    n = max(n,0)
    if with_zeros:
        beats = [0]+beats
    min = 10000
    res=0.25
    for b in beats:
        if abs(n-b)<min:
            res=b
            min = n-b
    return res

def ToolFreq2Midi(fInHz, fA4InHz = 440):
    def convert_freq2midi_scalar(f, fA4InHz):
 
        if f <= 0:
            return 0
        else:
            return (69 + 12 * np.log2(f/fA4InHz))

    fInHz = np.asarray(fInHz)
    if fInHz.ndim == 0:
       return convert_freq2midi_scalar(fInHz,fA4InHz)

    midi = np.zeros(fInHz.shape)
    for k,f in enumerate(fInHz):
        midi[k] =  convert_freq2midi_scalar(f,fA4InHz)
    
    return (midi)
            
    
def vector_to_midi(pars,name = 'test'):
    pitch = []
    duration = []
    rest = []
    EPS = 0.001
    tempo = 60/(np.mean([max(i[1],0)  for i in pars ])+EPS)
    while tempo > 300 :
 
        tempo = tempo/2
    beats = [0.25,0.5,1,2,4,8,16,32]
    for i in range(len(pars)):
        
        p =ToolFreq2Midi(pars[i][2])
        if p <0:
            p=0
        pitch += [int(p)]
        
            
        duration += [in_beats(pars[i][1],tempo,with_zeros=False)]
        rest += [in_beats(pars[i][0],tempo)]
        
 

        #syl = [(syll_pars[0][i][4])for i in range(len(syll_pars[0]))]

    time = 0
    print(pitch)
    volume   = 100  # 0-127, as per the MIDI standard

    MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
                          # automatically)
    MyMIDI.addTempo(0, 0, tempo)

    for i in range(len(pitch)):
        
        MyMIDI.addNote(0, 0, pitch[i], time + rest[i] , duration[i], volume)
        time = time + duration[i] + rest[i]

    with open(name+".mid", "wb") as output_file:
        MyMIDI.writeFile(output_file)

        
