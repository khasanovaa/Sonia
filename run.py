import numpy as np
from numpy import exp
import mido
from mido import MidiTrack
from mido import MidiFile, MetaMessage
from midi2audio import FluidSynth
import torch
import os
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device('cpu')
tempo = 2000000
count = 0
hid_size = 512
time_cl = [0, 1, 2, 3, 5, 7, 11, 15, 20, 23, 35, 80, 122, 187]
vel_cl = [0, 15, 33, 64, 80, 104, 125]
note_cl = [52, 48, 55, 67, 64, 60, 57, 69, 70, 58, 63, 51, 50, 65, 66, 68, 53, 72, 74, 62, 75, 36, 76, 79, 43, 31, 39, 40, 71, 45, 73, 34, 35, 38, 29, 32, 33, 41, 44, 77, 81, 37, 59, 47, 46, 82, 84, 87, 88, 100, 94, 98, 96, 86, 93, 91, 101, 80, 89, 83, 99, 42, 78, 90, 54, 56, 49, 28, 26, 61, 30, 85, 92, 95, 27, 24, 97, 103, 22, 102, 105, 107, 108, 104, 106]
SOS_token = len(note_cl)*2 + len(time_cl) + len(vel_cl)

def evaluate(decoder, decoder_hidden):
    with torch.no_grad():
        target_length =1000
        outputs = []
        decoder_input = torch.tensor([[SOS_token]]).to(device)
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            outputs.append(topi[0].item())
            decoder_input = torch.tensor([[topi[0].item()]]).to(device)
        return outputs

def postprocess(v, seed):
    global count
    d = decoding(v)
    events = d
    for i in range(len(events)):
        event = events[i]
        if (event[0] == 'v'):
            event[1] = vel_cl[event[1]]
        elif (event[0] == 't'):
            event[1] = time_cl[event[1]]
        else:
            event[1] = note_cl[event[1]]
    track = MidiTrack()
    track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
    velocity_n = 0
    time_n = 0
    for i in range(len(events)):
        if (events[i][0] == 'v'):
            velocity_n = events[i][1]
        if (events[i][0] == 't'):
            time_n = events[i][1]
        if (events[i][0] == 'no'):
            track.append(mido.Message('note_on', note=events[i][1], time=time_n, velocity=velocity_n))
        if (events[i][0] == 'nof'):
            track.append(mido.Message('note_off', note=events[i][1], time=time_n, velocity=velocity_n))
    mid_f = MidiFile()
    mid_f.tracks.append(track)
    mid_f.save('output_' + str(seed) + '.mid')
    count += 1

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

def process(seed):
    hid_size = 512
    classes = SOS_token + 1
    np.random.seed(seed)
    decoder = DecoderRNN(hid_size, classes)
    decoder.load_state_dict(torch.load("decoders/decoder4", map_location='cpu'))
    decoder_hidden = torch.FloatTensor([np.random.rand() for _ in range(hid_size)]).view(1,1,-1).to(device)
    postprocess(evaluate(decoder, decoder_hidden), seed)
    fs = FluidSynth(os.getcwd() + '/GeneralUser_GS_SoftSynth_v144.sf2')
    fs.midi_to_audio('output_' + str(seed) + '.mid', 'new_song_' + str(seed) + '.wav')
    
def decoding(ans):
    events = []
    for n in ans:
        if (n < len(vel_cl)):
            events.append(['v', n])
        elif (n < len(vel_cl) + len(time_cl)):
            events.append(['t', n - len(vel_cl)])
        elif (n < len(vel_cl) + len(time_cl) + len(note_cl)):
            events.append(['no', n - len(vel_cl) - len(time_cl)])
        else:
            events.append(['nof', n - len(vel_cl) - len(time_cl) - len(note_cl)])
    return events

print("print seed")
seed = int(input())
process(seed)
