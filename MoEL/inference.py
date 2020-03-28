from utils.data_loader import prepare_data_seq
from utils import config
from model.transformer import Transformer
from model.transformer_mulexpert import Transformer_experts
from model.common_layer import evaluate, count_parameters, make_infinite
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from copy import deepcopy
from tqdm import tqdm
import os
import time 
import numpy as np 
import math
from tensorboardX import SummaryWriter
from utils.data_loader import collate_fn
import pickle
from utils.data_loader import *
from utils.beam_omt_experts import Translator
import copy

#from torchsummary import summary

moel_model = torch.load("saved_models_testing/saved_model60000.pt")
#moel_model = moel_model.cuda()

data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size)
#print(data_loader_tst)
#evaluate(moel_model,data_loader_tst,ty="test", max_dec_step=50)

print("Value of config.label_smoothing : ",config.label_smoothing)
print("Value of naom : ",config.noam)
print("Vaue of cuda :",config.USE_CUDA)
print("Value of pretrain_emb : ",config.pretrain_emb)
print("Value of softmax : ",config.softmax)
print("Value of basic_listener : ",config.basic_learner)
print("Value of save_path : ",config.save_path)

# s = "i am sad"
s = ""
dialouge_conv = []
situation = []
data = {}
print("Enter the emotion of the conversation : ")
Emotion = input()
data['emotion'] = [Emotion]
#data['target'] = [["why","are","you","sad","?"]]
data['situation'] = []
data['context'] = []
if(os.path.exists('empathetic-dialogue/dataset_preproc.p')):
       print("LOADING empathetic_dialogue")
       with open('empathetic-dialogue/dataset_preproc.p', "rb") as f:
           [data_tra, data_val, data_tst, vocab] = pickle.load(f)
logging.info("Vocab  {} ".format(vocab.n_words))
t = Translator(moel_model, vocab)
print("start")
s = input()
# # sent11=["I","am","sad"]
# sent11 = s.split(' ')
# sent22=["I","am","happy"]
# conv1=[sent11]
# conv2=[sent11]
# conv2.append(sent22)
# diag=[conv1]
# diag.append(conv2)
# print(diag)
#diag=[]
conv = []
while(s!="exit"):
    # print(dialouge)
    if(conv == []):
        conv1 = conv
        conv1 = [s.split(" ")]
    else:
        conv1 = conv
        conv1.append(s.split(" "))
    data['context'].append(conv1)
    conv = copy.deepcopy(conv1)
    data["emotion"].append(data["emotion"][0])  
    # data['context'].append(dialouge_conv)
    # print(len(data['context']))
    # for i in range(len(data['context'])):
    #     print(i,data['context'][i])
    if(situation==[]):
        situation = s.split(' ')
    data['situation'].append(situation)
    if(len(data["context"])==3):
        data["context"] = data["context"][1:3]
        data["emotion"] = data["emotion"][1:3]
        data["target"] = data["target"][1:3]
        data["situation"] = data["situation"][1:3]
    # print(data)
    # data['context'][0]= [data_loader['context'][0][0]]
    # print("before")
    # print(data['context'])
    dataset_input = Dataset(data, vocab)
    # print(dataset_input)
    data_loader = torch.utils.data.DataLoader(dataset=dataset_input,
                                        batch_size=32,
                                        shuffle=False, collate_fn=collate_fn)
    # print("after")
    # print(data['context'])
    # s = input()
    # print(data['context'])
    # data_loader = data_loader.gpu()

    #change=================19-02-2020

    #pbar = tqdm(enumerate(data_loader),total=len(data_loader))
    
    # print("Dialogue CONV",dialouge_conv)
    # print("data[context] ",data["context"])
    #pbar = tqdm(enumerate(data_loader))
    #for j, batch in pbar:
    #    batch["input_batch"]=batch["input_batch"].cuda()
    #    batch["input_lengths"]=batch["input_lengths"].cuda()
    #    batch["mask_input"]=batch["mask_input"].cuda()
    #    batch["target_batch"]=batch["target_batch"].cuda()
    #    batch["target_lengths"]=batch["target_lengths"].cuda()
    #    #print("Here")
    #    #print(batch)
    #    sent_g = moel_model.decoder_greedy(batch,max_dec_step=60)
    #    sent_b = t.beam_search(batch, max_dec_step=60)
    #    sent_t = moel_model.decoder_topk(batch, max_dec_step=60)
    #    print("Greedy : ",sent_g)
    #    print("Beam : ",sent_b)
    #    print("Output from topk : ",sent_t)
    #    try:
    #        data["target"].append(sent_t[0].split(' '))
    #    except Exception:
    #        data["target"] = [sent_t[0].split(' ')]
    #    dialouge_conv.append(sent_t[0].split(' '))
        # data['emotion'].append(data["emotion"][0])
        # data['emotion'].append(data["emotion"][0])
        #print(sent_g)
    #=======commented---till---here---==================
    sent_g = model.decoder_greedy(batch,max_dec_step=max_dec_step)
    sent_b = t.beam_search(batch, max_dec_step=max_dec_step)
    # print(sent_g,sent_b)
    sent_t = model.decoder_topk(batch, max_dec_step=max_dec_step)
    for i, (greedy_sent, beam_sent)  in enumerate(zip(sent_g, sent_b)):
        rf = " ".join(batch["target_txt"][i])
        hyp_g.append(greedy_sent)
        hyp_b.append(beam_sent)
        #hyp_t.append(topk_sent)
        ref.append(rf)
        print_custum(emotion= batch["program_txt"][i],
                    dial=[" ".join(s) for s in batch['input_txt'][i]] if config.dataset=="empathetic" else " ".join(batch['input_txt'][i]),
                    ref=rf,
                    #hyp_t=topk_sent,
                    hyp_g=greedy_sent,
                    hyp_b=beam_sent)
    s= input()
    
    #break
    


