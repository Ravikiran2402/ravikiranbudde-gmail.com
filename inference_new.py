from utils.data_loader import prepare_data_seq
from utils import config
from model.transformer import Transformer
from model.transformer_mulexpert import Transformer_experts
from model.common_layer import evaluate, count_parameters, make_infinite , print_custum
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

#==========================================================================#
_,_,_,_,program_number =prepare_data_seq(batch_size=config.batch_size)
_,_,_,vocab = load_dataset()
logging.info("Vocab  {} ".format(vocab.n_words))

max_dec_step = 50

#saved_model = torch.load("saved_models_testing/saved_model60000.pt")
saved_model =  Transformer_experts(vocab,decoder_number=program_number, model_file_path="save2/model_21999", is_eval=True)

saved_model.cuda()
saved_model = saved_model.eval()

#==============change===============20-03-2020
#model_11999_43.5771_0.0000_0.0000_0.0000_0.0000

#==============change===============20-03-2020
saved_model.__id__logger = 0

#========change======= 20-03-2020
#saved_model.eval()
#========change======= 20-03-2020

dial = []
ref, hyp_g, hyp_b, hyp_t = [],[],[],[]
t = Translator(saved_model, saved_model.vocab)
l = []
p = []
bce = []
acc = []
data_test = {}
data_test["emotion"] = []
data_test["situation"] = []
data_test["target"] = []
data_test["context"] = []

target = ['This','is','the','default','value','for','the','target','attribute','.']

print("Enter the emotion of the conversation : ")
emotion = input()

print("Enter the situation of the conversation : ")
situation = input()
situtaion = situation.split(' ')

context = []
#print(type(data))
while(True):
    print("You :")
    s = input()
    if(s=="exit"):
        break
    #=====================creating=data======================#
    data_test["emotion"].append(emotion)
    data_test["target"].append(target)
    context.append(s.split(' '))
    data_test["context"].append(context)
    data_test["situation"].append(situation)
    
    if(len(data_test["emotion"])>2):
        data_test["emotion"] = data_test["emotion"][1:]
    if(len(data_test["situation"])>2):
        data_test["situation"] = data_test["situation"][1:]
    if(len(data_test["context"])>2):
        data_test["context"] = data_test["context"][1:]
    if(len(data_test["target"])>2):
        data_test["target"] = data_test["target"][1:]

    dataset_test = Dataset(data_test, vocab)
    data = torch.utils.data.DataLoader(dataset=dataset_test,
                                                 batch_size=1,
                                                 shuffle=False, collate_fn=collate_fn)

    #========================================================#
    pbar = tqdm(enumerate(data),total=len(data))
    output = []
    for j, batch in pbar:
        #print(type(batch))
        loss, ppl, bce_prog, acc_prog = saved_model.train_one_batch(batch, 0, train=False)
        l.append(loss)
        p.append(ppl)
        bce.append(bce_prog)
        acc.append(acc_prog)
        sent_g = saved_model.decoder_greedy(batch,max_dec_step=max_dec_step)
        sent_b = t.beam_search(batch, max_dec_step=max_dec_step)
        sent_t = saved_model.decoder_topk(batch, max_dec_step=max_dec_step)
        for i, (greedy_sent,beam_sent,topk_sent)  in enumerate(zip(sent_g,sent_b,sent_t)):
            rf = " ".join(batch["target_txt"][i])
            hyp_g.append(greedy_sent)
            hyp_b.append(beam_sent)
            #======change========19-02-2020
            hyp_t.append(topk_sent)
            output.append(topk_sent)
            ref.append(rf)
            #======change========19-02-2020
            """
            print_custum(emotion= batch["program_txt"][i],
                        dial=[" ".join(s) for s in batch['input_txt'][i]] if config.dataset=="empathetic" else " ".join(batch['input_txt'][i]),
                        ref=rf,
                        hyp_g=greedy_sent,
                        hyp_b=beam_sent,
                        hyp_t=topk_sent)
            """
        pbar.set_description("loss:{:.4f} ppl:{:.1f}".format(np.mean(l),math.exp(np.mean(l))))
    
    output = output[0]
    #======================#
    print("MOEL : {}".format(output))
    #=====================#
    output = output.split(' ')
    context.append(output)        



