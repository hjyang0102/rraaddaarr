from transformers import T5Tokenizer, T5ForConditionalGeneration

from tqdm import tqdm

import json
import torch
import numpy as np



tokenizer = T5Tokenizer.from_pretrained("../saved/flant5-large-tokenizer-origin.pt")
model = T5ForConditionalGeneration.from_pretrained("../saved/flant5-large-model-origin.pt", device_map="auto")

device = torch.device('cuda:0')


with open('entity2id_new.json', 'r', encoding='utf-8') as f:
    entity2id = json.load(f)

def reverse_dict(my_dict):
    return {val: key for key, val in my_dict.items()}

reversed_entity2id = reverse_dict(entity2id)

with open('movies_with_age_rating.json', 'r', encoding='utf-8') as f:
    movies_info = json.load(f)

with open('item_ids.json','r',encoding='utf-8') as f:
    item_list = json.load(f)
    item_list = item_list['data']

ents_size = 24636
flant5_emb_size=1024

ents_plot_emb = torch.zeros([ents_size, flant5_emb_size]).detach()


for entID, entName in tqdm(reversed_entity2id.items()):
    if entID in item_list:

        for itemID, itemINFO in movies_info.items():
            if itemINFO['name'] == reversed_entity2id[entID]:

                input_text = itemINFO['plot']
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
                with torch.no_grad():
                    outputs = model.encoder(input_ids)

                    input_ids = input_ids.detach()

                    ents_plot_emb[entID] = outputs.last_hidden_state.mean(dim=1)[0]




ents_plot_emb = ents_plot_emb.detach().numpy()
np.save('item_plot_embeddings.npy', ents_plot_emb)
