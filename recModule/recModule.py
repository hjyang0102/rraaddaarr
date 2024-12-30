import math
import numpy as np
import torch
import transformers
import json
from torch import nn
from tqdm.auto import tqdm

from kg_encoder import KGencoder
from evaluate_rec import RecEvaluator

from transformers import T5ForConditionalGeneration

from torch.nn import functional as F
device = torch.device('cuda:0')


class recommendation(nn.Module):
    def __init__(self, kg, pretrained_smallerLM):
        super(recommendation, self).__init__()
        self.KG_encoder = KGencoder(num_relations=kg['num_relations'],n_entity=kg['num_entities'],edge_index=kg['edge_index'], edge_type=kg['edge_type'])
        self.smallerLM = T5ForConditionalGeneration.from_pretrained(pretrained_smallerLM, device_map="auto")
        for param in self.smallerLM.parameters():
            param.requires_grad = False

        self.num_entities = kg['num_entities']  #24636



        #self.ent_embs_in_kg = self.KG_encoder()  #[24636,128]

        self.nnlayer_for_plot = nn.Linear(1024, 128, bias=False)

        #self.nnlayer_for_context = nn.Linear(1024, 128, bias=False) #FlanT5_emb_size, R-GCN_emb_size
        self.fc1 = nn.Linear(1024, 768, bias=False)
        self.fc2 = nn.Linear(768, 512, bias=False)
        self.fc3 = nn.Linear(512, 128, bias=False)
        self.relu = nn.ReLU()

        self.item_plot_embeddings = torch.tensor(np.load('item_plot_embeddings.npy')).to(device).detach() #[24636,1024]


        self.item_list = []
        with open('item_ids.json','r',encoding='utf-8') as f:
            item_list = json.load(f) #f.readline()
            self.item_list = item_list['data']


    def generate_ent_item_plot_embs(self):
        ent_embs_in_kg = self.KG_encoder() #[24631,128]


        item_plot_embeddings = self.nnlayer_for_plot(self.item_plot_embeddings) #[24631,1024] > [24631,128]

        ent_item_plot_embs = torch.cat((ent_embs_in_kg,item_plot_embeddings), dim=1)

        return ent_item_plot_embs


    def generate_rec_label(self,rec_label):

        batch_size = rec_label.size()[0]
        labels = torch.zeros(batch_size,self.num_entities)

        for count, label in enumerate(rec_label):
            labels[count][label]=1

        return labels

    def generate_rec_label_multiple(self,rec_label):

        list_size = len(rec_label)
        batch_size = rec_label[0].size()[0]
        labels = torch.zeros(batch_size,self.num_entities)


        for i in range(batch_size):
            for batch_count, tensor in enumerate(rec_label):
                labels[i][tensor[i].item()]=1



        return labels

    def generate_preferred_attr_label(self,preferred_attr):

        batch_size = preferred_attr.size()[0]
        labels = torch.zeros(batch_size,self.num_entities)

        for count, label in enumerate(preferred_attr):
            labels[count][label]=1

        return labels

    def cross_attention(self, query, key, value):
        dim_k = query.size(-1)
        scores = torch.mm(query, key.transpose(0, 1)) / math.sqrt(dim_k)
        weights = F.softmax(scores, dim=-1)
        return torch.mm(weights, value)

    def get_nn_encoded_context_embs(self, encoded_context_embs):
        #return self.nnlayer_for_context(encoded_context_embs)
        x = self.fc1(encoded_context_embs)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    def generate_user_emb_for_preferred_attr(self, ent_list, context_ids):
        ent_embs = self.KG_encoder() # device:0

        bs_size = ent_list[0].size()[0]
        list_size = len(ent_list)


        uset_emb_list = []
        for i in range(bs_size):
        #for tensor in ent_list:

            out_list = []
            count = 0
            #for i in range(bs_size):
            for tensor in ent_list:
                if tensor[i].item() == 24635:
                    if count == 0:
                        out_list.append(torch.zeros(128).unsqueeze(0).to(device).detach())
                        break
                    else:
                        break

                else:
                    count = count + 1
                    out_list.append(ent_embs[tensor[i].item()].unsqueeze(0))

            mean_tensor = torch.mean(torch.stack(out_list), dim=0)
            if (mean_tensor == 0).all():
                mean_tensor = ent_embs[24635].unsqueeze(0).to(device).detach()
            uset_emb_list.append(mean_tensor)


        user_embs = torch.cat(uset_emb_list, dim=0)


        output = self.smallerLM.encoder(context_ids)
        encoded_context_embs = output.last_hidden_state.mean(dim=1)

        encoded_encoded_context_embs = self.get_nn_encoded_context_embs(encoded_context_embs)

        user_embs = self.cross_attention(encoded_encoded_context_embs, user_embs, user_embs)

        return user_embs, encoded_context_embs

    def generate_user_emb_for_recommendation(self, ent_list, context_ids):
        user_embs_for_kg, encoded_context_embs = self.generate_user_emb_for_preferred_attr(ent_list, context_ids)
        user_embs_for_lm = self.get_nn_encoded_context_embs(encoded_context_embs)


        return torch.cat((user_embs_for_kg,user_embs_for_lm), dim=1)



    def cal_logits_for_preferred_attr(self, user_embs):
        ent_embs = self.KG_encoder()  #[24631,128]

        logits = user_embs @ ent_embs.T
        return logits

    def cal_logits_for_recommendation(self, user_embs):
        ent_item_plot_embs = self.generate_ent_item_plot_embs()

        logits = user_embs @ ent_item_plot_embs.T

        return logits


    def exclude_attrs_only_items_for_rec_training(self, labels):
        mask = torch.ones_like(labels, dtype=torch.bool)
        mask[:, self.item_list] = False


        masked_labels = labels.masked_fill(mask, float('-inf')) # 제외할 값을 -inf로 변경

        return masked_labels


    def training_preferred_attrs(self,ent_list,preferred_attr, context_ids):
        preferred_attr_labels = self.generate_preferred_attr_label(preferred_attr)

        user_embs = self.generate_user_emb_for_preferred_attr(ent_list, context_ids)

        rec_logits = self.cal_logits_for_preferred_attr(user_embs)

        preferred_attr_labels = preferred_attr_labels.to(device).detach()

        preferred_attr_loss = F.cross_entropy(rec_logits, preferred_attr_labels, reduction='mean')

        return preferred_attr_loss

    def training_recommendation(self, ent_list, rec_label, context_ids):

        recommendation_labels = self.generate_rec_label(rec_label)
        user_embs = self.generate_user_emb_for_recommendation(ent_list, context_ids)

        rec_logits = self.cal_logits_for_recommendation(user_embs)


        recommendation_labels = recommendation_labels.to(device).detach()


        recommendation_loss = F.cross_entropy(rec_logits, recommendation_labels, reduction='mean')

        return recommendation_loss

    def training_recommendation_multiple_rec(self, ent_list, rec_label, context_ids, rec_items):


        user_embs = self.generate_user_emb_for_recommendation(ent_list, context_ids)
        recommendation_labels = self.generate_rec_label_multiple(rec_items)

        rec_logits = self.cal_logits_for_recommendation(user_embs)


        recommendation_labels = recommendation_labels.to(device).detach()


        bce_criterion = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')
        recommendation_loss = bce_criterion(rec_logits, recommendation_labels)

        return recommendation_loss



    def eval_recommendation(self, ent_list, rec_label, context_ids, rec_items):

        user_embs = self.generate_user_emb_for_recommendation(ent_list, context_ids)

        rec_logits = self.cal_logits_for_recommendation(user_embs)

        mask = torch.ones_like(rec_logits, dtype=torch.bool)
        mask[:, self.item_list] = False
        masked_logits = rec_logits.masked_fill(mask, float('-inf'))

        ranks = torch.topk(masked_logits, k=50, dim=-1).indices


        return ranks, rec_label


    def forward(self,ent_list, rec_label, context_ids, preferred_attr, mode, rec_items):

        if mode == 'training_preferred_attrs':
            return self.training_preferred_attrs(ent_list,preferred_attr, context_ids)
        elif mode == 'training_recommendation':
            return self.training_recommendation(ent_list,rec_label, context_ids)
        elif mode == 'training_recommendation_multiple_rec':
            return self.training_recommendation_multiple_rec(ent_list,rec_label, context_ids, rec_items)
        elif mode == 'eval_recommendation':
            return self.eval_recommendation(ent_list,rec_label, context_ids, rec_items)
        elif mode == 'eval_recommendation_multiple_rec':
            return self.eval_recommendation(ent_list,rec_label, context_ids, rec_items)
        else:
            return 0
