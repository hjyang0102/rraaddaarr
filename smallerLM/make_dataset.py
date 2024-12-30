import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import T5Tokenizer

class CCdataset(Dataset):
    def __init__(
        self, dataset, split, datafile, tokenizer
    ):
        super(CCdataset, self).__init__()
        max_input_length = 1024


        self.tokenizer = tokenizer
        self.data = []
        self.prepare_data(datafile, max_input_length)

    def prepare_data(self, data_file, max_input_length):
         with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for conv_id, conv_info in data.items():
                context_ids = self.tokenizer(conv_info['context'], max_length=max_input_length, truncation=True)
                CC_ids = self.tokenizer(conv_info['list_rationale_CC_nextresponse_O'], max_length=max_input_length, truncation=True)

                data={
                    'context': context_ids,
                    'CC': CC_ids
                }
                self.data.append(data)
    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = 'redial'
    split = 'test'
    tokenizer = T5Tokenizer.from_pretrained("./saved/flant5-large-tokenizer-origin.pt")
    datafile = 'train_data_augmented_CC.json'



    dataset = CCdataset(dataset, split, datafile = datafile, tokenizer = tokenizer)
    for i in range(len(dataset)):
        if i == 3:
            break
        data = dataset[i]
        print(tokenizer.decode(data['context']['input_ids']))
        print(tokenizer.decode(data['CC']['input_ids']))
        print()
