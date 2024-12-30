import warnings
warnings.filterwarnings('ignore')


from tqdm.auto import tqdm
import torch
import wandb


from kg_unicrs import KGForUniCRS
from recModule_multiple_rec import recommendation
from koalaDataset_multiple_rec import ReDialDataset
from evaluate_rec import RecEvaluator



from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

sweep_config = {
    'method': 'random', #grid, random
    'parameters': {
        'epochs': {
            'values': [8,12]
        },
        'batch_size': {
            'values': [32] #8, 16,
        },
        'learning_rate': {
            'values': [1e-3] #[1e-3, 5e-4, 1e-4, 5e-5]
        },
        'weight_decay' : {
            'values': [1e-4] #[5e-5, 1e-5, 5e-6, 1e-6]
        },
        'tau' : {
            'values' : [35, 45, 50, 55, 65]
        },
    }
}

sweep_id = wandb.sweep(sweep_config, project="RADAR")





device = torch.device('cuda:0')


def train_main():

    config_defaults = {
        'epochs': 8,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay' : 5e-5,
        'tau' : 50,
    }
    wandb.init(config=config_defaults)
    config = wandb.config

    batch_size = config.batch_size
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay


    train_file = 'train_data_new.json'
    test_file = 'test_data_new.json'


    raw_dataset = load_dataset('json' , data_files ={'train' : train_file , 'validation' : test_file}, field='data')


    tokenizer_path = "../saved/flant5-large-tokenizer-origin.pt"
    preprocessed_dataset = ReDialDataset(raw_dataset['train'], tokenizer_path)
    train_dataloader = DataLoader(preprocessed_dataset, batch_size=batch_size, shuffle=True)
    preprocessed_dataset = ReDialDataset(raw_dataset['validation'], tokenizer_path)
    test_dataloader = DataLoader(preprocessed_dataset, batch_size=batch_size, shuffle=True)



    kg = 'MYdata' #'redial'
    kg = KGForUniCRS(kg=kg).get_kg_info()

    if config.tau == 40:
        pretrained_smallerLM = "../smallerLM/CC_trained_0.4"
    else:
        pretrained_smallerLM = "../smallerLM/CC_trained_0.6"

    recommendationModule = recommendation(kg,pretrained_smallerLM).to(device)
    optimizer = torch.optim.AdamW(recommendationModule.parameters(), lr=learning_rate, weight_decay=weight_decay)


    evaluator = RecEvaluator()
    for epoch in range(config.epochs):
        count = 0
        closs = 0
        for ent_list, rec_label, context_ids, most_preferred_attr, rec_items in tqdm(train_dataloader):
            mode = 'training_recommendation'
            count = count + 1




            rec_loss = recommendationModule(ent_list, rec_label, context_ids, most_preferred_attr, mode, rec_items)

            optimizer.zero_grad()
            rec_loss.backward()
            closs = closs + rec_loss.item()
            optimizer.step()

            if count%5000 ==0:
                print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}"
                    .format(epoch+1, config.epochs, count+1, len(train_dataloader), rec_loss.item()))
                wandb.log({"epoch ":epoch+1})
                wandb.log({"batch loss":rec_loss.item()})
        wandb.log({"loss":closs/batch_size})


        for ent_list, rec_label, context_ids, most_preferred_attr, rec_items in tqdm(test_dataloader):
            mode = 'eval_recommendation'
            recommendationModule.eval()
            with torch.no_grad():
                ranks, rec_label = recommendationModule(ent_list, rec_label, context_ids, most_preferred_attr, mode, rec_items)
                evaluator.evaluate(ranks, rec_label)

        report = evaluator.report()
        evaluator.reset_metric()
        valid_report = {}
        for k, v in report.items():
            valid_report[f'valid/{k}'] = v
        valid_report['epoch'] = epoch

        print(valid_report)
        wandb.log({"valid_report":valid_report})


wandb.agent(sweep_id, train_main)
