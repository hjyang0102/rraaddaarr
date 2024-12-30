import json
import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from evaluate import load
from datasets import load_dataset
import nltk

import numpy as np

tokenizer = T5Tokenizer.from_pretrained("./saved/flant5-large-tokenizer-origin.pt")

max_input_length = 1024
max_target_length = 1024
examples = {}

def preprocess_function(examples):
    inputs = examples['context']
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["list_rationale_CC_nextresponse_O"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_file = 'train_forLcnd_tau_0.6.json'
test_file = 'train_forLcnd_tau_0.6.json'

data_files={'train': train_file, 'test': test_file}

raw_datasets = load_dataset('json' , data_files=data_files, field='data')

metric = load("rouge")
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)


#================================================================================

model = T5ForConditionalGeneration.from_pretrained("./saved/flant5-large-model-origin.pt", device_map="auto")


batch_size = 32
model_name = 'my_flant5'
args = Seq2SeqTrainingArguments(
    model_name,
    evaluation_strategy = "epoch", # evalute after every epoch
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.00001,
    save_total_limit=8,
    num_train_epochs=8,
    predict_with_generate=True,
    fp16=False,  #True,
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]




    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)

    # Extract a few results
    #result = {key: value * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()
print("-------------train done-------------")

trainer.save_model('CC_trained_0.6')


#trainer.evaluate()
