# adapt from here: https://huggingface.co/docs/transformers/tasks/sequence_classification#load-imdb-dataset
import os
import torch
import pickle
import evaluate
import statistics
import numpy as np

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import IntervalStrategy
from transformers import DataCollatorWithPadding
from datasets import load_dataset, Dataset


#======================
# ===== CONSTANTS =====
#======================
label2id = {'Question_answer_pair':0,
        'Comment':1,
        'Acknowledgement':2,
        'Continuation':3,
        'Elaboration':4,
        'Q_Elab':5,
        'Result':6,
        'Contrast':7,
        'Explanation':8,
        'Clarification_question':9,
        'Parallel':10,
        'Correction':11,
        'Alternation':12,
        'Narration':13,
        'Conditional':14,
        'Background':15}

id2label = {0: 'Question_answer_pair',
        1: 'Comment',
        2: 'Acknowledgement',
        3: 'Continuation',
        4: 'Elaboration',
        5: 'Q_Elab',
        6: 'Result',
        7: 'Contrast',
        8: 'Explanation',
        9: 'Clarification_question',
        10: 'Parallel',
        11: 'Correction',
        12: 'Alternation',
        13: 'Narration',
        14: 'Conditional',
        15: 'Background'}

id2labelshort = {0: 'qap',
        1: 'comm',
        2: 'ack',
        3: 'cont',
        4: 'elab',
        5: 'q_el',
        6: 'res',
        7: 'contr',
        8: 'expl',
        9: 'clari',
        10: 'para',
        11: 'corr',
        12: 'alte',
        13: 'narr',
        14: 'cond',
        15: 'back'}


# stratified so every relation has the same distribution in train and dev, cf find_perfect_train_dev_split() in aux.py script
dev_stratified_train = ['pilot04_2', 's1-league2-game2_2', 's1-league1-game4_2', 's2-league4-game1_1', 's1-league3-game1_2', 
                        's1-league1-game4_3', 's2-league3-game5_3', 's2-league5-game5_2', 's2-league3-game5_1', 's1-league1-game2_3', 
                        's2-practice4_2', 's2-league4-game1_5', 's2-league4-game3_1', 's2-league4-game3_3', 's2-practice3_3', 
                        's2-league5-game2_1', 's1-league1-game2_2', 's1-league1-game3_1', 's1-league1-game5_5', 's2-league5-game5_1', 
                        's2-league5-game3_3', 's2-practice4_4', 's1-league1-game2_4', 's2-leagueM-game3_1', 's2-leagueM-game5_1', 
                        's2-leagueM-game2_3', 'pilot01_1', 's2-practice2_4', 's2-league3-game5_4', 's2-league5-game4_4', 
                        'pilot03_5', 'pilot03_2', 's1-league2-game1_3', 's2-leagueM-game3_2', 's1-league1-game1_2', 
                        's1-league3-game1_4', 'pilot14_4', 's2-practice4_3', 's2-practice2_3', 's1-league3-game4_1', 
                        's1-league1-game4_4', 's1-league1-game3_2', 's2-practice2_6', 's2-leagueM-game2_1', 's1-league1-game3_3', 
                        's1-league2-game3_1', 's2-practice3_1', 's1-league1-game5_2', 's2-league4-game1_4', 'pilot01_5', 's2-practice3_2']

dev_stratified_dev = ['s2-league5-game2_2', 's2-practice2_2', 's2-league3-game1_3', 'pilot03_4', 's2-league5-game0_1', 
                      's1-league3-game5_1', 's2-league4-game1_3', 'pilot01_4', 's1-league3-game1_1', 's1-league1-game4_1', 
                      's2-league5-game3_1', 's1-league2-game1_2', 's2-league4-game1_6', 's2-league4-game3_4', 'pilot14_5', 
                      's1-league1-game5_6', 's1-league2-game2_3', 's1-league1-game5_4', 'pilot04_1', 's2-league5-game4_3', 
                      's2-league5-game4_2', 's1-league1-game5_1', 's2-league5-game4_1', 's2-practice2_5', 's2-league5-game1_2', 
                      's1-league1-game5_3', 's2-league3-game1_2', 's2-leagueM-game2_2', 's2-league3-game5_2', 's2-league3-game1_1',
                      's1-league1-game2_1', 'pilot14_3', 's2-league5-game1_1', 'pilot01_3', 's1-league2-game1_1', 's1-league3-game1_3', 
                      's2-league4-game3_2', 'pilot20_1', 'pilot01_2', 'pilot03_3', 's2-leagueM-game3_3', 's1-league1-game1_1', 
                      's2-practice4_1', 's2-practice2_1', 's2-league5-game2_3', 's2-league4-game1_2', 'pilot14_2', 'pilot03_1', 
                      's2-league5-game3_2', 's1-league2-game2_1', 'pilot20_2', 'pilot14_1']

stratified_train_dev = {'dev': {'train': dev_stratified_train, 'dev': dev_stratified_dev}}


#========================
#===== TRAIN & EVAL =====
#========================

def preprocess_function(examples):
    tokenized_batch = tokenizer(examples["text"], padding='max_length', max_length=512, truncation=True, return_tensors="pt")
    tokenized_batch["label"] = [label2id[label] for label in examples["label"]]
    return tokenized_batch

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def train(model, tokenizer, train_data, dev_data, out_dir, learning_rate, epochs, steps=500):

    training_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=2, 
        per_device_eval_batch_size=2, 
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=steps,
        save_strategy="steps",
        save_steps = steps, 
        log_level='info',
        logging_strategy='steps',
        logging_steps=steps,
        save_total_limit=2,
        push_to_hub=False,
        report_to=None,
        metric_for_best_model='accuracy',
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=512)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer.state.log_history
    

def predict(test_data, model_dir, device):
    # parameters:
    #   - test_data: list of strings, or just a string
    #   - model_dir: name of huggingface model or path to fine-tuned model checkpoint
    #   - device: cuda or cpu
    
    text = test_data['train']['text'] #len=1128
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device) #also need to put input text to cuda

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    
    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits

    predicted_class_id = torch.argmax(logits, dim=1).tolist()
    gold = [model.config.label2id[label] for label in test_data['train']['label']]
    
    acc = accuracy.compute(predictions=predicted_class_id, references=gold)
    return acc
    


if __name__=="__main__":
    
    MODE = 'TRAIN' # 'TRAIN', 'EVAL'

    my_model_path = "MODEL_PATH"

    trainset = 'train'
    testset = 'test'
    trainf = "DATA_PATH/stac_train_spk.csv"
    devf = "DATA_PATH/stac_dev_spk.csv"
    testf = "DATA_PATH/stac_test_spk.csv"

    for subset_train_size in [700, 1500, 2500, 5000, 7500]:
        test_record = []
        for seed in [27]: # [27, 30, 42, 55, 78]
            data_train = load_dataset('csv', data_files=trainf, delimiter='\t')['train'] #note: load local csv return a datasetDict object with the key 'train'
            base_train = Dataset.from_dict(data_train.shuffle(seed=seed)[:subset_train_size])

            data_dev = load_dataset('csv', data_files=devf, delimiter='\t')['train']
            base_dev = Dataset.from_list([i for i in data_dev if i['file'] in stratified_train_dev['dev']['dev']])
            
            data_test = load_dataset('csv', data_files=testf, delimiter='\t')
            print(len(base_train['text']), len(base_dev['text']), len(data_test['train'])) # 9163 1019 1128
            
            model_dir = os.path.join(my_model_path, f"bert_base_stac_{trainset}{subset_train_size}_seed{seed}")
            
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")

            if MODE == 'TRAIN':
                # set up model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
                tokenized_train = base_train.map(preprocess_function, batched=True)
                tokenized_dev = base_dev.map(preprocess_function, batched=True)

                model = AutoModelForSequenceClassification.from_pretrained(
                    "bert-base-cased", num_labels=16, id2label=id2label, label2id=label2id)

                LR = 2e-5
                EPOCH = 10
                STEP = 500
                
                training_logs = train(model, tokenizer, tokenized_train, tokenized_dev, 
                                      out_dir=model_dir, learning_rate=LR, epochs=EPOCH, steps=STEP)

                # recording train subset and finetune result
                train_dev = {'trainset': base_train, 'devset': base_dev, 'losslog': training_logs}
                pif = os.path.join(model_dir, 'traininglog')
                with open(pif, 'wb') as outf:
                    pickle.dump(train_dev, outf)

            elif MODE == 'EVAL':
                # TODO: change to your own finetuned model checkpoint
                train2checkpoint = {
                    'train700_27': 'checkpoint-1400', 'train700_30': 'checkpoint-1400', 'train700_42': 'checkpoint-1200', 'train700_55': 'checkpoint-1200', 'train700_78': 'checkpoint-1400',\
                    'train1500_27': 'checkpoint-3200', 'train1500_30': 'checkpoint-2800', 'train1500_42': 'checkpoint-3200', 'train1500_55': 'checkpoint-3200', 'train1500_78': 'checkpoint-3200',\
                    'train2500_27': 'checkpoint-2500', 'train2500_30': 'checkpoint-4000', 'train2500_42': 'checkpoint-3000', 'train2500_55': 'checkpoint-2500', 'train2500_78': 'checkpoint-4500',\
                    'train5000_27': 'checkpoint-8500', 'train5000_30': 'checkpoint-6500', 'train5000_42': 'checkpoint-6500', 'train5000_55': 'checkpoint-8000', 'train5000_78': 'checkpoint-11000',\
                    'train7500_27': 'checkpoint-12000', 'train7500_30': 'checkpoint-13000', 'train7500_42': 'checkpoint-9500', 'train7500_55': 'checkpoint-4500', 'train7500_78': 'checkpoint-6000',\
                    } 
                checkpoint = f"bert_base_stac_{trainset}{subset_train_size}_seed{seed}/" + train2checkpoint[f'{trainset}{subset_train_size}_{seed}']
                acc = predict(data_test, model_dir=os.path.join(my_model_path, checkpoint), device=device)
                test_record.append(acc['accuracy']*100)

            else:
                raise ValueError(f"choose MODE from TRAIN or EVAL.")
        
        print(f"{subset_train_size:<5}: avg={sum(test_record)/len(test_record):.2f}, stdev={statistics.stdev(test_record):.2f}, {test_record}" )
        test_record = []