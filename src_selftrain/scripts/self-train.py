import os
import datetime
import pickle
import statistics
import torch
import evaluate
import numpy as np
from collections import Counter, defaultdict

from torch.nn import functional as F
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import IntervalStrategy
from transformers import DataCollatorWithPadding
from datasets import load_dataset, Dataset, concatenate_datasets


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
    tokenized_batch["label"] = [label2id[label] for label in examples["label"]] #convert string label to int
    return tokenized_batch

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def get_train700_distrib(seed=27):
    file = f"DATA_PATH/combined/al_train700_seed{seed}.csv"
    data = load_dataset('csv', data_files=file, delimiter='\t')['train']
    labels = Counter(data['label'])
    return {lab: cnt/700 for lab, cnt in labels.items()}


def train(model, tokenizer, train_data, dev_data, out_dir, tb_file, learning_rate, epochs, steps=500):

    training_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=learning_rate, #2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2, #2 | 8
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy = IntervalStrategy.STEPS,
        eval_steps = steps,
        save_strategy="steps",
        save_steps=steps,
        log_level='info',
        logging_strategy='steps',
        logging_steps=steps,
        logging_dir=tb_file,
        save_total_limit=2,
        push_to_hub=False,
        report_to='tensorboard',
        metric_for_best_model = 'accuracy',
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
    
    text = test_data['text'] #could be the rest in train set or 1128 test set
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device) #also need to put input text to cuda

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    
    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits

    predicted_class_id = torch.argmax(logits, dim=1).tolist()
    gold = [model.config.label2id[label] for label in test_data['label']]
    details = {'prediction': predicted_class_id, 'gold': gold} 

    acc = accuracy.compute(predictions=predicted_class_id, references=gold) #output is a dict

    output_probs = F.softmax(logits.to('cpu')) #[1128, 16]
    pred_proba_list = [max(prob).item() for prob in output_probs] # list of floats
    
    return acc['accuracy'], details, pred_proba_list
    

def select_real_or_pseudo_labeled_data_for_selftraining(bootstrap, \
                                                        testset_gold, \
                                                        confidence_scores, \
                                                        k, \
                                                        criteria='top', \
                                                        testset_pred='', \
                                                        train700_distrib=''):
    """
    Parameters
    ----------
    bootstrap type: 'st'
    testset_gold: Dataset, 'text' and 'label'
    confidence_scores: list of probabilities
    k: int, additionally added n of train examples
    criteria: 'top' | 'topW700distrib'
    testset_pred: if use 'st', provide predicted labels
    train700_distrib: if use 'st' and 'topW700distrib', provide train700 label distribution
    """
    def get_st_train700_distrib_pseudo_data(sorted_scores, bootstrap='st', criteria='topW700distrib'):
        assert train700_distrib != '', f"When using self training and 'topW700distrib' selection criteria, should proivde train700 label distribution."
        
        pseudo_label_nb = defaultdict(int)
        for lab, v in train700_distrib.items():
            pseudo_label_nb[lab] = round(v * k)
        text, plabel, glabel, fs, edu1, edu2, = [], [], [], [], [], []
        # select pseudo/gold examples
        for i, (_, t, gl, l, f, e1, e2) in enumerate(sorted_scores):
            if plabel.count(l) < pseudo_label_nb[l]:
                text.append(t)
                plabel.append(l)
                glabel.append(gl)
                fs.append(f)
                edu1.append(e1)
                edu2.append(e2)
        # end of loop, check final distribution
        min_score = min(sorted_scores[i][0], sorted_scores[0][0])
        max_score = max(sorted_scores[i][0], sorted_scores[0][0])
        if bootstrap == 'st':
            data = {'text': text, 'label': plabel, 'file': fs, 'edu1_idx': edu1, 'edu2_idx': edu2}
        else:
            raise NotImplementedError
        return data, min_score, max_score
    
    if bootstrap == 'st': 
        pred_label = list(map(lambda pred: id2label[pred], testset_pred))
        sorted_scores = zip(confidence_scores, \
                            testset_gold['text'], \
                            testset_gold['label'], \
                            pred_label, \
                            testset_gold['file'], \
                            testset_gold['edu1_idx'], \
                            testset_gold['edu2_idx'],\
                            )
        sorted_scores = sorted(sorted_scores, key=lambda tpl: -tpl[0])
        if k < len(confidence_scores) and criteria == 'top':
            selected = sorted_scores[:k]
            max_score = selected[0][0]
            min_score = selected[-1][0]
            _, t, gl, l, f, e1, e2 = zip(*selected)
            pseudo_data = {'text': list(t), 'label': list(l), 'file': list(f), 'edu1_idx': list(e1), 'edu2_idx': list(e2)}
            return Dataset.from_dict(pseudo_data), max_score, min_score
        elif criteria == 'topW700distrib': #use pseudo-label that correspond to train700 label distribution
            pseudo_data, min_score, max_score = get_st_train700_distrib_pseudo_data(sorted_scores, bootstrap=bootstrap)
            print(len(pseudo_data['text']))
            return Dataset.from_dict(pseudo_data), max_score, min_score
        else:
            raise NotImplementedError
    
    else:
        raise NotImplementedError


if __name__=="__main__":
    
    dataset = 'stac' # stac | molweni 

    dt = datetime.datetime.today()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load full train
    trainf = "DATA_PATH/stac_train_spk.csv"
    full_train = load_dataset('csv', data_files=trainf, delimiter='\t')['train']

    # load dev
    devf = "DATA_PATH/stac_dev_spk.csv"
    data_dev = load_dataset('csv', data_files=devf, delimiter='\t')['train']
    base_dev = Dataset.from_list([i for i in data_dev if i['file'] in stratified_train_dev['dev']['dev']])

    # load test
    testf = "DATA_PATH/stac_test_spk.csv"
    data_test = load_dataset('csv', data_files=testf, delimiter='\t')['train']

    # TODO: change to your own model path
    my_model_path = "MODEL_PATH"
    # TODO: change to your own finetuned model checkpoint
    train2checkpoint = {
        'train700_27': 'checkpoint-1400', 'train700_30': 'checkpoint-1400', 'train700_42': 'checkpoint-1200', 'train700_55': 'checkpoint-1200', 'train700_78': 'checkpoint-1400',\
        'train1500_27': 'checkpoint-3200', 'train1500_30': 'checkpoint-2800', 'train1500_42': 'checkpoint-3200', 'train1500_55': 'checkpoint-3200', 'train1500_78': 'checkpoint-3200',\
        'train2500_27': 'checkpoint-2500', 'train2500_30': 'checkpoint-4000', 'train2500_42': 'checkpoint-3000', 'train2500_55': 'checkpoint-2500', 'train2500_78': 'checkpoint-4500',\
        'train5000_27': 'checkpoint-8500', 'train5000_30': 'checkpoint-6500', 'train5000_42': 'checkpoint-6500', 'train5000_55': 'checkpoint-8000', 'train5000_78': 'checkpoint-11000',\
        'train7500_27': 'checkpoint-12000', 'train7500_30': 'checkpoint-13000', 'train7500_42': 'checkpoint-9500', 'train7500_55': 'checkpoint-4500', 'train7500_78': 'checkpoint-6000',\
        } 
    
    ################## 
    # Parameters
    ##################
    loop = 1
    subset_train = 700 # number of trained subset, starting point for self-training
    test_record = []
    bootstrap = 'st' # st=self training
    ################## 
    # /parameters
    ################## 

    if bootstrap == 'st':
        selection_criteria = ['top', 'topW700distrib']
    else:
        raise NotImplementedError

    for k in [800]: # [800, 1800, 2800, 3800, 4800, 5800]
        for criteria in ['top']: # ['top', 'topW700distrib']
            for seed in [27, 30, 42, 55, 78]:
                ################## 
                # 1. Select subset
                ################## 
                checkpoint = train2checkpoint[f'train{subset_train}_{seed}']
                train_dir = os.path.join(my_model_path, f"bert_base_stac_train{subset_train}_seed{seed}")
                model_dir = os.path.join(my_model_path, f"bert_base_stac_train{subset_train}_seed{seed}/{checkpoint}")
                trainlog = os.path.join(train_dir, 'traininglog')
                
                with open(trainlog, 'rb') as inf:
                    info = pickle.load(inf)
                
                remain_candidate = full_train.filter(lambda exp: exp['text'] not in info['text']) #filter out 700 already used as train

                # predict on remain candidate
                acc, details, pred_proba_list = predict(remain_candidate, model_dir=model_dir, device=device)
                
                # select from pseudo-labeled candidate
                # get train700 distribution, only for top-class-k selection 700distrib
                train700_distrib = get_train700_distrib(seed=seed) if criteria == 'topW700distrib' else ''
                # for k in [800, 1800, 2800, 3800, 4800, 5800]:
                selected_data, max_score, min_score = select_real_or_pseudo_labeled_data_for_selftraining(\
                                                                            bootstrap=bootstrap,\
                                                                            testset_gold=remain_candidate, \
                                                                            confidence_scores=pred_proba_list, \
                                                                            k=k, \
                                                                            criteria=criteria,\
                                                                            testset_pred=details['prediction'],
                                                                            train700_distrib=train700_distrib)
                # combine with original subset_train
                combined_train = concatenate_datasets([info, selected_data])

                # write combined data into csv
                combine_train_csv = f"DATA_PATH/combined/{bootstrap}_train{subset_train}_seed{seed}+{criteria}{k}.csv"
                combined_train.to_csv(combine_train_csv, sep='\t', index=False)
                print(f"{bootstrap}_train{subset_train}+{criteria}{k}_seed{seed}: range: [{min_score, max_score}]")

                ################## 
                # / Select subset
                ################## 

                ################## 
                # 2. Fine-tune bert with combined data
                ################## 
                # for k in [800, 1800, 2800, 3800, 4800, 5800]:
                tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
                if loop == 1 and subset_train == 700:
                    combine_train_csv = f"DATA_PATH/combined/{bootstrap}_train{subset_train}_seed{seed}+{criteria}{k}.csv"
                else:
                    combine_train_csv = f"DATA_PATH/combined/{bootstrap}_train{subset_train}_seed{seed}+{criteria}{k}_loop{loop}.csv"
                combined_train = load_dataset('csv', data_files=combine_train_csv, delimiter='\t')['train'] #reload from csv
                tokenized_train = combined_train.map(preprocess_function, batched=True)
                tokenized_dev = base_dev.map(preprocess_function, batched=True)

                model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", \
                                                                        num_labels=16, \
                                                                        id2label=id2label, \
                                                                        label2id=label2id)
                model_dir = os.path.join(my_model_path, f"bert_base_{bootstrap}_train{subset_train}_seed{seed}+{criteria}{k}_loop{loop}")
                tb_file = os.path.join(my_model_path, f"runs/{dt.strftime('%b')}{dt.day}_{dt.hour}h{dt.minute}_{bootstrap}_train{subset_train}_seed{seed}+{criteria}{k}_loop{loop}")

                LR = 2e-5
                EPOCH = 10 #k=800: 8 epochs | 1800,2800,3800: 10 epochs | 4800, 5800, 6800, 7800: 12 epochs
                STEP = 500 #k=200, 400: 200 steps | others: 500 steps
                
                training_logs = train(model, tokenizer, tokenized_train, tokenized_dev, \
                                    out_dir=model_dir, tb_file=tb_file, \
                                    learning_rate=LR, epochs=EPOCH, steps=STEP)
                
                # record to training log
                log_dir = "PATH_TO_LOG"
                infos = {'trainset': combined_train, 'devset': base_dev, 'losslog': training_logs}
                pif = os.path.join(log_dir, f"{bootstrap}_train{subset_train}_seed{seed}+{criteria}{k}_loop{loop}.log")
                with open(pif, 'wb') as outf:
                    pickle.dump(infos, outf)
                ################## 
                # /fine-tune bert with combined data
                ################## 

                ################## 
                # 3. Prediction together
                ##################

                # TODO: change to your own model checkpoint
                retrain2checkpoint = {
                    'st_train700_seed27+top800': 1800, 'st_train700_seed30+top800': 2800, 'st_train700_seed42+top800': 2800, 'st_train700_seed55+top800': 2600, 'st_train700_seed78+top800': 2800, \
                    'st_train700_seed27+top1800': 2500, 'st_train700_seed30+top1800': 4500, 'st_train700_seed42+top1800': 2500, 'st_train700_seed55+top1800': 4000, 'st_train700_seed78+top1800': 4500, \
                    'st_train700_seed27+top2800': 4000, 'st_train700_seed30+top2800': 7000, 'st_train700_seed42+top2800': 4000, 'st_train700_seed55+top2800': 7000, 'st_train700_seed78+top2800': 7000, \
                    'st_train700_seed27+top3800': 5000, 'st_train700_seed30+top3800': 8000, 'st_train700_seed42+top3800': 4000, 'st_train700_seed55+top3800': 7500, 'st_train700_seed78+top3800': 8000, \
                    'st_train700_seed27+top4800': 10000, 'st_train700_seed30+top4800': 11500, 'st_train700_seed42+top4800': 13000, 'st_train700_seed55+top4800': 11500, 'st_train700_seed78+top4800': 7000, \
                    'st_train700_seed27+top5800': 19000, 'st_train700_seed30+top5800': 17000, 'st_train700_seed42+top5800': 10500, 'st_train700_seed55+top5800': 8500, 'st_train700_seed78+top5800': 17000, \
                    'st_train700_seed27+top6800': 21000, 'st_train700_seed30+top6800': 16000, 'st_train700_seed42+top6800': 9500, 'st_train700_seed55+top6800': 17000, 'st_train700_seed78+top6800': 4000, \
                    'st_train700_seed27+top7800': 22000, 'st_train700_seed30+top7800': 24500, 'st_train700_seed42+top7800': 14500, 'st_train700_seed55+top7800': 15500, 'st_train700_seed78+top7800': 5000, \
                    
                    'st_train700_seed27+topW700distrib800': 2500, 'st_train700_seed30+topW700distrib800': 2500, 'st_train700_seed42+topW700distrib800': 3000, 'st_train700_seed55+topW700distrib800': 3000, 'st_train700_seed78+topW700distrib800': 2500, \
                    'st_train700_seed27+topW700distrib1800': 4000, 'st_train700_seed30+topW700distrib1800': 3000, 'st_train700_seed42+topW700distrib1800': 3000, 'st_train700_seed55+topW700distrib1800': 3000, 'st_train700_seed78+topW700distrib1800': 2500, \
                    'st_train700_seed27+topW700distrib2800': 7500, 'st_train700_seed30+topW700distrib2800': 6500, 'st_train700_seed42+topW700distrib2800': 3500, 'st_train700_seed55+topW700distrib2800': 4500, 'st_train700_seed78+topW700distrib2800': 6000, \
                    'st_train700_seed27+topW700distrib3800': 10000, 'st_train700_seed30+topW700distrib3800': 7500, 'st_train700_seed42+topW700distrib3800': 8000, 'st_train700_seed55+topW700distrib3800': 2000, 'st_train700_seed78+topW700distrib3800': 3000, \
                    'st_train700_seed27+topW700distrib4800': 3500, 'st_train700_seed30+topW700distrib4800': 13500, 'st_train700_seed42+topW700distrib4800': 9000, 'st_train700_seed55+topW700distrib4800': 2500, 'st_train700_seed78+topW700distrib4800': 2500, \
                    'st_train700_seed27+topW700distrib5800': 15000, 'st_train700_seed30+topW700distrib5800': 15500, 'st_train700_seed42+topW700distrib5800': 1000, 'st_train700_seed55+topW700distrib5800': 14000, 'st_train700_seed78+topW700distrib5800': 5000, \
                    'st_train700_seed27+topW700distrib6800': 2000, 'st_train700_seed30+topW700distrib6800': 4000, 'st_train700_seed42+topW700distrib6800': 3000, 'st_train700_seed55+topW700distrib6800': 2000, 'st_train700_seed78+topW700distrib6800': 5500, \
                    'st_train700_seed27+topW700distrib7800': 1500, 'st_train700_seed30+topW700distrib7800': 5500, 'st_train700_seed42+topW700distrib7800': 3500, 'st_train700_seed55+topW700distrib7800': 3000, 'st_train700_seed78+topW700distrib7800': 3500, \

                    'st_train700_seed27+topW700distrib800_loop2': 5000, 'st_train700_seed30+topW700distrib800_loop2': 3500, 'st_train700_seed42+topW700distrib800_loop2': 2500, 'st_train700_seed55+topW700distrib800_loop2': 5000, 'st_train700_seed78+topW700distrib800_loop2': 1000, \
                    'st_train700_seed27+topW700distrib800_loop3': 5500, 'st_train700_seed30+topW700distrib800_loop3': 2000, 'st_train700_seed42+topW700distrib800_loop3': 5000, 'st_train700_seed55+topW700distrib800_loop3': 1500, 'st_train700_seed78+topW700distrib800_loop3': 3500, \
                    'st_train700_seed27+topW700distrib1800_loop2': 5500, 'st_train700_seed30+topW700distrib1800_loop2': 9500, 'st_train700_seed42+topW700distrib1800_loop2': 5000, 'st_train700_seed55+topW700distrib1800_loop2': 6500, 'st_train700_seed78+topW700distrib1800_loop2': 6500, \
                    'st_train700_seed27+topW700distrib1800_loop3': 5500, 'st_train700_seed30+topW700distrib1800_loop3': 1500, 'st_train700_seed42+topW700distrib1800_loop3': 14000, 'st_train700_seed55+topW700distrib1800_loop3': 1500, 'st_train700_seed78+topW700distrib1800_loop3': 3500, \
                    'st_train700_seed27+topW700distrib2800_loop2': 9000, 'st_train700_seed30+topW700distrib2800_loop2': 3000, 'st_train700_seed42+topW700distrib2800_loop2': 13500, 'st_train700_seed55+topW700distrib2800_loop2': 12000, 'st_train700_seed78+topW700distrib2800_loop2': 9000, \
                    'st_train700_seed27+topW700distrib2800_loop3': 3500, 'st_train700_seed30+topW700distrib2800_loop3': 5500, 'st_train700_seed42+topW700distrib2800_loop3': 7500, 'st_train700_seed55+topW700distrib2800_loop3': 2000, 'st_train700_seed78+topW700distrib2800_loop3': 18000, \
 
                    'st_train1500_seed27+topW700distrib1800_loop1': 7500, 'st_train1500_seed30+topW700distrib1800_loop1': 1000, 'st_train1500_seed42+topW700distrib1800_loop1': 3500, 'st_train1500_seed55+topW700distrib1800_loop1': 2500, 'st_train1500_seed78+topW700distrib1800_loop1': 1500, \
                    'st_train1500_seed27+topW700distrib1800_loop2': 4000, 'st_train1500_seed30+topW700distrib1800_loop2': 3000, 'st_train1500_seed42+topW700distrib1800_loop2': 3500, 'st_train1500_seed55+topW700distrib1800_loop2': 2000, 'st_train1500_seed78+topW700distrib1800_loop2': 2000, \
                    'st_train1500_seed27+topW700distrib1800_loop3': 4000, 'st_train1500_seed30+topW700distrib1800_loop3': 2500, 'st_train1500_seed42+topW700distrib1800_loop3': 2000, 'st_train1500_seed55+topW700distrib1800_loop3': 2000, 'st_train1500_seed78+topW700distrib1800_loop3': 2000, \
                        
                    'st_train2500_seed27+topW700distrib1800_loop1': 3000, 'st_train2500_seed30+topW700distrib1800_loop1': 2500, 'st_train2500_seed42+topW700distrib1800_loop1': 3000, 'st_train2500_seed55+topW700distrib1800_loop1': 2000, 'st_train2500_seed78+topW700distrib1800_loop1': 2000, \
                    'st_train2500_seed27+topW700distrib1800_loop2': 4500, 'st_train2500_seed30+topW700distrib1800_loop2': 2500, 'st_train2500_seed42+topW700distrib1800_loop2': 6000, 'st_train2500_seed55+topW700distrib1800_loop2': 4500, 'st_train2500_seed78+topW700distrib1800_loop2': 2500, \
                    'st_train2500_seed27+topW700distrib1800_loop3': 2000, 'st_train2500_seed30+topW700distrib1800_loop3': 4000, 'st_train2500_seed42+topW700distrib1800_loop3': 4500, 'st_train2500_seed55+topW700distrib1800_loop3': 2500, 'st_train2500_seed78+topW700distrib1800_loop3': 5000, \
                        
                    'st_train5000_seed27+topW700distrib1800_loop1': 3500, 'st_train5000_seed30+topW700distrib1800_loop1': 12500, 'st_train5000_seed55+topW700distrib1800_loop1': 5000, 'st_train5000_seed78+topW700distrib1800_loop1': 4000, \
                    'st_train5000_seed27+topW700distrib1800_loop2': 5500, 'st_train5000_seed30+topW700distrib1800_loop2': 3000, 'st_train5000_seed55+topW700distrib1800_loop2': 3000, 'st_train5000_seed78+topW700distrib1800_loop2': 8500, \
                        
                    'st_train7500_seed27+topW700distrib1800_loop1': 4500, 'st_train7500_seed30+topW700distrib1800_loop1': 8000, 'st_train7500_seed42+topW700distrib1800_loop1': 8500, 'st_train7500_seed55+topW700distrib1800_loop1': 10000, 'st_train7500_seed78+topW700distrib1800_loop1': 6000, \
                    
                }

                if loop == 1 and subset_train == 700:
                    checkpoint = 'checkpoint-' + str(retrain2checkpoint[f"{bootstrap}_train{subset_train}_seed{seed}+{criteria}{k}"])
                    model_dir = os.path.join(my_model_path, f"bert_base_{bootstrap}_train{subset_train}_seed{seed}+{criteria}{k}")
                    pif = os.path.join(log_dir, f"{bootstrap}_train{subset_train}_seed{seed}+{criteria}{k}.log")
                else:
                    checkpoint = 'checkpoint-' + str(retrain2checkpoint[f"{bootstrap}_train{subset_train}_seed{seed}+{criteria}{k}_loop{loop}"])
                    model_dir = os.path.join(my_model_path, f"bert_base_{bootstrap}_train{subset_train}_seed{seed}+{criteria}{k}_loop{loop}")
                    pif = os.path.join(log_dir, f"{bootstrap}_train{subset_train}_seed{seed}+{criteria}{k}_loop{loop}.log")
                
                # final prediction
                acc, details, pred_proba_list = predict(data_test, model_dir=os.path.join(model_dir, checkpoint), device=device)
                test_record.append(acc*100)
                                
                try:
                    with open(pif, 'rb') as inf:
                        oldinfo = pickle.load(inf)
                except: 
                    oldinfo = {}
                
                oldinfo.update({'accuracy': acc*100, 'prediction': details['prediction'], 'gold': details['gold']})
                with open(pif, 'wb') as outf:
                    pickle.dump(oldinfo, outf)

                ################## 
                # /prediction
                ################## 
            print(f"{bootstrap}_train{subset_train}+{criteria}{k}_loop{loop}: avg={sum(test_record)/len(test_record):.2f}, stdev={statistics.stdev(test_record):.2f}, {test_record}")
    print('Done.')

        

