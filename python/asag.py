import re
import random
import pandas as pd
import torch
import tensorflow as tf
import numpy as np

from nlp_id.lemmatizer import Lemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import f1_score, cohen_kappa_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from nltk.corpus import stopwords


seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def qwk_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return cohen_kappa_score(labels_flat, preds_flat)

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

def evaluate(dataloader_val, device, model):

    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

def train_eval(df_final, pretrainedmodel):
    # bin nilai (continuous variable) into intervals
    df_final['nilai'] = pd.qcut(df_final['nilai'], 5, labels=False)

    # concatenate soal and jawaban
    df_final['soal-jawaban'] = df_final['soal']+df_final['jawaban']

    # preprocessing
    # lowercasing
    df_final['soal-jawaban'] = df_final['soal-jawaban'].apply(lambda x: x.lower())
    # lemmatization
    lemmatizer = Lemmatizer()
    df_final['soal-jawaban'] = df_final['soal-jawaban'].apply(lambda x: lemmatizer.lemmatize(x))
    # stopword removal
    list_stopwords = set(stopwords.words('indonesian'))
    df_final['soal-jawaban'] = df_final['soal-jawaban'].apply(lambda x: ' '.join([item for item in x.split() if item not in list_stopwords]))
    # punctuation removal
    df_final['soal-jawaban'] = df_final['soal-jawaban'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    # make sure that the training set and test set ratio is 80:20
    add = len(df_final[df_final['tipe'] == 'test']) - (round(0.2*(len(df_final[df_final['tipe'] == 'train'])+len(df_final[df_final['tipe'] == 'test']))))
    for i in df_final[df_final['tipe'] == 'test'].sample(n = add).itertuples():
        df_final.at[i.Index, 'tipe'] = 'train'

    # load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrainedmodel, ignore_mismatched_sizes=True)

    encoded_data_train = tokenizer.batch_encode_plus(
        df_final[df_final.tipe=='train']['soal-jawaban'].values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        truncation=True,
        max_length=256,
        padding='max_length',
        return_tensors='pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        df_final[df_final.tipe=='test']['soal-jawaban'].values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        truncation=True,
        max_length=256,
        padding='max_length',
        return_tensors='pt'
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df_final[df_final.tipe=='train'].nilai.values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df_final[df_final.tipe=='test'].nilai.values)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    model = BertForSequenceClassification.from_pretrained(pretrainedmodel,
                                                          num_labels=5,
                                                          output_attentions=False,
                                                          output_hidden_states=False, ignore_mismatched_sizes=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    batch_size = 15

    dataloader_train = DataLoader(dataset_train,
                                  sampler=RandomSampler(dataset_train),
                                  batch_size=batch_size)

    dataloader_validation = DataLoader(dataset_val,
                                       sampler=SequentialSampler(dataset_val),
                                       batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(),
                      lr=1e-5,
                      eps=1e-8)

    epochs = 10

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train)*epochs)

    for epoch in tqdm(range(1, epochs+1)):

        model.train()

        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:

            model.zero_grad()

            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[2],
                     }

            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})


        torch.save(model.state_dict(), f'data_volume/finetuned_BERT_epoch_{epoch}.model')

        tqdm.write(f'\nEpoch {epoch}')

        loss_train_avg = loss_train_total/len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')

        val_loss, predictions, true_vals = evaluate(dataloader_validation, device, model)
        val_f1 = f1_score_func(predictions, true_vals)
        val_qwk = qwk_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')
        tqdm.write(f'QWK Score: {val_qwk}')
