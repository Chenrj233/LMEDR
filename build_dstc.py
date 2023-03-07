from tqdm import trange, tqdm
import torch
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json
import jsonlines


def get_token_id(tokenizer):
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    sep_id = tokenizer.sep_token_id
    query_id, res_id, latent_id, persona_id = tokenizer.convert_tokens_to_ids(
        ['<query>', '<response>', '<latent>', '<persona>'])
    return bos_id, eos_id, pad_id, sep_id, query_id, res_id, latent_id, persona_id

def create_data(data_file):
    with open(data_file, "r", encoding="utf8") as f:
        data = json.load(f)
    dialogs = data["dialogs"]
    persona = []
    query = []
    response = []
    sum_u = 0
    cnt = 0
    for obj in tqdm(dialogs):
        cnt += 1
        tmp_query = []
        tmp_response = []
        dialog = obj["dialog"]
        summary = obj["summary"].strip(".").lower().split(".")
        caption = obj["caption"].strip(".").lower().split(".")
        tmp_persona = summary + caption

        persona.append(tmp_persona)
        for i in range(len(dialog)):
            tmp_query.append(dialog[i]["question"].lower())
            tmp_response.append(dialog[i]["answer"].lower())
            sum_u += 1
        query.append(tmp_query)
        response.append(tmp_response)

    assert len(query) == len(response) == len(persona)

    print("{} has {} dialog and {} query".format(data_file, len(query), sum_u))

    return persona, query, response


def create_encoder_input(per, history, query_id, res_id, latent_id, persona_id, sep_id, eos_id):
    encoder_input_ids = []
    per_input_ids = [latent_id] + [persona_id]

    for x in per:
        per_input_ids += x + [sep_id]

    encoder_input_ids += per_input_ids
    for i in range(len(history)):
        if i % 2 == 0:
            encoder_input_ids += [query_id] + history[i] + [eos_id]
        else:
            encoder_input_ids += [res_id] + history[i] + [eos_id]
    attention_mask = [1] * len(encoder_input_ids)
    per_attention_mask = [1] * len(per_input_ids)
    return encoder_input_ids, attention_mask, per_input_ids, per_attention_mask

def create_decoder_input(response_ids, res_id, eos_id):


    decoder_lmlabel= response_ids + [eos_id]
    decoder_input_ids = [res_id] + response_ids
    decoder_attention_mask = [1] * len(decoder_input_ids)

    assert len(decoder_lmlabel) == len(decoder_input_ids)
    return decoder_lmlabel, decoder_input_ids, decoder_attention_mask



def build_dataloader(persona, query, response, tokenizer, max_query=4):
    bos_id, eos_id, pad_id, sep_id, query_id, res_id, latent_id, persona_id = get_token_id(tokenizer)
    dataset = defaultdict(list)
    for i in trange(len(persona)):
        persona_ = persona[i]
        per_list = []
        for per in persona_:

            persona_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(per, add_prefix_space=True))
            per_list.append(persona_ids)

        query_ = query[i]
        response_ = response[i]
        history = []
        assert len(query_) == len(response_)
        for j in range(len(query_)):

            query_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query_[j], add_prefix_space=True))
            response_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response_[j], add_prefix_space=True))

            history.append(query_ids)
            history.append(response_ids)
            tmp_history = history[-2 * max_query: -1]


            encoder_input_ids, attention_mask, \
            per_input_ids, per_attention_mask = create_encoder_input(per_list, tmp_history, query_id,
                                                                    res_id,latent_id, persona_id, sep_id, eos_id)
            decoder_lmlabel, decoder_input_ids,\
                decoder_attention_mask = create_decoder_input(response_ids, res_id, eos_id)

            dataset["input_ids"].append(encoder_input_ids)
            dataset["attention_mask"].append(attention_mask)
            dataset["per_input_ids"].append(per_input_ids)
            dataset["per_attention_mask"].append(per_attention_mask)
            dataset["lmlabels"].append(decoder_lmlabel)
            dataset["decoder_input_ids"].append(decoder_input_ids)
            dataset["decoder_attention_mask"].append(decoder_attention_mask)



    for item_name, item in dataset.items():
        if item_name == "input_ids" or item_name == "per_input_ids":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                              batch_first=True, padding_value=pad_id)

            dataset[item_name] = item
            print(item_name, dataset[item_name].size())
        elif item_name == "lmlabels":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=-100)
            dataset[item_name] = item
            print(item_name, dataset[item_name].size())
        elif item_name == "attention_mask" or item_name == "decoder_attention_mask" or item_name == "per_attention_mask":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=0)
            dataset[item_name] = item
            print(item_name, dataset[item_name].size())
        elif item_name == "decoder_input_ids":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item
            print(item_name, dataset[item_name].size())


    return dataset

def build_test(persona, dialog, tokenizer):
    bos_id, eos_id, pad_id, sep_id, query_id, res_id, latent_id, persona_id = get_token_id(tokenizer)
    dataset = defaultdict(list)
    per_list = []
    for per in persona:
        persona_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(per, add_prefix_space=True))
        per_list.append(persona_ids)

    history = []
    for i in range(len(dialog)):
        query_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dialog[i]["question"].lower(),
                                                                                  add_prefix_space=True))
        response_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dialog[i]["answer"].lower(),
                                                                                  add_prefix_space=True))
        history.append(query_ids)
        history.append(response_ids)

    encoder_input_ids, attention_mask, per_input_ids, per_attention_mask = create_encoder_input(per_list, history[:-1],
                                                                                                query_id, res_id, latent_id,
                                                                                                persona_id, sep_id, eos_id)

    dataset["input_ids"].append(encoder_input_ids)
    dataset["attention_mask"].append(attention_mask)
    dataset["per_input_ids"].append(per_input_ids)
    dataset["per_attention_mask"].append(per_attention_mask)

    for item_name, item in dataset.items():
        if item_name == "input_ids" or item_name == "per_input_ids":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                              batch_first=True, padding_value=pad_id)

            dataset[item_name] = item
        elif item_name == "attention_mask" or item_name == "per_attention_mask":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=0)
            dataset[item_name] = item


    return dataset



def build_nli_dataset(tokenizer, file_path):
    bos_id, eos_id, pad_id, sep_id, query_id, res_id, latent_id, persona_id = get_token_id(tokenizer)
    positive_set = defaultdict(list)
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            if obj['gold_label'] == "neutral" or obj['gold_label'] == "contradiction":
                continue
            pre = obj["sentence1"].lower()
            hyp = obj["sentence2"].lower()
            pre_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(pre, add_prefix_space=True))
            hyp_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(hyp, add_prefix_space=True))
            encoder_input_ids = [latent_id] + [persona_id] + pre_ids + [eos_id]
            attention_mask = [1] * len(encoder_input_ids)

            decoder_input_ids = [res_id] + hyp_ids
            decoder_attention_mask = [1] * len(decoder_input_ids)

            if obj['gold_label'] == 'entailment':
                positive_set["encoder_input_ids"].append(encoder_input_ids)
                positive_set["decoder_input_ids"].append(decoder_input_ids)
                positive_set["attention_mask"].append(attention_mask)
                positive_set["decoder_attention_mask"].append(decoder_attention_mask)
                decoder_lmlabel = hyp_ids + [eos_id]
                positive_set["lmlabels"].append(decoder_lmlabel)

        for item_name, item in positive_set.items():
            if item_name == "encoder_input_ids":
                item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                    batch_first=True, padding_value=pad_id)

                positive_set[item_name] = item
            elif item_name == "lmlabels":
                item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                    batch_first=True, padding_value=-100)
                positive_set[item_name] = item
            elif item_name == "attention_mask" or item_name == "decoder_attention_mask":
                item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                    batch_first=True, padding_value=0)
                positive_set[item_name] = item
            elif item_name == "decoder_input_ids":
                item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                    batch_first=True, padding_value=pad_id)
                positive_set[item_name] = item


    return positive_set






