from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np

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


def create_decoder_input(response_ids, res_id, eos_id, golden=None):
    assert golden != None

    decoder_lmlabel= response_ids + [eos_id]
    decoder_input_ids = [res_id] + response_ids
    decoder_cls_index = [-100] * (len(decoder_lmlabel) - 1) + [eos_id]
    decoder_attention_mask = [1] * len(decoder_input_ids)

    if golden == False:
        decoder_lmlabel = [-100] * len(decoder_lmlabel)

    assert len(decoder_lmlabel) == len(decoder_input_ids)
    return decoder_lmlabel, decoder_input_ids, decoder_cls_index, decoder_attention_mask


def pad_dataset(dataset, pad_id):
    for item_name, item in dataset.items():
        if item_name == "input_ids" or item_name == "per_input_ids":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                              batch_first=True, padding_value=pad_id)

            dataset[item_name] = item
        elif item_name == "lmlabels":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=-100)
            dataset[item_name] = item
        elif item_name == "attention_mask" or item_name == "decoder_attention_mask" or item_name == "per_attention_mask":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=0)
            dataset[item_name] = item
        elif item_name == "decoder_input_ids":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item
        elif item_name == "clslabel":
            dataset[item_name] = torch.tensor(item).view(-1,1)
        elif item_name == "cls_index":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=-100)
            dataset[item_name] = item


    return dataset