import sys
import logging
import os

import argparse
from transformers import BartTokenizer
from model.modeling_bart import LMEDRModel
import torch
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict
from pprint import pformat
from build_dstc import build_test


def init_config():
    parser = argparse.ArgumentParser(description='Dialog_BART')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--beam', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')
    parser.add_argument("--load_from", type=str, default=None, help="save model")
    parser.add_argument("--num_latent", type=int, default=20, help="number of latent")
    parser.add_argument("--num_latent2", type=int, default=5, help="number of latent2")


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = "cuda"
    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args = argparse.Namespace(**vars(args))
    return args


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    add_special_tokens = {'additional_special_tokens': ['<query>', '<response>', '<latent>', '<persona>']}
    args = init_config()
    assert args.load_from != None
    log_file = os.path.join(args.load_from, "generate.log")
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    format_str = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')
    logger.setLevel(level=logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    fh = logging.FileHandler(filename=log_file, encoding='utf-8', mode='w')
    fh.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.info(r"running %s" % ''.join(sys.argv))
    logger.info("Arguments: %s", pformat(args))

    logger.info("Get pretrained model and tokenizer")
    tokenizer = BartTokenizer.from_pretrained(args.load_from)
    logger.info('We have {} tokens'.format(len(tokenizer)))

    model = LMEDRModel.from_pretrained(args.load_from, num_labels=1,
                                       num_token=len(tokenizer),
                                       num_latent=args.num_latent,
                                       num_latent2=args.num_latent2)

    model.to(args.device)

    logger.info("Complete loading model.")

    logger.info("Begin generating")
    MODEL_INPUTS = ["input_ids", "attention_mask",
                    "per_input_ids", "per_attention_mask"]
    ans = defaultdict(list)
    with open("data/DSTC/test_set4DSTC7-AVSD.json", "r", encoding="utf8") as f:
        data = json.load(f)
    dialogs = data["dialogs"]

    for obj in tqdm(dialogs):
        tmp_dict = dict()
        dialog = obj["dialog"]
        last_dialog = len(dialog) - 1
        image_id = obj["image_id"]
        tmp_dict["image_id"] = image_id
        summary = obj["summary"].strip(".").lower().split(".")
        caption = obj["caption"].strip(".").lower().split(".")
        tmp_persona = summary + caption
        history = []
        inputs = build_test(tmp_persona, dialog, tokenizer)
        tensor_input = {}
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(inputs[input_name], device=args.device)
            tensor_input[input_name] = tensor
        model.eval()
        with torch.no_grad():
            out_ids = model.generate(input_ids=tensor_input["input_ids"],
                                    attention_mask=tensor_input["attention_mask"],
                                    per_input_ids=tensor_input["per_input_ids"],
                                    per_attention_mask=tensor_input["per_attention_mask"],
                                    max_length=args.max_length, num_beams=args.beam)
            out_text = tokenizer.batch_decode(out_ids, skip_special_tokens=True,
                                                   spaces_between_special_tokens=False,
                                                   clean_up_tokenization_spaces=False)

        ans_ = out_text[0].strip()
        dialog[last_dialog]["answer"] = ans_
        tmp_dict["dialog"] = dialog
        ans["dialogs"].append(tmp_dict)

    with open(os.path.join(args.load_from, f"dstc.json"), "w", encoding='utf-8') as f:
        json.dump(ans, f, indent=2)










