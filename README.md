# LMEDR <img src="https://pytorch.org/assets/images/logo-dark.svg" width = "90" />
Code for AAAI 2023 paper: [Learning to Memorize Entailment and Discourse Relations for Persona-Consistent Dialogues](https://arxiv.org/pdf/2301.04871.pdf).

## Requirements

Check the package requirements

- python==3.8
- torch==1.9.1
- transformers==4.14.1
- pytorch-ignite==0.4.9

Please install ParlAI, which can be done in the following ways
```bash
git clone https://github.com/Chenrj233/ParlAI.git
cd ParlAI
python setup.py install
```

Please replace `eval_f1.py` and `eval_hits.py` in `/ParlAI/projects/convai2/` with the corresponding files in `/other/`. Similarly, replace the `generation_utils.py` in `transformers/` with the corresponding files in `/other/`, the file is in a path similar to
```
| -- python3.8
	| -- site-packages
		| -- transformers
			| -- modeling_utils.py
			| -- generation_utils.py
			| -- ...
```

## Data

The datasets used in the paper can be obtained from the following link:

|  Dataset| Paper  |
|  ----   |  ----  |
|  [ConvAI2 PersonaChat](http://parl.ai/downloads/convai2/convai2_fix_723.tgz) | [The Second Conversational Intelligence Challenge (ConvAI2)](https://link.springer.com/chapter/10.1007/978-3-030-29135-8_7)  |
| [DSTC7-AVSD](https://drive.google.com/open?id=1SlZTySJAk_2tiMG5F8ivxCfOl_OWwd_Q) | [Audio Visual Scene-aware dialog (AVSD) Track for Natural Language Generation in DSTC7](http://workshop.colips.org/dstc7/papers/DSTC7_Task_3_overview_paper.pdf)  |
| [MNLI](https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip) | [ A broad-coverage challenge corpus for sentence understanding through inference](https://aclanthology.org/N18-1101.pdf)  |
| [DNLI](https://wellecks.com/dialogue_nli/) | [Dialogue Natural Language Inference](https://aclanthology.org/P19-1363.pdf)  |


## Training

* **PersonaChat**

	Use the following script to train on the PersonaChat original dataset. If you want to train on the revised dataset, please add `--revised`
```
python train_PersonaChat.py --lr 8e-6 \
--epochs 20 \
--train_batch_size 2 \
--valid_batch_size 2 \
--infer_batch_size 64 
```

* **DSTC7-AVSD**

	For training on DSTC7-AVSD, it can be run as
```
python train_dstc.py --lr 8e-6 \
--epochs 20 \
--train_batch_size 2 \
--valid_batch_size 2 \
--infer_batch_size 10
```

## Evaluation

* **PersonaChat**

	Model checkpoints can be obtained from [persona_original](https://drive.google.com/drive/folders/1po__VfU9WxM8XUS4mOAJsfVoOrdv6B63?usp=share_link), [persona_revised](https://drive.google.com/drive/folders/1KdyrFHm808ZbWQGU0bcogQDQFciwmHx7?usp=share_link).

- Hits@1
```
python evaluation_PersonaChat.py --model_checkpoint persona_original \
--eval_type hits@1
```
- F1
```
python evaluation_PersonaChat.py --model_checkpoint persona_original \
--eval_type f1 \
--beam 2 \
--max_history 7
```
- PPL
```
python train_PersonaChat.py --load_from persona_original \
--eval
```
- C.Score

	Please refer to [PAML](https://github.com/HLTCHKUST/PAML).

* **DSTC7-AVSD**

	First, we use `dstc_generate.py` to generate the predicted response, and then use [dstc7avsd_eval](https://github.com/hudaAlamri/DSTC7-Audio-Visual-Scene-Aware-Dialog-AVSD-Challenge) to evaluate，model checkpoint can be obtained from [dstc_model](https://drive.google.com/drive/folders/1gj5qLseAeYFSBnCSaNrxe0tMX42UhW3M?usp=share_link).
```
python dstc_generate.py --load_from dstc_model \
--beam 5
```

## Results

We also provide the final generated texts, which can be found in `/results/`.