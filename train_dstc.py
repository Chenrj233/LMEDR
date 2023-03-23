import sys
import logging
import os
import argparse
from transformers import BartTokenizer, AdamW, WEIGHTS_NAME, CONFIG_NAME
from model.modeling_bart import LMEDRModel
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel
import time
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
import math
from pprint import pformat
from build_dstc import create_data, build_dataloader, build_nli_dataset


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()



def init_config():
    parser = argparse.ArgumentParser(description='Dialog_BART')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--infer_batch_size", type=int, default=10, help="Batch size for infer")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--warmup", type=float, default=0.2, help="warmup rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--num_latent", type=int, default=20, help="number of latent")
    parser.add_argument("--num_latent2", type=int, default=5, help="number of latent2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--output_dir", type=str, default="dstc_model",
                        help="save model")
    parser.add_argument("--load_from", type=str, default=None,
                        help="save model")
    parser.add_argument("--fp16", type=str, default="",
                        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Accumulate gradients on several steps")
    parser.add_argument('--max_query', type=int, default=6)


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    elif args.cuda:
        torch.cuda.set_device(args.gpu)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    args = argparse.Namespace(**vars(args))
    return args





if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    add_special_tokens = {'additional_special_tokens': ['<query>', '<response>', '<latent>', '<persona>']}
    args = init_config()


    if not os.path.exists(os.path.join(args.output_dir)):
        os.makedirs(os.path.join(args.output_dir))
    log_file = os.path.join(args.output_dir, "train.log")


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

    if args.load_from != None:
        tokenizer = BartTokenizer.from_pretrained(args.load_from)
        model = LMEDRModel.from_pretrained(args.load_from)
    else:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        num_added_toks = tokenizer.add_special_tokens(add_special_tokens)
        logger.info('We have added {} tokens'.format(num_added_toks))
        model = LMEDRModel.from_pretrained("facebook/bart-large", num_labels=1,
                                           num_token=len(tokenizer),
                                           num_latent=args.num_latent, num_latent2=args.num_latent2)

        model.resize_token_embeddings(len(tokenizer))

        model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids('<response>')
        model.config.forced_bos_token_id = None

    logger.info('We have {} tokens'.format(len(tokenizer)))
    model.to(args.device)

    logger.info("Complete loading model.")

    logger.info("Build train data")
    persona, query, response = create_data(f"data/DSTC/train_set4DSTC7-AVSD.json")
    train_data = build_dataloader(persona, query, response, tokenizer, max_query=args.max_query)
    logger.info("Build valid data")
    persona, query, response = create_data(f"data/DSTC/valid_set4DSTC7-AVSD.json")
    val_data = build_dataloader(persona, query, response, tokenizer, max_query=args.max_query)
    logger.info("Build infer data")
    infer_data = build_nli_dataset(tokenizer, "data/mnli/multinli_1.0_train.jsonl")
    logger.info("Build test data")

    MODEL_INPUTS = ["input_ids", "attention_mask", "lmlabels", "decoder_input_ids", "decoder_attention_mask",
                "per_input_ids", "per_attention_mask"]
    INFER_INPUTS = ["encoder_input_ids", "decoder_input_ids", "attention_mask", "decoder_attention_mask",
                    "lmlabels"]
    trainset = []
    valset = []
    inferset = []
    for input_name in MODEL_INPUTS:
        tensor = train_data[input_name]
        trainset.append(tensor)
        logger.info("{}: {}".format(input_name, tensor.size()))
        tensor = val_data[input_name]
        valset.append(tensor)
        logger.info("{}: {}".format(input_name, tensor.size()))

    for input_name in INFER_INPUTS:
        tensor = infer_data[input_name]
        logger.info("{}: {}".format(input_name, tensor.size()))
        inferset.append(tensor)

    train_dataset = TensorDataset(*trainset)
    val_dataset = TensorDataset(*valset)
    infer_dataset = TensorDataset(*inferset)
    logger.info("Prepare dataloader.")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    infer_sampler = torch.utils.data.distributed.DistributedSampler(infer_dataset) if args.distributed else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                              shuffle=(not args.distributed), num_workers=0)
    infer_loader = DataLoader(infer_dataset, sampler=infer_sampler, batch_size=args.infer_batch_size,
                              shuffle=(not args.distributed), num_workers=0)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.valid_batch_size, shuffle=False)


    train_iter = len(train_loader)


    memory1_params = list(map(id, model.memory1))
    base_params = filter(lambda p: id(p) not in memory1_params,
                         model.parameters())


    optimizer_infer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    optimizer_bart = AdamW(base_params, lr=args.lr, correct_bias=True)


    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)


    def infer_update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        infer_input_ids, infer_decoder_input_ids, infer_attention_mask, \
        infer_decoder_attention_mask, infer_lmlabels = batch
        outputs = model(infer_input_ids=infer_input_ids,
                        infer_decoder_input_ids=infer_decoder_input_ids,
                        infer_attention_mask=infer_attention_mask,
                        infer_lmlabels=infer_lmlabels,
                        infer_decoder_attention_mask=infer_decoder_attention_mask
                        )
        loss = outputs.loss
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer_infer.step()
        optimizer_infer.zero_grad()
        return {'loss': loss.item()}


    infer_trainer = Engine(infer_update)
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)


        input_ids, attention_mask, lmlabels, decoder_input_ids, decoder_attention_mask,  \
        per_input_ids, per_attention_mask = batch

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, lmlabels=lmlabels,
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask,
                        per_input_ids=per_input_ids,
                        per_attention_mask=per_attention_mask
                        )

        (lm_loss, __, m_loss, _, bow) = outputs.loss

        loss = lm_loss + m_loss + bow

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer_bart.step()
            model.zero_grad()

        return {'loss': loss.item(),  'lm': lm_loss.item(),
                "mem": m_loss.item(), "bow": bow.item()}


    trainer = Engine(update)


    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, attention_mask, lmlabels, decoder_input_ids, decoder_attention_mask, \
            per_input_ids, per_attention_mask = batch

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask,
                            per_input_ids=per_input_ids,
                            per_attention_mask=per_attention_mask
                            )
            (lm_logits, cls_logits) = outputs.logits

            tmp_lmlogits = lm_logits.view(-1, lm_logits.size(-1))

            lmlabels = lmlabels.view(-1)

            return (tmp_lmlogits, ), (lmlabels, )


    evaluator = Engine(inference)

    infer_trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: trainer.run(train_loader))

    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))

    if args.eval_before_start:
        infer_trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        infer_trainer.add_event_handler(Events.EPOCH_STARTED,
                                        lambda engine: infer_sampler.set_epoch(engine.state.epoch))
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: val_sampler.set_epoch(engine.state.epoch))



    scheduler_infer = PiecewiseLinear(optimizer_infer, "lr", [(0, args.lr), (args.epochs * len(infer_loader), 0.0)])
    scheduler_bart = PiecewiseLinear(optimizer_bart, "lr", [(0, args.lr), (args.epochs * len(train_loader), 0.0)])


    infer_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler_infer)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler_bart)

    RunningAverage(output_transform=lambda x: x["loss"]).attach(infer_trainer, "loss")
    RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "loss")
    RunningAverage(output_transform=lambda x: x["lm"]).attach(trainer, "lm")
    RunningAverage(output_transform=lambda x: x["mem"]).attach(trainer, "mem")
    RunningAverage(output_transform=lambda x: x["bow"]).attach(trainer, "bow")


    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0][0], x[1][0]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar_infer = ProgressBar(persist=True, ncols=140)
        pbar = ProgressBar(position=0, persist=True, ncols=140)

        pbar_infer.attach(infer_trainer, metric_names=["loss"])

        pbar.attach(trainer, metric_names=["loss",  "lm", "mem", "bow"])

        infer_trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                        lambda engine: logger.info(f"Complete infer epoch: {engine.state.epoch}"))
        trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                  lambda engine: logger.info(f"Complete trainer epoch: {engine.state.epoch}"))
        evaluator.add_event_handler(Events.COMPLETED,
                                    lambda __: logger.info("Validation: %s" % pformat(evaluator.state.metrics)))

        log_dir = os.path.join(args.output_dir)
        tb_logger = TensorboardLogger(log_dir)
        tb_logger.attach(infer_trainer, log_handler=OutputHandler(tag="infer", metric_names=["loss"]),
                         event_name=Events.ITERATION_COMPLETED)

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training",
                                                            metric_names=["loss", "lm", "mem", "bow"]),
                         event_name=Events.ITERATION_COMPLETED)

        tb_logger.attach(infer_trainer, log_handler=OptimizerParamsHandler(optimizer_infer),
                         event_name=Events.ITERATION_STARTED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer_bart),
                         event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()),
                                                              global_step_transform=global_step_from_engine(trainer)),
                         event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', n_saved=None)
        infer_trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation


        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)

    logger.info("Begin training")
    # Run the training
    infer_trainer.run(infer_loader, max_epochs=args.epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.epochs > 0:
        os.rename(os.path.join(log_dir, checkpoint_handler._saved[-1][1]),
                  os.path.join(log_dir,
                               WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()


