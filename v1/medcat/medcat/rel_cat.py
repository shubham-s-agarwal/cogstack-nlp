import json
import logging
import os
import random
import numpy
import torch.optim
import torch
import torch.nn as nn
import spacy
import time  # add to your imports (replaces the datetime timing)
from tqdm import tqdm
from datetime import date, datetime
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import compute_class_weight
from medcat.cdb import CDB
from medcat.config import Config
from medcat.config_rel_cat import ConfigRelCAT
from medcat.pipeline.pipe_runner import PipeRunner
from spacy.tokens import Doc, Span
from typing import Dict, Iterable, Iterator, List
from torch.utils.data import DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from medcat.utils.relation_extraction.base_component import BaseComponent_RelationExtraction
from medcat.utils.meta_cat.ml_utils import set_all_seeds
from medcat.utils.relation_extraction.ml_utils import load_results, load_state, save_results, save_state, split_list_train_test_by_class
from medcat.utils.relation_extraction.rel_dataset import RelData


def resolve_device(requested: str) -> torch.device:
    """Pick the best available device, honouring an explicit request where possible.

    Supports cpu, cuda (incl. 'cuda:0') and mps (Apple Silicon). If the requested
    accelerator isn't available, falls back to whatever is.
    """
    requested = (requested or "cpu").lower()
    if requested.startswith("cpu"):
        return torch.device("cpu")
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested.startswith("mps") and mps_ok:
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_ok:
        return torch.device("mps")
    return torch.device("cpu")


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, classes, batch_size, max_samples, max_minority):
        self.dataset = dataset
        self.classes = classes
        self.batch_size = batch_size
        self.num_classes = len(classes)
        self.indices = list(range(len(dataset)))

        self.max_minority = max_minority

        self.max_samples_per_class = max_samples

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batch_counter = 0
        indices = self.indices.copy()
        while batch_counter != self.__len__():
            batch = []

            class_counts = {c: 0 for c in self.classes}
            while len(batch) < self.batch_size:

                index = random.choice(indices)
                label = self.dataset[index][2].numpy().tolist()[0]
                if class_counts[label] < self.max_samples_per_class[label]:
                    batch.append(index)
                    class_counts[label] += 1
                    if self.max_samples_per_class[label] > self.max_minority:
                        indices.remove(index)

            yield batch
            batch_counter += 1


class RelCAT(PipeRunner):
    """The RelCAT class used for training 'Relation-Annotation' models, i.e., annotation of relations
     between clinical concepts.

    Args:
        cdb (CDB): cdb, this is used when creating relation datasets.

        tokenizer (TokenizerWrapperBERT):
            The Huggingface tokenizer instance. This can be a pre-trained tokenzier instance from
            a BERT-style model. For now, only BERT models are supported.

        config (ConfigRelCAT):
            the configuration for RelCAT. Param descriptions available in ConfigRelCAT class docs.

        task (str, optional): What task is this model supposed to handle. Defaults to "train"
        init_model (bool, optional): loads default model. Defaults to False.

    """

    name = "rel_cat"

    log = logging.getLogger(__name__)

    def __init__(self, cdb: CDB, config: ConfigRelCAT = ConfigRelCAT(), task="train", init_model=False):

        self.component: BaseComponent_RelationExtraction = BaseComponent_RelationExtraction()  # type: ignore
        self.task: str = task
        self.checkpoint_path: str = "./"

        set_all_seeds(config.general.seed)

        if init_model:
            self.component = BaseComponent_RelationExtraction(config=config,
                                                              task=task,
                                                              init_model=True)

        self.cdb = cdb
        logging.basicConfig(level=self.component.relcat_config.general.log_level)
        self.log.setLevel(self.component.relcat_config.general.log_level)

        self.is_cuda_available = torch.cuda.is_available()
        self.device = resolve_device(self.component.relcat_config.general.device)

    def _move_model_to_device(self) -> None:
        """Moves the HF model AND every other submodule (fc1/fc2/fc3, dropout, ...) to self.device."""
        self.component.model = self.component.model.to(self.device)  # type: ignore

    def save(self, save_path: str = "./") -> None:
        self.component.save(save_path=save_path)

    @classmethod
    def load(cls, load_path: str = "./") -> "RelCAT":

        cdb = CDB(config=Config())
        if os.path.exists(os.path.join(load_path, "cdb.dat")):
            cdb = CDB.load(os.path.join(load_path, "cdb.dat"))
        else:
            cls.log.info("The default CDB file name 'cdb.dat' doesn't exist in the specified path, you will need to load & set \
                a CDB manually via rel_cat.cdb = CDB.load('path') ")

        component = BaseComponent_RelationExtraction.load(pretrained_model_name_or_path=load_path)

        device = resolve_device(component.relcat_config.general.device)

        rel_cat = RelCAT(cdb=cdb, config=component.relcat_config, task=component.task)
        rel_cat.device = device
        rel_cat.component = component

        return rel_cat

    def __call__(self, doc: Doc) -> Doc:
        doc = next(self.pipe(iter([doc])))
        return doc

    def _create_test_train_datasets(self, data: Dict, split_sets: bool = False):
        train_data: Dict = {}
        test_data: Dict = {}

        if split_sets:
            train_data["output_relations"], test_data["output_relations"] = split_list_train_test_by_class(
                data["output_relations"],
                test_size=self.component.relcat_config.train.test_size,
                shuffle=self.component.relcat_config.train.shuffle_data,
                sample_limit=self.component.relcat_config.general.limit_samples_per_class)

            test_data_label_names = [rec[4] for rec in test_data["output_relations"]]

            test_data["nclasses"], test_data["labels2idx"], test_data["idx2label"] = RelData.get_labels(
                test_data_label_names, self.component.relcat_config)

            for idx in range(len(test_data["output_relations"])):
                test_data["output_relations"][idx][5] = test_data["labels2idx"][test_data["output_relations"][idx][4]]
        else:
            train_data["output_relations"] = data["output_relations"]

        for k, v in data.items():
            if k != "output_relations":
                train_data[k] = []
                test_data[k] = []

        train_data_label_names = [rec[4]
                                  for rec in train_data["output_relations"]]

        train_data["nclasses"], train_data["labels2idx"], train_data["idx2label"] = RelData.get_labels(
            train_data_label_names, self.component.relcat_config)

        for idx in range(len(train_data["output_relations"])):
            train_data["output_relations"][idx][5] = train_data["labels2idx"][train_data["output_relations"][idx][4]]

        return train_data, test_data

    def train(self, export_data_path: str = "", train_csv_path: str = "", test_csv_path: str = "",
              checkpoint_path: str = "./"):

        if self.is_cuda_available:
            self.log.info("Training on device:" +
                          str(torch.cuda.get_device_name(0)) + str(self.device))
        else:
            self.log.info("Training on device: " + str(self.device))

        self._move_model_to_device()

        # resize vocab just in case more tokens have been added
        self.component.model_config.vocab_size = self.component.tokenizer.get_size()

        cfg = self.component.relcat_config

        train_rel_data = RelData(cdb=self.cdb, config=cfg, tokenizer=self.component.tokenizer)
        test_rel_data = RelData(cdb=self.cdb, config=cfg, tokenizer=self.component.tokenizer)

        # ------------------------------------------------------------------ data
        if train_csv_path != "":
            if test_csv_path != "":
                train_rel_data.dataset, _ = self._create_test_train_datasets(
                    train_rel_data.create_base_relations_from_csv(train_csv_path), split_sets=False)
                test_rel_data.dataset, _ = self._create_test_train_datasets(
                    train_rel_data.create_base_relations_from_csv(test_csv_path), split_sets=False)
            else:
                train_rel_data.dataset, test_rel_data.dataset = self._create_test_train_datasets(
                    train_rel_data.create_base_relations_from_csv(train_csv_path), split_sets=True)
        elif export_data_path != "":
            with open(export_data_path) as f:
                export_data = json.load(f)
            train_rel_data.dataset, test_rel_data.dataset = self._create_test_train_datasets(
                train_rel_data.create_relations_from_export(export_data), split_sets=True)
        else:
            raise ValueError("NO DATA HAS BEEN PROVIDED (MedCAT Trainer export JSON/CSV/spacy_DOCS)")

        # ------------------------------------- class count / label maps (before
        # the sampler, criterion and training loop use them)
        if train_rel_data.dataset["nclasses"] > cfg.train.nclasses:
            cfg.train.nclasses = train_rel_data.dataset["nclasses"]
            self.component.model.relcat_config.train.nclasses = cfg.train.nclasses

        cfg.general.labels2idx.update(train_rel_data.dataset["labels2idx"])
        cfg.general.idx2labels = {int(v): k for k, v in cfg.general["labels2idx"].items()}
        print("Labels2idx:", cfg.general.idx2labels)

        nclasses = cfg.train.nclasses

        # ----------------------------------------------------------- dataloaders
        train_dataset_size = len(train_rel_data)
        batch_size = min(train_dataset_size, cfg.train.batch_size)

        if cfg.train['stratified_batching']:
            sampler = BalancedBatchSampler(train_rel_data, [i for i in range(nclasses)],
                                           batch_size,
                                           cfg.train['batching_samples_per_class'],
                                           cfg.train['batching_minority_limit'])

            train_dataloader = DataLoader(train_rel_data, num_workers=0, collate_fn=self.component.padding_seq,
                                          batch_sampler=sampler, pin_memory=cfg.general.pin_memory)
        else:
            train_dataloader = DataLoader(train_rel_data, batch_size=batch_size,
                                          shuffle=cfg.train.shuffle_data,
                                          num_workers=0,
                                          collate_fn=self.component.padding_seq,
                                          pin_memory=cfg.general.pin_memory)

        test_dataset_size = len(test_rel_data)
        test_batch_size = min(test_dataset_size, cfg.train.batch_size)
        test_dataloader = DataLoader(test_rel_data,
                                     batch_size=test_batch_size,
                                     shuffle=cfg.train.shuffle_data,
                                     num_workers=0,
                                     collate_fn=self.component.padding_seq,
                                     pin_memory=cfg.general.pin_memory)

        # ------------------------------------------------------------- criterion
        if cfg.train.class_weights is not None and cfg.train.enable_class_weights:
            criterion = nn.CrossEntropyLoss(
                weight=torch.FloatTensor(numpy.asarray(cfg.train.class_weights)).to(self.device))
        elif cfg.train.enable_class_weights:
            all_class_lbl_ids = [rec[5] for rec in train_rel_data.dataset["output_relations"]]
            present = numpy.unique(all_class_lbl_ids)
            present_weights = compute_class_weight(class_weight="balanced",
                                                   classes=present,
                                                   y=all_class_lbl_ids)
            # full-length vector so classes absent from the train split don't
            # shorten/misalign the weights
            full_weights = numpy.ones(nclasses, dtype=numpy.float64)
            full_weights[present.astype(int)] = present_weights
            cfg.train.class_weights = full_weights.tolist()
            criterion = nn.CrossEntropyLoss(
                weight=torch.FloatTensor(cfg.train.class_weights).to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()

        # ------------------------------------------------- optimizer / scheduler
        if self.component.optimizer is None:
            parameters = filter(lambda p: p.requires_grad, self.component.model.parameters())
            self.component.optimizer = AdamW(parameters, lr=cfg.train.lr,
                                             weight_decay=cfg.train.adam_weight_decay,
                                             betas=cfg.train.adam_betas,
                                             eps=cfg.train.adam_epsilon)

        if self.component.scheduler is None:
            self.component.scheduler = MultiStepLR(
                self.component.optimizer, milestones=cfg.train.multistep_milestones,
                gamma=cfg.train.multistep_lr_gamma)  # type: ignore

        # Load model/optimizer/scheduler state (and best_f1) from the checkpoint,
        # but do NOT continue the epoch count: always start from epoch 0.
        _, self.best_f1 = load_state(
            self.component.model, self.component.optimizer, self.component.scheduler,
            load_best=False, path=checkpoint_path, relcat_config=cfg)
        self.epoch = 0

        self.log.info("Starting training process...")

        losses_per_epoch, accuracy_per_epoch, f1_per_epoch = load_results(path=checkpoint_path)

        gradient_acc_steps = max(1, cfg.train.gradient_acc_steps)
        max_grad_norm = cfg.train.max_grad_norm
        n_epochs = cfg.train.nepochs
        n_batches = len(train_dataloader)
        log_every = max(1, n_batches // 10)

        optimizer = self.component.optimizer
        optimizer.zero_grad()

        # ------------------------------------------------------------------ loop
        for epoch in range(n_epochs):
            start_time = time.perf_counter()

            loss_per_batch = []
            accuracy_per_batch = []

            self.log.info("Currently training epoch %d of %d" % (epoch + 1, n_epochs))

            self.component.model.train()
            pbar = tqdm(total=train_dataset_size)

            for i, data in enumerate(train_dataloader):
                current_batch_size = len(data[0])
                token_ids, e1_e2_start, labels, _, _ = data

                attention_mask = (token_ids != self.component.pad_id).float().to(self.device)
                token_type_ids = torch.zeros(
                    (token_ids.shape[0], token_ids.shape[1])).long().to(self.device)

                token_ids = token_ids.to(self.device)
                labels = labels.to(self.device)

                _, classification_logits = self.component.model(
                    input_ids=token_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    e1_e2_start=e1_e2_start
                )

                batch_loss = criterion(
                    classification_logits.view(-1, nclasses).to(self.device), labels.squeeze(1))

                # size of the accumulation window this batch belongs to (the last
                # window of an epoch may be shorter than gradient_acc_steps)
                window_start = (i // gradient_acc_steps) * gradient_acc_steps
                window_len = min(gradient_acc_steps, n_batches - window_start)

                # scale so accumulated gradients equal the mean over the window
                (batch_loss / window_len).backward()

                batch_acc, _, batch_precision, batch_f1, _, _, batch_stats_per_label = self.evaluate_(
                    classification_logits, labels, ignore_idx=-1)

                loss_per_batch.append(batch_loss.item())
                accuracy_per_batch.append(batch_acc)

                # step at the end of each accumulation window, including the last
                # (possibly partial) window of the epoch
                if ((i + 1) % gradient_acc_steps == 0) or ((i + 1) == n_batches):
                    torch.nn.utils.clip_grad_norm_(self.component.model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                if (i + 1) % log_every == 0:
                    self.log.debug(
                        "[Epoch: %d, batch %d/%d, loss: %.3f, accuracy: %.3f, running mean loss: %.3f]" %
                        (epoch, i + 1, n_batches, loss_per_batch[-1], accuracy_per_batch[-1],
                         sum(loss_per_batch) / len(loss_per_batch)))

                pbar.update(current_batch_size)

            pbar.close()

            # MultiStepLR milestones are interpreted as epochs here
            self.component.scheduler.step()

            if len(loss_per_batch) > 0:
                losses_per_epoch.append(sum(loss_per_batch) / len(loss_per_batch))
                self.log.info("Losses at Epoch %d: %.5f" % (epoch, losses_per_epoch[-1]))

            if len(accuracy_per_batch) > 0:
                accuracy_per_epoch.append(sum(accuracy_per_batch) / len(accuracy_per_batch))
                self.log.info("Train accuracy at Epoch %d: %.5f" % (epoch, accuracy_per_epoch[-1]))

            self.log.info("======================== TRAIN SET TEST RESULTS ========================")
            _ = self.evaluate_results(train_dataloader, self.component.pad_id)

            self.log.info("======================== TEST SET TEST RESULTS ========================")
            results = self.evaluate_results(test_dataloader, self.component.pad_id)

            f1_per_epoch.append(results['f1'])

            self.log.info("Epoch finished, took %.1f seconds" % (time.perf_counter() - start_time))

            self.epoch += 1

            if f1_per_epoch[-1] > self.best_f1:
                self.best_f1 = f1_per_epoch[-1]
                save_state(self.component.model, self.component.optimizer, self.component.scheduler,
                           self.epoch, self.best_f1, checkpoint_path,
                           model_name=cfg.general.model_name, task=self.task, is_checkpoint=False)
                self.log.info("######################### Model saved to disk ##############")

            save_results({"losses_per_epoch": losses_per_epoch, "accuracy_per_epoch": accuracy_per_epoch,
                          "f1_per_epoch": f1_per_epoch, "epoch": epoch},
                         file_prefix="train", path=checkpoint_path)
            save_state(self.component.model, self.component.optimizer, self.component.scheduler,
                       self.epoch, self.best_f1, checkpoint_path,
                       model_name=cfg.general.model_name, task=self.task, is_checkpoint=True)


    def evaluate_(self, output_logits, labels, ignore_idx):
        """Per-batch metrics via sklearn. Same return signature as before:
        (acc, recall, precision, f1, pred_labels, true_labels, stat_per_label).

        The recall/precision/f1 returned here are weighted averages (weighted by class support).
        Per-batch values on small batches are noisy; use evaluate_results for trustworthy numbers.
        """
        # ignore index (padding) when calculating metrics
        idxs = (labels != ignore_idx).view(-1).to(output_logits.device)
        true_labels = labels.view(-1).to(output_logits.device)[idxs].cpu().tolist()
        pred_labels = torch.softmax(output_logits, dim=1).max(1)[1][idxs].cpu().tolist()

        if len(true_labels) == 0:
            return 0.0, 0.0, 0.0, 0.0, pred_labels, true_labels, {}

        report = classification_report(true_labels, pred_labels,
                                       output_dict=True, zero_division=0)

        skip = ("accuracy", "micro avg", "macro avg", "weighted avg")
        stat_per_label = {
            int(k): {"f1": v["f1-score"], "prec": v["precision"],
                     "recall": v["recall"], "support": v["support"]}
            for k, v in report.items() if k not in skip
        }

        acc = accuracy_score(true_labels, pred_labels)
        w = report["weighted avg"]

        return acc, w["recall"], w["precision"], w["f1-score"], pred_labels, true_labels, stat_per_label

    def evaluate_results(self, data_loader, pad_id):
        self.log.info("Evaluating test samples...")
        if self.component.relcat_config.train.class_weights is not None and self.component.relcat_config.train.enable_class_weights:
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.component.relcat_config.train.class_weights).to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        num_batches = 0
        all_true: List[int] = []
        all_pred: List[int] = []

        self.component.model.eval()

        for i, data in enumerate(data_loader):
            with torch.no_grad():
                token_ids, e1_e2_start, labels, _, _ = data
                attention_mask = (token_ids != pad_id).float().to(self.device)
                token_type_ids = torch.zeros(
                    (token_ids.shape[0], token_ids.shape[1])).long().to(self.device)

                token_ids = token_ids.to(self.device)
                # e1_e2_start = e1_e2_start.to(self.device)
                labels = labels.to(self.device)

                model_output, pred_classification_logits = self.component.model(
                    token_ids, token_type_ids=token_type_ids,
                    attention_mask=attention_mask, Q=None,
                    e1_e2_start=e1_e2_start)

                pred_classification_logits = pred_classification_logits.to(self.device)

                batch_loss = criterion(pred_classification_logits.view(
                    -1, self.component.relcat_config.train.nclasses), labels.squeeze(1))
                total_loss += batch_loss.item()
                num_batches += 1

                # collect predictions over the whole loader; metrics are computed once at the end
                labels_flat = labels.view(-1)
                keep = labels_flat != -1
                all_true.extend(labels_flat[keep].cpu().tolist())
                all_pred.extend(pred_classification_logits.argmax(dim=1)[keep].cpu().tolist())

        idx2labels = self.component.relcat_config.general.idx2labels

        # only score classes that actually occur (in truth or prediction), so absent classes
        # don't drag the macro average down with 0 scores
        label_ids = sorted(set(all_true) | set(all_pred))
        target_names = [str(idx2labels.get(i, i)) for i in label_ids]

        if len(all_true) > 0:
            report = classification_report(
                all_true, all_pred, labels=label_ids, target_names=target_names,
                output_dict=True, zero_division=0)
            report_text = classification_report(
                all_true, all_pred, labels=label_ids, target_names=target_names,
                zero_division=0)
            accuracy = report["accuracy"]
            macro = report["macro avg"]
        else:
            report_text = "(no labelled samples)"
            accuracy = 0.0
            macro = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}

        results = {
            "loss": total_loss / max(num_batches, 1),
            "accuracy": accuracy,
            "precision": macro["precision"],   # macro averages
            "recall": macro["recall"],
            "f1": macro["f1-score"],           # macro F1 drives best-model selection
        }

        self.log.info("==================== Evaluation Results ===================")
        self.log.info(" no. of batches:" + str(num_batches))
        for key in sorted(results.keys()):
            self.log.info(" %s = %0.3f" % (key, results[key]))
        self.log.info("----------------------- class stats -----------------------")
        self.log.info("\n" + report_text)
        self.log.info("===========================================================")

        return results

    def pipe(self, stream: Iterable[Doc], *args, **kwargs) -> Iterator[Doc]:

        predict_rel_dataset = RelData(
            cdb=self.cdb, config=self.component.relcat_config, tokenizer=self.component.tokenizer)

        self._move_model_to_device()
        self.component.model.eval()

        for doc_id, doc in enumerate(stream, 0):
            predict_rel_dataset.dataset, _ = self._create_test_train_datasets(
                data=predict_rel_dataset.create_base_relations_from_doc(doc, doc_id=str(doc_id)),
                split_sets=False)

            predict_dataloader = DataLoader(dataset=predict_rel_dataset, shuffle=False,
                                            batch_size=self.component.relcat_config.train.batch_size,
                                            num_workers=0, collate_fn=self.component.padding_seq,
                                            pin_memory=self.component.relcat_config.general.pin_memory)

            total_rel_found = len(predict_rel_dataset.dataset["output_relations"])
            rel_idx = -1

            self.log.info("total relations for doc: " + str(total_rel_found))
            self.log.info("processing...")

            pbar = tqdm(total=total_rel_found)

            for data in predict_dataloader:
                with torch.no_grad():
                    token_ids, e1_e2_start, labels, _, _ = data

                    attention_mask = (token_ids != self.component.pad_id).float().to(self.device)
                    token_type_ids = torch.zeros(
                        token_ids.shape[0], token_ids.shape[1]).long().to(self.device)

                    token_ids = token_ids.to(self.device)
                    # e1_e2_start = e1_e2_start.to(self.device)

                    model_output, pred_classification_logits = self.component.model(
                        token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                        e1_e2_start=e1_e2_start)  # type: ignore

                    for pred_rel_logits in pred_classification_logits:
                        rel_idx += 1

                        confidence = torch.softmax(
                            pred_rel_logits, dim=0).max(0)
                        predicted_label_id = int(confidence[1].item())

                        doc._.relations.append({"relation": self.component.relcat_config.general.idx2labels[predicted_label_id],
                                                "label_id": predicted_label_id,
                                                "ent1_text": predict_rel_dataset.dataset["output_relations"][rel_idx][2],
                                                "ent2_text": predict_rel_dataset.dataset["output_relations"][rel_idx][3],
                                                "confidence": float("{:.3f}".format(confidence[0].item())),
                                                "start_ent1_char_pos": predict_rel_dataset.dataset["output_relations"][rel_idx][18],
                                                "end_ent1_char_pos": predict_rel_dataset.dataset["output_relations"][rel_idx][19],
                                                "start_ent2_char_pos": predict_rel_dataset.dataset["output_relations"][rel_idx][20],
                                                "end_ent2_char_pos": predict_rel_dataset.dataset["output_relations"][rel_idx][21],
                                                "start_entity_id": predict_rel_dataset.dataset["output_relations"][rel_idx][8],
                                                "end_entity_id": predict_rel_dataset.dataset["output_relations"][rel_idx][9]})
                    pbar.update(len(token_ids))
            pbar.close()

            yield doc

    def predict_text_with_anns(self, text: str, annotations: List[Dict]) -> Doc:
        """ Creates spacy doc from text and annotation input. Predicts using self.__call__

        Args:
            text (str): text
            annotations (Dict): dict containing the entities from NER (of your choosing), the format
                must be the following format:
                        [
                            {
                                "cui": "202099003", -this is optional
                                "value": "discoid lateral meniscus",
                                "start": 294,
                                "end": 318
                            },
                            {
                                "cui": "202099003",
                                "value": "Discoid lateral meniscus",
                                "start": 1905,
                                "end": 1929,
                            }
                        ]

        Returns:
            Doc: spacy doc with the relations.
        """

        Span.set_extension('id', default=0, force=True)
        Span.set_extension('cui', default=None, force=True)
        Span.set_extension('start', default=None, force=True)
        Span.set_extension('end', default=None, force=True)
        Doc.set_extension('ents', default=[], force=True)
        Doc.set_extension('relations', default=[], force=True)
        nlp = spacy.blank(self.component.relcat_config.general.language)
        doc = nlp(text)

        for ann in annotations:
            tkn_idx = []
            for ind, word in enumerate(doc):
                end_char = word.idx + len(word.text)
                # identify token index by checking against ann char position
                if end_char <= ann['end'] and end_char > ann['start']:
                    tkn_idx.append(ind)
            entity = Span(doc, min(tkn_idx), max(tkn_idx) + 1, label=ann["value"])
            entity._.cui = ann["cui"]
            entity._.start = ann["start"]
            entity._.end = ann["end"]
            doc._.ents.append(entity)

        doc = self(doc)

        return doc