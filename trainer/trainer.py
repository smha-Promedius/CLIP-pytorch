#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : trainer
# @Date : 2022-08-24-14-25
# @Project : CLIP
# @Author : seungmin

import torch
from transformers import DistilBertTokenizer
from model.clip import ImageEncoder, TextEncoder, ProjectionHead, CLIPModel

import itertools
from tqdm import tqdm
from utils.utils import AvgMeter, get_lr


class CLIPTrainer(object):
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.device = self.get_device()

    def get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def train(self):
        train_df, valid_df = self.dataset.make_train_valid_dfs()
        tokenizer = DistilBertTokenizer.from_pretrained(self.config['train']['text_tokenizer'])
        train_loader = self.dataset.build_loaders(train_df, tokenizer, mode="train")
        valid_loader = self.dataset.build_loaders(valid_df, tokenizer, mode="valid")

        img_encoder = ImageEncoder(**self.config['model']['image_encoder'])
        txt_encoder = TextEncoder(**self.config['model']['text_encoder'])
        img_prj_head = ProjectionHead(embedding_dim=self.config['model']['clip']['image_embedding'],
                                      **self.config['model']['projection_head'])
        txt_prj_head = ProjectionHead(embedding_dim=self.config['model']['clip']['text_embedding'],
                                      **self.config['model']['projection_head'])

        clip = CLIPModel(img_encoder, txt_encoder, img_prj_head, txt_prj_head, self.config['model']['clip']['temperature'])
        clip = clip.to(self.device)

        params = [
            {"params": clip.image_encoder.parameters(), "lr": self.config['train']['image_encoder_lr']},
            {"params": clip.text_encoder.parameters(), "lr": self.config['train']['text_encoder_lr']},
            {"params": itertools.chain(
                clip.image_projection.parameters(), clip.text_projection.parameters()
            ), "lr": self.config['train']['head_lr'], "weight_decay": self.config['train']['weight_decay']}
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=0.)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=self.config['train']['patience'], factor=self.config['train']['factor']
        )
        step = "epoch"

        best_loss = float('inf')
        for epoch in range(self.config['train']['epoch']):
            print(f"Epoch: {epoch + 1}")
            clip.train()
            train_loss = self.train_epoch(clip, train_loader, optimizer, lr_scheduler, step)
            clip.eval()
            with torch.no_grad():
                valid_loss = self.valid_epoch(clip, valid_loader)

            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                torch.save(clip.state_dict(), "best.pt")
                print("Saved Best Model!")

            lr_scheduler.step(valid_loss.avg)

    def train_epoch(self, model, train_loader, optimizer, lr_scheduler, step):
        loss_meter = AvgMeter()
        tqdm_object = tqdm(train_loader, total=len(train_loader))
        for batch in tqdm_object:
            batch = {k: v.to(self.device) for k, v in batch.items() if k != "caption"}
            loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step == "batch":
                lr_scheduler.step()

            count = batch["image"].size(0)
            loss_meter.update(loss.item(), count)

            tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
        return loss_meter


    def valid_epoch(self, model, valid_loader):
        loss_meter = AvgMeter()

        tqdm_object = tqdm(valid_loader, total=len(valid_loader))
        for batch in tqdm_object:
            batch = {k: v.to(self.device) for k, v in batch.items() if k != "caption"}
            loss = model(batch)

            count = batch["image"].size(0)
            loss_meter.update(loss.item(), count)

            tqdm_object.set_postfix(valid_loss=loss_meter.avg)
        return loss_meter