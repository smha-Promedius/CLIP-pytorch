#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : train
# @Date : 2022-08-23-14-37
# @Project : CLIP
# @Author : seungmin

import os, yaml

from trainer.trainer import CLIPTrainer
from utils.dataloader import MyCLIPDatasetWrapper


def main():
    config = yaml.load(open(os.path.join("./config") + "/config.yaml", "r"), Loader=yaml.FullLoader)

    trainset = MyCLIPDatasetWrapper(**config['data'])
    downstream = CLIPTrainer(trainset, config)

    downstream.train()

if __name__ == "__main__":
    main()