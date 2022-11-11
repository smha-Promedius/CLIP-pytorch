#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : dataloader
# @Date : 2022-08-23-14-38
# @Project : CLIP
# @Author : seungmin

import torch, cv2
import numpy as np
import pandas as pd

import albumentations as A


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, max_length, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        """
        self.image_path = image_path
        self.max_length = max_length
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=self.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{self.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item

    def __len__(self):
        return len(self.captions)


class MyCLIPDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, image_size, image_path, captions_path, max_length, debug):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.image_path = image_path
        self.captions_path = captions_path
        self.debug = debug
        self.max_length = max_length

    def get_transforms(self, mode="train"):
        if mode == "train":
            return A.Compose(
                [
                    A.Resize(self.image_size, self.image_size, always_apply=True),
                    A.Normalize(max_pixel_value=255.0, always_apply=True),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.image_size, self.image_size, always_apply=True),
                    A.Normalize(max_pixel_value=255.0, always_apply=True),
                ]
            )


    def make_train_valid_dfs(self):
        dataframe = pd.read_csv(f"{self.captions_path}/captions.csv")
        max_id = dataframe["id"].max() + 1 if not self.debug else 100
        image_ids = np.arange(0, max_id)
        np.random.seed(42)
        valid_ids = np.random.choice(
            image_ids, size=int(0.2 * len(image_ids)), replace=False
        )
        train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
        train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
        valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
        return train_dataframe, valid_dataframe


    def build_loaders(self, dataframe, tokenizer, mode):
        transforms = self.get_transforms(mode=mode)
        dataset = CLIPDataset(
            self.image_path,
            self.max_length,
            dataframe["image"].values,
            dataframe["caption"].values,
            tokenizer=tokenizer,
            transforms=transforms,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True if mode == "train" else False,
        )
        return dataloader