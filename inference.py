#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : inference
# @Date : 2022-08-29-08-38
# @Project : CLIP
# @Author : seungmin

import os, yaml, cv2
import torch
import torch.nn.functional as F

from model.clip import ImageEncoder, TextEncoder, ProjectionHead, CLIPModel
from utils.dataloader import MyCLIPDatasetWrapper
from transformers import DistilBertTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt


class CLIPInference(object):

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.device = self.get_device()

    def get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def get_image_embeddings(self, df, model_path):
        #_, valid_df = self.dataset.make_train_valid_dfs()
        tokenizer = DistilBertTokenizer.from_pretrained(self.config['train']['text_tokenizer'])
        valid_loader = self.dataset.build_loaders(df, tokenizer, mode="valid")

        img_encoder = ImageEncoder(**self.config['model']['image_encoder'])
        txt_encoder = TextEncoder(**self.config['model']['text_encoder'])
        img_prj_head = ProjectionHead(embedding_dim=self.config['model']['clip']['image_embedding'],
                                      **self.config['model']['projection_head'])
        txt_prj_head = ProjectionHead(embedding_dim=self.config['model']['clip']['text_embedding'],
                                      **self.config['model']['projection_head'])

        clip = CLIPModel(img_encoder, txt_encoder, img_prj_head, txt_prj_head,
                         self.config['model']['clip']['temperature'])
        model = clip.to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        valid_image_embeddings = []
        with torch.no_grad():
            for batch in tqdm(valid_loader):
                image_features = model.image_encoder(batch["image"].to(self.device))
                image_embeddings = model.image_projection(image_features)
                valid_image_embeddings.append(image_embeddings)
        return model, torch.cat(valid_image_embeddings)

    def find_matches(self, model, image_embeddings, query, image_filenames, save_to, n=9):
        tokenizer = DistilBertTokenizer.from_pretrained(self.config['train']['text_tokenizer'])
        encoded_query = tokenizer([query])
        batch = {
            key: torch.tensor(values).to(self.device)
            for key, values in encoded_query.items()
        }
        with torch.no_grad():
            text_features = model.text_encoder(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            text_embeddings = model.text_projection(text_features)

        image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
        dot_similarity = text_embeddings_n @ image_embeddings_n.T

        values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
        matches = [image_filenames[idx] for idx in indices[::5]]

        _, axes = plt.subplots(3, 3, figsize=(10, 10))
        for match, ax in zip(matches, axes.flatten()):
            image = cv2.imread(f"{self.config['data']['image_path']}/{match}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            ax.axis("off")
        plt.savefig(save_to)
        #plt.show()


def main(input_query):
    config = yaml.load(open(os.path.join("./config") + "/config.yaml", "r"), Loader=yaml.FullLoader)

    trainset = MyCLIPDatasetWrapper(**config['data'])
    inference = CLIPInference(trainset, config)

    _, valid_df = trainset.make_train_valid_dfs()
    model, image_embeddings = inference.get_image_embeddings(valid_df, "./best.pt")

    inference.find_matches(model,
                           image_embeddings,
                           query=input_query,
                           image_filenames=valid_df['image'].values,
                           save_to=input_query + '.png',
                           n=9)


if __name__ == "__main__":
    main("A cat looking at person")