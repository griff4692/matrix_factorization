import os

import numpy as np
import pandas as pd
import torch
from torch.optim import SGD
import torch.nn as nn
from torch.utils.data import DataLoader

SEED = 1992
np.random.seed(SEED)


def most_similar_to(movie_id, movie_matrix_similarity, movie_vocab, movie_names, topk=5):
    similarity = movie_matrix_similarity[movie_id]
    similarity[movie_id] = -1
    closest_movie_ids = np.argsort(similarity)[-topk:][::-1]
    closest_titles = map(lambda x: movie_id_to_name(x, movie_vocab, movie_names), closest_movie_ids)
    print(movie_id_to_name(movie_id, movie_vocab, movie_names))
    print('\t' + '\n\t'.join(closest_titles))


def sample_similar_movies(attributes, movie_vocab, movie_names):
    norm = torch.pow(attributes, 2).sum(-1) + 1e-5
    movie_matrix_similarity = torch.bmm(attributes.unsqueeze(0), attributes.transpose(1,0).unsqueeze(0)).squeeze()
    movie_matrix_similarity /= norm.unsqueeze(0).repeat(M, 1)
    movie_matrix_similarity /= norm.unsqueeze(1).repeat(1, M)
    movie_matrix_similarity = movie_matrix_similarity.detach().numpy()
    for movie_id in random_movie_ids:
        most_similar_to(movie_id, movie_matrix_similarity, movie_vocab, movie_names)


def train_test_split(arr, train_fract=0.8):
    indices = np.arange(len(arr))
    np.random.shuffle(indices)
    target_train = round(len(arr) * train_fract)
    return arr[indices[:target_train]], arr[indices[target_train:]]


class Vocab:
    def __init__(self):
        self.w2i = {}
        self.i2w = []

    def add_tokens(self, tokens):
        for token in tokens:
            self.add_token(token)

    def add_token(self, token):
        if token not in self.w2i:
            self.w2i[token] = len(self.i2w)
            self.i2w.append(token)

    def get_id(self, token):
        if token in self.w2i:
            return self.w2i[token]
        return -1

    def get_token(self, id):
        return self.i2w[id]

    def size(self):
        return len(self.i2w)


class MatrixFactorizer(nn.Module):
    def __init__(self, num_users, num_movies, model_dim=50):
        super(MatrixFactorizer, self).__init__()
        self.users = nn.Embedding(num_embeddings=num_users, embedding_dim=model_dim, padding_idx=-1)
        self.movies = nn.Embedding(num_embeddings=num_movies, embedding_dim=model_dim, padding_idx=-1)

        self.user_bias = nn.Embedding(num_embeddings=num_users, embedding_dim=1, padding_idx=-1)
        self.movie_bias = nn.Embedding(num_embeddings=num_movies, embedding_dim=1, padding_idx=-1)

    def forward(self, user_ids, movie_ids):
        user_embeds = self.users(user_ids)
        movie_embeds = self.movies(movie_ids)
        output = torch.bmm(user_embeds.unsqueeze(1), movie_embeds.unsqueeze(-1)).squeeze()
        user_bias = self.user_bias(user_ids).squeeze()
        movie_bias = self.movie_bias(movie_ids).squeeze()
        return output + user_bias + movie_bias


def movie_id_to_name(movie_id, movie_vocab, movie_titles):
    return movie_titles[movie_vocab.get_token(movie_id)]


def get_movie_names(data_dir):
    data = pd.read_csv(os.path.join(data_dir, 'movies.csv'))
    movie_names = {}
    for index, row in data.iterrows():
        row = row.to_dict()
        movie_names[row['movieId']] = row['title']
    return movie_names


if __name__ == '__main__':
    data_dir = os.path.expanduser('~/Desktop/movielens/')
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
    movie_names = get_movie_names(data_dir)
    ratings = ratings.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})
    users = ratings['user_id'].unique().tolist()
    movies = ratings['movie_id'].unique().tolist()
    user_vocab = Vocab()
    movie_vocab = Vocab()
    user_vocab.add_tokens(users)
    movie_vocab.add_tokens(movies)
    ratings['user_embed_idx'] = ratings['user_id'].apply(lambda x: user_vocab.get_id(x))
    ratings['movie_embed_idx'] = ratings['movie_id'].apply(lambda x: movie_vocab.get_id(x))
    ratings = ratings[['user_embed_idx', 'movie_embed_idx', 'rating']].sample(frac=1).reset_index(drop=True)
    N = len(users)
    M = len(movies)
    rating_mean = ratings['rating'].mean()
    train_idx = int(ratings.shape[0] * 0.8)
    train_data = ratings[:train_idx]
    dev_data = ratings[train_idx:]

    train_loader = DataLoader(train_data.to_numpy(), batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_data.to_numpy(), batch_size=32)

    num_epochs = 20
    batch_size = 128

    model = MatrixFactorizer(N, M)
    optimizer = SGD(model.parameters(), lr=0.01, weight_decay=1e-2)

    for epoch in range(1, num_epochs + 1):
        epoch_train_loss = 0.0
        num_train_batches = 0
        num_dev_batches = 0
        epoch_dev_loss = 0.0
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            users = batch[:, 0].long()
            movies = batch[:, 1].long()
            # ratings = (batch[:, 2] - rating_mean).float()
            ratings = batch[:, 2].float()

            ratings_guess = model(users, movies)
            batch_loss = torch.pow(ratings - ratings_guess, 2).mean()
            batch_loss.backward()
            optimizer.step()
            num_train_batches += 1
            epoch_train_loss += batch_loss.item()
        model.eval()
        for batch_idx, batch in enumerate(dev_loader):
            users = batch[:, 0].long()
            movies = batch[:, 1].long()
            ratings = (batch[:, 2] - rating_mean).float()
            with torch.no_grad():
                ratings_guess = model(users, movies)
            batch_loss = torch.pow(ratings - ratings_guess, 2).mean()
            num_dev_batches += 1
            epoch_dev_loss += batch_loss.item()
        print('Epoch loss. Train={}, Dev={}'.format(epoch_train_loss / float(num_train_batches),
                                                    epoch_dev_loss / float(num_dev_batches)))

    random_movie_ids = np.random.choice(np.arange(M), size=25, replace=False)
    attributes = model.movies.weight
    sample_similar_movies(attributes, movie_vocab, movie_names)

