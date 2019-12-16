import torch as tt
import torch.nn as nn
import torch.nn.functional as ff
import numpy as np


class TransE(nn.Module):
    def __init__(self, n_entities, n_relations, margin, k):
        super(TransE, self).__init__()

        # Hyper params
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.margin = tt.tensor(margin)
        self.k = k

        # Initialize embeddings
        self.entity_embeddings = nn.Embedding(n_entities, k, )
        self.relation_embeddings = nn.Embedding(n_relations, k)

        self.entity_embeddings.weight.data.uniform_(-6 / np.sqrt(k), 6 / np.sqrt(k))
        self.relation_embeddings.weight.data.uniform_(-6 / np.sqrt(k), 6 / np.sqrt(k))

        self.device = tt.device('cuda:0') if tt.cuda.is_available() else tt.device('cpu')

        self.to(self.device)

    def params_to(self, h, r, t, device):
        head = tt.tensor(h)
        relation = tt.tensor(r)
        tail = tt.tensor(t)

        head = head.to(device)
        relation = relation.to(device)
        tail = tail.to(device)

        return head, relation, tail

    def forward(self, head, relation, tail):
        head, relation, tail = self.params_to(head, relation, tail, self.device)

        h_e = self.entity_embeddings(head)
        r_e = self.relation_embeddings(relation)
        t_e = self.entity_embeddings(tail)

        h_e = ff.normalize(h_e, p=2, dim=1)
        r_e = ff.normalize(r_e, p=2, dim=1)
        t_e = ff.normalize(t_e, p=2, dim=1)

        p_t_e = h_e + r_e

        loss = tt.norm(p_t_e - t_e, p=2, dim=1)  # The distance between p_t_e and t_e

        return loss

    def normalize_entity_embeddings(self):
        self.entity_embeddings.weight.data = ff.normalize(self.entity_embeddings.weight.data, 2, 1)

    def head_energies(self, relation, tail):
        _, relation, tail = self.params_to(0, relation, tail, self.device)
        head_embeddings = self.entity_embeddings(tt.tensor(range(self.n_entities)).to(self.device))
        p_t_e = head_embeddings + self.relation_embeddings(relation)
        differences = p_t_e - self.entity_embeddings(tail)  # The lowest difference should be when the head e == h
        differences = tt.norm(differences, p=2, dim=1)
        return differences

    def tail_energies(self, head, relation):
        head, relation, _ = self.params_to(head, relation, 0, self.device)
        tail_embeddings = self.entity_embeddings(tt.tensor(range(self.n_entities)).to(self.device))
        p_t_e = self.entity_embeddings(head) + self.entity_embeddings(relation)
        differences = p_t_e - tail_embeddings  # The lowest difference should be when the tail e == t
        differences = tt.norm(differences, p=2, dim=1)
        return differences

    def fast_validate(self, h, r, t):
        h, r, t = self.params_to(h, r, t, self.device)

        self.normalize_entity_embeddings()

        h_e = self.entity_embeddings(h)
        r_e = self.relation_embeddings(r)
        t_e = self.entity_embeddings(t)

        target_loss = tt.norm(h_e + r_e - t_e, 2).repeat(self.n_entities, 1)

        tmp_h_loss = tt.norm(self.entity_embeddings.weight.data + (r_e - t_e), 2, 1).view(-1, 1)
        tmp_t_loss = tt.norm((h_e + r_e) - self.entity_embeddings.weight.data, 2, 1).view(-1, 1)

        rank_h = tt.nonzero(ff.relu(target_loss - tmp_h_loss)).size()[0]
        rank_t = tt.nonzero(ff.relu(target_loss - tmp_t_loss)).size()[0]

        return (rank_h + rank_t) / 2

    def predict_movies_for_user(self, u_idx, relation_idx, movie_indices):

        u_idx, relation_idx, movie_indices = self.params_to(u_idx, relation_idx, movie_indices, self.device)
        prediction_vector = self.entity_embeddings(u_idx) + self.relation_embeddings(relation_idx)

        # Calculate similarity to all movie embeddings
        movie_embeddings = self.entity_embeddings(movie_indices)
        similarities = tt.norm(prediction_vector - movie_embeddings, p=2, dim=1)
        # similarities = movie_embeddings @ prediction_vector

        return zip(movie_indices, similarities)

    def fast_rank(self, u_idx, relation_idx, pos_sample, neg_samples):
        u_idx, relation_idx, movie_indices = self.params_to(u_idx, relation_idx, neg_samples + [pos_sample], self.device)
        pos_sample = tt.tensor(pos_sample).to(self.device)

        prediction_vector = self.entity_embeddings(u_idx) + self.relation_embeddings(relation_idx)
        movie_embeddings = self.entity_embeddings(movie_indices)
        target_distance = tt.norm(prediction_vector - self.entity_embeddings(pos_sample), p=2)

        all_distances = tt.norm(prediction_vector - movie_embeddings, p=2, dim=1)
        all_distances -= target_distance
        better_items = tt.where(all_distances < 0)[0]
        return len(better_items)


if __name__ == '__main__':
    pass



