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
        self.entity_embeddings = nn.Embedding(n_entities, k)
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

    def head_differences(self, relation, tail):
        _, relation, tail = self.params_to(0, relation, tail, self.device)
        head_embeddings = self.entity_embeddings(tt.tensor(range(self.n_entities)).to(self.device))
        p_t_e = head_embeddings + self.relation_embeddings(relation)
        differences = p_t_e - self.entity_embeddings(tail)  # The lowest difference should be when the head e == h
        differences = tt.norm(differences, p=2, dim=1)
        return differences

    def tail_differences(self, head, relation):
        head, relation, _ = self.params_to(head, relation, 0, self.device)
        tail_embeddings = self.entity_embeddings(tt.tensor(range(self.n_entities)).to(self.device))
        p_t_e = self.entity_embeddings(head) + self.entity_embeddings(relation)
        differences = p_t_e - tail_embeddings  # The lowest difference should be when the tail e == t
        differences = tt.norm(differences, p=2, dim=1)
        return differences

    def predict_movies_for_user(self, u_idx, relation_idx, movie_indices):

        u_idx, relation_idx, movie_indices = self.params_to(u_idx, relation_idx, movie_indices, self.device)
        prediction_vector = self.entity_embeddings(u_idx) + self.relation_embeddings(relation_idx)

        # Calculate similarity to all movie embeddings
        movie_embeddings = self.entity_embeddings(movie_indices)
        similarities = tt.norm(prediction_vector - movie_embeddings, p=2, dim=1)

        score_dict = {m: s for m, s in zip(movie_indices, similarities)}

        return score_dict


if __name__ == '__main__':
    pass



