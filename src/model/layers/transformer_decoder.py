import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class InitialDecoder(nn.Module):
    def __init__(self, modalities, neighbors, future_len):
        super(InitialDecoder, self).__init__()
        dim = 128
        self._modalities = modalities
        self.multi_modal_query_embedding = nn.Embedding(modalities, dim)
        self.agent_query_embedding = nn.Embedding(neighbors+1, dim)
        self.query_encoder = CrossTransformer()
        self.predictor = GMMPredictor(future_len)
        self.register_buffer('modal', torch.arange(modalities).long())
        self.register_buffer('agent', torch.arange(neighbors+1).long())

    def forward(self, id, current_state, encoding, mask):
        # get query
        multi_modal_query = self.multi_modal_query_embedding(self.modal)
        agent_query = self.agent_query_embedding(self.agent[id])
        multi_modal_agent_query = multi_modal_query + agent_query[None, :]
        query = encoding[:, None, id] + multi_modal_agent_query

        # decode trajectories
        query_content = self.query_encoder(query, encoding, encoding, mask)
        predictions, scores = self.predictor(query_content)

        # post process
        predictions[..., :2] += current_state[:, None, None, :2]

        return query_content, predictions, scores


class GMMPredictor(nn.Module):
    def __init__(self, future_len):
        super(GMMPredictor, self).__init__()
        self._future_len = future_len
        self.gaussian = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(256, self._future_len * 4))
        self.score = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, input):
        B, M, _ = input.shape
        res = self.gaussian(input).view(B, M, self._future_len, 4)  # mu_x, mu_y, log_sig_x, log_sig_y
        score = self.score(input).squeeze(-1)

        return res, score


class CrossTransformer(nn.Module):
    def __init__(self):
        super(CrossTransformer, self).__init__()
        heads, dim, dropout = 8, 128, 0.1
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, query, key, value, mask=None):
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class InteractionDecoder(nn.Module):
    def __init__(self, future_encoder, future_len):
        super(InteractionDecoder, self).__init__()
        self.interaction_encoder = SelfTransformer()
        self.query_encoder = CrossTransformer()
        self.future_encoder = future_encoder
        self.decoder = GMMPredictor(future_len)

    def forward(self, id, current_states, actors, scores, last_content, encoding, mask):
        B, N, M, T, _ = actors.shape

        # encoding the trajectories from the last level
        multi_futures = self.future_encoder(actors[..., :2], current_states)
        futures = (multi_futures * scores.softmax(-1).unsqueeze(-1)).mean(dim=2)

        # encoding the interaction using self-attention transformer
        interaction = self.interaction_encoder(futures, mask[:, :N])

        # append the interaction encoding to the context encoding
        encoding = torch.cat([interaction, encoding], dim=1)
        mask = torch.cat([mask[:, :N], mask], dim=1).clone()
        mask[:, id] = True  # mask the agent future itself from last level

        # decoding the trajectories from the current level
        query = last_content + multi_futures[:, id]
        query_content = self.query_encoder(query, encoding, encoding, mask)
        trajectories, scores = self.decoder(query_content)

        # post process
        trajectories[..., :2] += current_states[:, id, None, None, :2]

        return query_content, trajectories, scores

class FutureEncoder(nn.Module):
    def __init__(self):
        super(FutureEncoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 128))
        self.type_emb = nn.Embedding(5, 128, padding_idx=0)

    def state_process(self, trajs, current_states):
        M = trajs.shape[2]
        current_states = current_states.unsqueeze(2).expand(-1, -1, M, -1)
        xy = torch.cat([current_states[:, :, :, None, :2], trajs], dim=-2)
        dxy = torch.diff(xy, dim=-2)
        v = dxy / 0.1
        theta = torch.atan2(dxy[..., 1], dxy[..., 0].clamp(min=1e-3)).unsqueeze(-1)
        T = trajs.shape[3]
        # size = current_states[:, :, :, None, 5:8].expand(-1, -1, -1, T, -1)
        trajs = torch.cat([trajs, theta, v], dim=-1) # (x, y, heading, vx, vy, w, l, h)

        return trajs

    def forward(self, trajs, current_states):
        trajs = self.state_process(trajs, current_states)
        trajs = self.mlp(trajs.detach())
        type = self.type_emb(current_states[:, :, None, 2].int())
        output = torch.max(trajs, dim=-2).values
        output = output + type

        return output

class SelfTransformer(nn.Module):
    def __init__(self):
        super(SelfTransformer, self).__init__()
        heads, dim, dropout = 8, 128, 0.1
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, inputs, mask=None):
        attention_output, _ = self.self_attention(inputs, inputs, inputs, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output + inputs)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


