import sys

sys.path.append('..')
import fewshot_re_kit
import torch
from torch import nn
from torch.nn import functional as F


class HCRP(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, hidden_size, max_len):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.rel_glo_linear = nn.Linear(hidden_size, hidden_size * 2)
        self.temp_proto = 1  # moco 0.07

    def __dist__(self, x, y, dim):
        return (x * y).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, rel_text, N, K, total_Q, is_eval=False):
        """
        :param support: Inputs of the support set. (B*N*K)
        :param query: Inputs of the query set. (B*total_Q)
        :param rel_text: Inputs of the relation description.  (B*N)
        :param N: Num of classes
        :param K: Num of instances for each class in the support set
        :param total_Q: Num of instances in the query set
        :param is_eval:
        :return: logits, pred, logits_proto, labels_proto, sim_scalar
        """
        support_glo, support_loc = self.sentence_encoder(support)  # (B * N * K, 2D), (B * N * K, L, D)
        query_glo, query_loc = self.sentence_encoder(query)  # (B * total_Q, 2D), (B * total_Q, L, D)
        rel_text_glo, rel_text_loc = self.sentence_encoder(rel_text, cat=False)  # (B * N, D), (B * N, L, D)

        # global features
        support_glo = support_glo.view(-1, N, K, self.hidden_size * 2)  # (B, N, K, 2D)
        query_glo = query_glo.view(-1, total_Q, self.hidden_size * 2)  # (B, total_Q, 2D)
        rel_text_glo = self.rel_glo_linear(rel_text_glo.view(-1, N, self.hidden_size))  # (B, N, 2D)

        B = support_glo.shape[0]

        # global prototypes
        proto_glo = torch.mean(support_glo, 2) + rel_text_glo  # Calculate prototype for each class (B, N, 2D)

        # local features
        rel_text_loc_s = rel_text_loc.unsqueeze(1).expand(-1, K, -1, -1).contiguous().view(B * N * K, -1, self.hidden_size)  # (B * N * K, L, D)
        rel_support = torch.bmm(support_loc, torch.transpose(rel_text_loc_s, 2, 1))  # (B * N * K, L, L)
        ins_att_score_s, _ = rel_support.max(-1)  # (B * N * K, L)

        ins_att_score_s = F.softmax(torch.tanh(ins_att_score_s), dim=1).unsqueeze(-1)  # (B * N * K, L, 1)
        support_loc = torch.sum(ins_att_score_s * support_loc, dim=1)  # (B * N * K, D)
        support_loc = support_loc.view(B, N, K, self.hidden_size)

        ins_att_score_r, _ = rel_support.max(1)  # (B * N * K, L)
        ins_att_score_r = F.softmax(torch.tanh(ins_att_score_r), dim=1).unsqueeze(-1)  # (B * N * K, L, 1)
        rel_text_loc = torch.sum(ins_att_score_r * rel_text_loc_s, dim=1).view(B, N, K, self.hidden_size)
        rel_text_loc = torch.mean(rel_text_loc, 2)  # (B, N, D)

        query_query = torch.bmm(query_loc, torch.transpose(query_loc, 2, 1))  # (B * total_Q, L, L)
        ins_att_score_q, _ = query_query.max(-1)  # (B * total_Q, L)
        ins_att_score_q = F.softmax(torch.tanh(ins_att_score_q), dim=1).unsqueeze(-1)  # (B * total_Q, L, 1)
        query_loc = torch.sum(ins_att_score_q * query_loc, dim=1)  # (B * total_Q, D)
        query_loc = query_loc.view(B, total_Q, self.hidden_size)  # (B, total_Q, D)

        # local prototypes
        proto_loc = torch.mean(support_loc, 2) + rel_text_loc  # (B, N, D)

        # hybrid prototype
        proto_hyb = torch.cat((proto_glo, proto_loc), dim=-1)  # (B, N, 3D)
        query_hyb = torch.cat((query_glo, query_loc), dim=-1)  # (B, total_Q, 3D)
        rel_text_hyb = torch.cat((rel_text_glo, rel_text_loc), dim=-1)  # (B, N, 3D)

        logits = self.__batch_dist__(proto_hyb, query_hyb)  # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)

        logits_proto, labels_proto, sim_scalar = None, None, None

        if not is_eval:
            # relation-prototype contrastive learning
            # # relation as anchor
            rel_text_anchor = rel_text_hyb.view(B * N, -1).unsqueeze(1)  # (B * N, 1, 3D)

            # select positive prototypes
            proto_hyb = proto_hyb.view(B * N, -1)  # (B * N, 3D)
            pos_proto_hyb = proto_hyb.unsqueeze(1)  # (B * N, 1, 3D)

            # select negative prototypes
            neg_index = torch.zeros(B, N, N - 1)  # (B, N, N - 1)
            for b in range(B):
                for i in range(N):
                    index_ori = [i for i in range(b * N, (b + 1) * N)]
                    index_ori.pop(i)
                    neg_index[b, i] = torch.tensor(index_ori)

            neg_index = neg_index.long().view(-1).cuda()  # (B * N * (N - 1))
            neg_proto_hyb = torch.index_select(proto_hyb, dim=0, index=neg_index).view(B * N, N - 1, -1)

            # compute prototypical logits
            proto_selected = torch.cat((pos_proto_hyb, neg_proto_hyb), dim=1)  # (B * N, N, 3D)
            logits_proto = self.__batch_dist__(proto_selected, rel_text_anchor).squeeze(1)  # (B * N, N)
            logits_proto /= self.temp_proto  # scaling temperatures for the selected prototypes

            # targets
            labels_proto = torch.cat((torch.ones(B * N, 1), torch.zeros(B * N, N - 1)), dim=-1).cuda()  # (B * N, 2N)
            
            # task similarity scalar
            features_sim = torch.cat((proto_hyb.view(B, N, -1), rel_text_hyb), dim=-1)
            features_sim = self.l2norm(features_sim)
            sim_task = torch.bmm(features_sim, torch.transpose(features_sim, 2, 1))  # (B, N, N)
            sim_scalar = torch.norm(sim_task, dim=(1, 2))  # (B)
            sim_scalar = torch.softmax(sim_scalar, dim=-1)
            sim_scalar = sim_scalar.repeat(total_Q, 1).t().reshape(-1)  # (B*totalQ)

        return logits, pred, logits_proto, labels_proto, sim_scalar
