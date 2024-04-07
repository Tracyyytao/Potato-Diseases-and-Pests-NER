import torch.nn as nn
from config import *
from torchcrf import CRF
import torch
from transformers import BertModel
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, WORD_PAD_ID)
        self.bert = BertModel.from_pretrained(BERT_MODEL, output_hidden_states=True)

        self.conv1 = nn.Conv1d(1536, 768, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(768, 768, kernel_size=3, padding=1)

        self.conv3 = nn.Conv1d(1536,768, kernel_size=1)

        self.norm = nn.LayerNorm(768)
        #self.norm = nn.BatchNorm1d(768)


        self.lstm = nn.LSTM(
            768,
            HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Linear(2*HIDDEN_SIZE, TARGET_SIZE)
        #self.drop = nn.Dropout(0.1)
        self.crf = CRF(TARGET_SIZE, batch_first=True)

    def _get_lstm_feature(self, input, mask):
        # out = self.embed(input)
        bert_outputs = self.bert(input.to(self.device), mask.to(self.device))
        bert_hidden_states = bert_outputs.hidden_states

        # Concatenate BERT hidden states starting from the second layer
        concatenated_output = bert_hidden_states[1].to(self.device)
        for i in range(2, 13):
            layer_output = bert_hidden_states[i].to(self.device)
            concatenated_output = torch.cat((concatenated_output, layer_output), dim=-1)
            a = concatenated_output

            concatenated_output = F.relu(self.norm(self.conv1(concatenated_output.permute(0, 2, 1)).permute(0, 2, 1)))
            concatenated_output = self.norm(self.conv2(concatenated_output.permute(0, 2, 1)).permute(0, 2, 1))

            concatenated_output = F.relu(self.conv3(a.permute(0,2,1)).permute(0,2,1) + concatenated_output)


        # LSTM processing
        lstm_out, _ = self.lstm(concatenated_output)
        return self.linear(lstm_out)

    def forward(self, input, mask):
        out = self._get_lstm_feature(input, mask)
        return self.crf.decode(out, mask)

    def loss_fn(self, input, target, mask):
        y_pred = self._get_lstm_feature(input, mask)
        return -self.crf.forward(y_pred, target, mask, reduction='mean')

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # tensor.clone()会创建一个与被clone的tensor完全一样的tensor，两者不共享内存但是新tensor仍保存在计算图中，即新的tensor仍会被autograd追踪
                # 这里是在备份
                self.backup[name] = param.data.clone()
                # 归一化
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

if __name__ == '__main__':
    model = Model()
    print(model)
    exit()

