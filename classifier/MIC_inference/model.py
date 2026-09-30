from transformers import BertModel
import torch.nn as nn

from config import PROTBERT_DIR

local_model_path = str(PROTBERT_DIR)

class REG(nn.Module):
    def __init__(self):
        super(REG, self).__init__()
        self.bert = BertModel.from_pretrained(local_model_path, output_attentions=True)
        self.regressor= nn.Sequential(nn.LayerNorm(self.bert.config.hidden_size),
                                      #
                                      nn.Linear(self.bert.config.hidden_size, 512),
                                      nn.LeakyReLU(inplace=False),
                                      nn.Dropout(p=0.2),
                                      #
                                      nn.Linear(512, 128),
                                      nn.LeakyReLU(inplace=False),
                                      nn.Dropout(p=0.2),

                                      nn.Linear(128, 1))

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.regressor(output.pooler_output), self.bert
