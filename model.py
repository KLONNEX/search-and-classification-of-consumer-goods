from torch import nn
from transformers import BertForSequenceClassification


class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()



    def forward(self, input_id, mask):
        classifier_output = self.backbone(input_ids=input_id, attention_mask=mask, return_dict=False)
        print(classifier_output)
        return classifier_output
