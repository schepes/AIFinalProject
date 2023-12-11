from transformers import BertTokenizer
import torch

class BertPreprocessor:
    def __init__(self, model_name='bert-base-uncased', max_length=256):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def preprocess(self, texts):
        encoding = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            return_attention_mask=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'  # Return PyTorch tensors
        )
        return encoding['input_ids'], encoding['attention_mask']
