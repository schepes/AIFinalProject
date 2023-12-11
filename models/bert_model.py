from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torch.nn import BCEWithLogitsLoss

class BertModel:
    def __init__(self, model_name='bert-base-uncased'):
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, input_ids, attention_masks, labels, batch_size=32, epochs=4):
        dataset = TensorDataset(input_ids, attention_masks, labels)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        loss_fct = BCEWithLogitsLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_dataloader:
                b_input_ids, b_input_mask, b_labels = [t.to(self.device) for t in batch]
                self.model.zero_grad()

                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = outputs.logits

                
                b_labels = b_labels.unsqueeze(1)  
                b_labels = torch.cat((1 - b_labels, b_labels), dim=1) 

                loss = loss_fct(logits, b_labels.float())
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.2f}")

            self.model.eval()
            total_eval_accuracy = 0
            for batch in val_dataloader:
                b_input_ids, b_input_mask, b_labels = [t.to(self.device) for t in batch]
                with torch.no_grad():
                    outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = outputs.logits
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                predictions = np.argmax(logits, axis=1)
                total_eval_accuracy += accuracy_score(label_ids, predictions)

            avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
            print(f"Validation Accuracy: {avg_val_accuracy:.2f}")

    def evaluate(self, input_ids, attention_masks, labels, batch_size=32):
        test_dataset = TensorDataset(input_ids, attention_masks, labels)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        total_eval_accuracy = 0
        predictions, true_labels = [], []

        for batch in test_dataloader:
            b_input_ids, b_input_mask, b_labels = [t.to(self.device) for t in batch]
            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            batch_predictions = np.argmax(logits, axis=1)
            predictions.extend(batch_predictions)
            true_labels.extend(label_ids)

        print("Accuracy:", accuracy_score(true_labels, predictions))
        print("Classification Report:")
        print(classification_report(true_labels, predictions))
