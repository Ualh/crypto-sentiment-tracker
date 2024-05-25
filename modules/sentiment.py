from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch

class SentimentAnalyzer:
    def __init__(self, model_name='ProsusAI/finbert'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        if torch.cuda.is_available():
                    self.model.cuda()

    def predict_sentiment_batch(self, texts, batch_size=16):
            self.model.eval()
            sentiments = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    sentiments.extend(predictions[:, 0].detach().cpu().numpy())
            return sentiments

    def add_sentiments_to_df(self, df, text_column='description'):
            df['sentiment'] = self.predict_sentiment_batch(df[text_column].dropna().tolist())
            return df