from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import MarianMTModel, MarianTokenizer
from scipy.special import softmax
import numpy as np

class AIPipeline:
    def __init__(self):
        # Initialize sentiment analysis model
        self.sentiment_model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
        self.sentiment_config = AutoConfig.from_pretrained(self.sentiment_model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_name)

    @staticmethod
    def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    @staticmethod
    def translate_to_english_or_french(text, source_lang):
        if source_lang in ['en', 'fr']:
            return text  # No translation needed
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-en"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    def evaluate_sentiment(self, text):
        text = self.preprocess(text)
        encoded_input = self.sentiment_tokenizer(text, return_tensors='pt')
        output = self.sentiment_model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        # Get labels and scores
        ranking = np.argsort(scores)[::-1]
        results = []
        for i in range(scores.shape[0]):
            label = self.sentiment_config.id2label[ranking[i]]
            score = scores[ranking[i]]
            results.append((label, np.round(float(score), 4)))
        return results

    def process(self, comment, source_lang):
        # Step 1: Translate if necessary
        comment = self.translate_to_english_or_french(comment, source_lang)
        # Step 2: Evaluate sentiment
        sentiment_results = self.evaluate_sentiment(comment)
        return sentiment_results

# Example usage
if __name__ == "__main__":
    pipeline = AIPipeline()
    comment = "Je suis triste"  # Example comment
    source_lang = "fr"  # Language of the comment
    result = pipeline.process(comment, source_lang)
    print(result)
