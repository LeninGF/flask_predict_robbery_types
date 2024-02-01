# from transformers import TFAutoModelForSequenceClassification
from transformers import pipeline
from transformers import AutoTokenizer
import re


class RobberyAi:
    PATH_MODEL = '/home/falconiel/ML_Models/robbery_tf20221113'
    # model_tf = (TFAutoModelForSequenceClassification.from_pretrained(PATH_MODEL)) # para usar este modo hay que previamente tokenizar y dar formato de tensor al texto
    model_ckpt = "distilbert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    classifier = pipeline("text-classification", model=PATH_MODEL, tokenizer=tokenizer)
    SEQ_LEN = 300
    
    def __init__(self, data):
        self.text = data["text"]
    @classmethod 
    def predict(cls, data):
        # adding text formatting for data, according to operations during training. Do I receive one by one or a list of texts?
        data_processed = cls.preprocess_text(data)
        y_hat = cls.classifier(data_processed, truncation=True)
        return y_hat
    
    @classmethod
    def load_model_tf(cls):
        return pipeline("text-classification", model=cls.PATH_MODEL, tokenizer=cls.tokenizer)
    @classmethod
    def tokenize(cls, batch):
        return cls.tokenizer(batch["relato"], padding="max_length", truncation=True, max_length=cls.SEQ_LEN)
    @classmethod
    def tokenize_through_pipe(cls, data):
        return cls.classifier.tokenizer(data, truncation=True)
    @staticmethod
    def preprocess_text(atext):
        text_processed = atext.lower()
        text_processed = re.sub("[^A-Za-z0-9áéíóúñ]+", " ", text_processed)
        text_processed = text_processed.strip()
        return text_processed
    