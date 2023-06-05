import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from streamlit_extras.stateful_button import button
# tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
# model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

@st.cache_resource
def load():
    tokenizer = AutoTokenizer.from_pretrained("/home/jbasa/Desktop/Jero/SA/tokenizer/")
    model = AutoModelForSequenceClassification.from_pretrained("/home/jbasa/Desktop/Jero/SA/model/")
    return tokenizer, model

tokenizer, model = load()

def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1

text = st.text_area("Escriba un comentario","",height=20)

if button("Analizar",key="button"):
    st.write("El comentario recibió:", sentiment_score(text), "estrellas")