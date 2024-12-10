# model_manager.py
import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import pickle

MODEL_DIR = "models"

def initialize_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Initialize and save QA model
    qa_model_path = os.path.join(MODEL_DIR, "qa_model")
    if not os.path.exists(qa_model_path):
        qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        qa_model.save_pretrained(qa_model_path)
    
    # Initialize and save sentence transformer
    st_model_path = os.path.join(MODEL_DIR, "sentence_transformer")
    if not os.path.exists(st_model_path):
        st_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        st_model.save(st_model_path)

def load_models():
    qa_model = pipeline("question-answering", model=os.path.join(MODEL_DIR, "qa_model"))
    st_model = SentenceTransformer(os.path.join(MODEL_DIR, "sentence_transformer"))
    return qa_model, st_model