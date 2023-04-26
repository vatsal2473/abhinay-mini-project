import pandas as pd
from flask_cors import CORS
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
import torch
from sentence_transformers import SentenceTransformer
import pinecone

pinecone.init(
    api_key="0c344797-0cd5-4532-90c6-91091f7f1c23",
    environment="us-east1-gcp"  # find next to API key in console
)

index_name = "abhinay-mini-project"

index = pinecone.Index(index_name)

df = pd.read_csv('data.csv')
df['Merged'] = df.apply(lambda row: ', '.join(row.values.astype(str)), axis=1)


# set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load the retriever model from huggingface model hub
retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base", device=device)
retriever

app = Flask(__name__)
CORS(app)

def query_pinecone(query, top_k):
    # generate embeddings for the query
    xq = retriever.encode([query]).tolist()
    # search pinecone index for context passage with the answer
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    return xc

@app.route('/insert_new_data', methods=['GET', 'POST'])
def insert_new_data():
    if request.method == "POST":
        new_data = request.form.get('new_data')

    
    unique_id = index.describe_index_stats()["total_vector_count"]
    meta = dict(zip(df.columns, new_data.split(",")))
    emb = retriever.encode([new_data]).tolist()
    to_upsert = [(str(unique_id), emb[0], meta)]
    _ = index.upsert(vectors=to_upsert)
    # index.describe_index_stats()
    return "success"

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == "POST":
        new_data = request.form.get('data')

    print(new_data)
    result = query_pinecone(new_data, top_k=10)

    matches = result['matches']

    data = [{'id': match['id'],
            'brand': match['metadata']['brand'],
            'features': match['metadata']['features'],
            'gear': match['metadata']['gear'],
            'price': match['metadata']['price'],
            'title': match['metadata']['title'],
            'score': match['score']} for match in matches]
    
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)