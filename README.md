# turkish-game-rules-rag-app
It is a intelligent RAG App that answers the questions asked based on to the 2023-2024 game booklet published by TFF.

### Requirements

```
langchain
python-multipart
qdrant-client
torch
sentence_transformers
chromadb
pypdf
openai
```

## Usage
```bash
# Setup virtual environment
virtualenv venv && source venv/bin/activate && pip install -r requirements.txt

# For creating vector database, you need to run following Python script.
python ingest.py --model_name oguuzhansahin/bi-encoder-mnrl-dbmdz-bert-base-turkish-cased-margin_3.0-msmarco-tr-10k \
                 --data_dir YOUR_PDF_DATA_DIRECTORY
                 --persist_directory Directory to persist vector store

#Afterwward, you can run your Streamlit app by following Python script.
streamlit run app.py 

```
