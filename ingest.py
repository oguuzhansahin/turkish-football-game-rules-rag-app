import os
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for processing documents and creating vector stores.")

    parser.add_argument("--model_name", type=str, default="oguuzhansahin/bi-encoder-mnrl-dbmdz-bert-base-turkish-cased-margin_3.0-msmarco-tr-10k",
                        help="Hugging Face model name for embeddings.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for Hugging Face model (cpu or cuda).")
    parser.add_argument("--normalize_embeddings", action="store_true", help="Normalize embeddings or not.")
    parser.add_argument("--data_dir", type=str, default="data/", help="Directory containing input documents.")
    parser.add_argument("--use_splitter", action="store_true", help="Use splitter for recursive text splitter")
    parser.add_argument("--chunk_size", type=int, default=None, help="Chunk size for splitting text.")
    parser.add_argument("--chunk_overlap", type=int, default=None)
    parser.add_argument("--persist_directory", type=str, default="stores/futbol_kurallari_normalized",
                        help="Directory to persist vector store.")

    return parser.parse_args()


def main():
    
    args = parse_arguments()

    model_kwargs = {"device":args.device}
    encode_kwargs = {"normalize_embeddings":args.normalize_embeddings}

    embeddings = HuggingFaceEmbeddings(
        model_name = args.model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )

    print("Embeddings:", embeddings)

    loader = DirectoryLoader(
        args.data_dir, 
        show_progress=True, 
        loader_cls=PyPDFLoader
    )
    
    document = loader.load()

    if args.use_splitter:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, 
                                                       chunk_overlap=args.chunk_overlap)
        document = text_splitter.split_documents(document)
        
    vector_store = Chroma.from_documents(
        document, 
        embeddings, 
        collection_metadata = {"hnsw:space":"cosine"}, 
        persist_directory = args.persist_directory
    )
    
    print("Vector store: ", vector_store)
    
if __name__ == "__main__":
    main()