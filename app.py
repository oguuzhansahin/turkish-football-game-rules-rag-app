import streamlit as st
import logging
import os
import sys

from typing import Any, Dict, Generator, List, Union
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

ResponseType = Union[Generator[Any, None, None], Any, List, Dict]

OPENAI_KEY = ""

llm = ChatOpenAI(openai_api_key= OPENAI_KEY)

model_name = "oguuzhansahin/bi-encoder-mnrl-dbmdz-bert-base-turkish-cased-margin_3.0-msmarco-tr-10k"
persist_directory = "stores/futbol_kurallari"
model_kwargs = {"device":"cpu"}
encode_kwargs = {"normalize_embeddings":False}

embeddings = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)


@st.cache_resource(show_spinner=False)  # type: ignore[misc]
def load_index() -> Any:
    """Load the index from the storage directory."""
    print("Loading index...")

    # rebuild storage context
    load_vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    query_engine  = load_vector_store.as_retriever(search_kwargs={"k": 1})
    return query_engine


@st.cache_resource(show_spinner=False)  # type: ignore[misc]
def get_response(context, query) -> Any:

    prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """

    prompt_template = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt_template.format(context=context, question=query)

    response_text = llm.predict(prompt)
    return response_text


def main() -> None:
    print("""Run the chatbot.""")
    st.session_state.messages = []
    
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = load_index()
        
    st.title("2023-2024 Futbol Oyun Kuralları Kitabıyla Konuş!!")
    st.write("Özellikle hakem hatalarının artmasıyla beraber kuralların doğru uygulanabilmesi için doğru soruları sor!")
    st.image("static/hakem-hatalari.jpg")
    if "messages" not in st.session_state:
        system_prompt = (
            "Your purpose is to answer questions about specific documents only. "
            "Please answer the user's questions based on what you know about the document. "
            "If the question is outside scope of the document, please politely decline. "
            "If you don't know the answer, say `I don't know`. "
        )
        st.session_state.messages = [{"role": "system", "content": system_prompt}]

    for message in st.session_state.messages:
        if message["role"] not in ["user", "assistant"]:
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            print("Querying query engine API...")
            relevant_doc = st.session_state.query_engine.get_relevant_documents(prompt)
            print("relevant doc: ", relevant_doc)
            response = get_response(context= relevant_doc[0].page_content, query=prompt)
            full_response = f"{response}"
            #print(full_response)
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()