from flask import Flask, request, jsonify
import json
import time
from dotenv import load_dotenv
from flask_cors import CORS
from kafka import KafkaConsumer, KafkaProducer
from pymongo import MongoClient

import getpass
from langchain.llms import OpenAI
from langchain.memory import ConversationEntityMemory,ConversationBufferMemory
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain import PromptTemplate

from langchain.memory.entity import InMemoryEntityStore

llm = OpenAI(temperature=0)
import os
import numpy as np


app = Flask(__name__)

load_dotenv()

# init kafka producer
KAFKA_SERVER = "localhost:9092"
producer = KafkaProducer(
    bootstrap_servers = KAFKA_SERVER,
    api_version = (0, 11, 15)
)

# Init QA chain
emb_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_n = HuggingFaceEmbeddings(
    model_name=emb_model,
    cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME')
)

client = MongoClient(os.getenv("MDB_CONNECTION_STR"))
db = client["search"]
col = client['sample']['kafkatest']

# docsearch = MongoDBAtlasVectorSearch(collection=col,embedding=embeddings_n,index_name="default",embedding_key="vec", text_key="searchf")
# retriever = docsearch.as_retriever()
# qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)


def get_emb_trnsformers():
    emb_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_w = HuggingFaceEmbeddings(
        model_name=emb_model,
        cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME')
    )
    return embeddings_w

def get_vector_store():
    col = db["catalog_final_myn"]
    vs = MongoDBAtlasVectorSearch(collection=col,embedding=get_emb_trnsformers(),index_name="default",embedding_key="vec", text_key="title")
    return vs

def get_conversation_chain(vectorstore,mem_key):
    llm = OpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key=mem_key, return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

@app.route('/pushToCollection', methods=['POST'])
def kafkaProducer():
    req = request.get_json()
    msg = req['message']
    topic_name = req['topic']
    msg["id"] = str(msg["id"])
    json_payload = json.dumps(msg)
    json_payload = str.encode(json_payload)
    # push data into respective mongocollection TOPIC
    producer.send(topic_name, json_payload)
    producer.flush()
    print("Sent to consumer")
    return jsonify({
        "message": f"{topic_name} is updated with the message", 
        "status": "Pass"})

@app.route('/qna', methods=['GET','POST'])
def get_qna():
    req = request.get_json()
    question = req['question']
    mem_key = req["user"]
    return get_conversation_chain(get_vector_store(), mem_key).run(question)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port = 5001)