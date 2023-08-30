from flask import Flask, request, jsonify, session
import json
import time
from dotenv import load_dotenv
from flask_cors import CORS
from kafka import KafkaConsumer, KafkaProducer
from pymongo import MongoClient
import langchain

import getpass
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationEntityMemory,ConversationBufferMemory
from langchain.docstore.document import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, ConversationChain,LLMChain
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain import PromptTemplate

from langchain.memory.entity import InMemoryEntityStore
from functools import lru_cache

import os
import numpy as np
from pathlib import Path


template='You are an assistant to a human, powered by a large language model trained by OpenAI.\n\nYou are designed to be able to assist with a wide range of fashion apparel, beauty products, fashion accessories and casual wear dresses, from answering simple questions to providing in-depth explanations and product recommendation to occassion , theme and situation on a wide range of product categories. As a language model, you are able to generate human-like text based on the input you receive, allowing you to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.\n\nYou are constantly learning and improving, and your capabilities are constantly evolving. You are able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. You have access to some personalized information provided by the human in the Context section below. Additionally, you are able to generate your own text based on the input you receive, allowing you to engage in discussions and provide explanations and descriptions on a wide range of topics.\n\nOverall, you are a powerful tool that can help with a wide range of tasks on this fashion ecommerce website and provide valuable insights and information on a wide range of product categories. Whether the human needs help with a specific question or just wants to have a conversation about a particular product/inventory, you are here to assist.\n\nContext:\nMyntra Products and Categories\n\nCurrent conversation:\n{chat_history}\nLast line:\nHuman: {question}\nYou:'

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = 'super secret key'
client = MongoClient(os.getenv("MDB_CONNECTION_STR"))
db = client["search"]
col = client['sample']['kafkatest']

@lru_cache
def get_openai_emb_transformers():
    embeddings = OpenAIEmbeddings()
    return embeddings

def get_emb_trnsformers():
    emb_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_w = HuggingFaceEmbeddings(
        model_name=emb_model,
        cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME')
    )
    return embeddings_w

@lru_cache
def get_vector_store():
    col = db["catalog_final_myn"]
    vs = MongoDBAtlasVectorSearch(collection=col,embedding=get_openai_emb_transformers(),index_name="default",embedding_key="openAIVec", text_key="title")
    return vs

@lru_cache
def get_conversation_chain(vectorstore):
    llm = OpenAI(temperature=0.2)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(template=template,input_variables=["chat_history", "question"])
    )
    return conversation_chain

@lru_cache
def get_conversation_chain_rag(vectorstore):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.7)
    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 25})
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    return qa_chain

def similarity_search(question,filter=[{"query":"Men", "path": "gender"}],k=25):
    collection = client['search']['catalog_final_myn']
    query_vector = get_openai_emb_transformers().embed_query(question)
    knnBeta = {
      "vector": query_vector,
      "path": "openAIVec",
      "k": k
    }
    if len(filter)>0:
        compound = {}
        compound["must"] = []
        for fil in filter:
            compound["must"]  += [{"text":fil}]
        knnBeta["filter"] = {"compound": compound}
    pipeline = [{
    "$search": {
    "index": "default",
    "knnBeta": knnBeta
    }
    },
    {"$addFields":{
    "score":{
        '$meta': 'searchScore'
        }
    }},
    {"$match":{
      "score":{
          "$gt":0.8
      }
    }},
    {"$project":{
      "_id": 0,
      "id": 1,
      "title":1,
      "price":1,
      "atp": 1,
      "baseColour":1,
      "gender":1,
      "mfg_brand_name":1,
      "link":1,
      "score":1
    }}]
    res = list(collection.aggregate(pipeline))
    return res

def preprocess_ip_query(text):
    text = text.replace("product", "fashion clothes")
    text = text.replace("products", "fashion clothes")
    return text


# init kafka producer
KAFKA_SERVER = "localhost:9092"
producer = KafkaProducer(
    bootstrap_servers = KAFKA_SERVER,
    api_version = (0, 11, 15)
)

# Api call to publish to kafka
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

@app.route("/similar", methods=["GET"])
def get_similar():
    req = request.get_json()
    q = req["query"]
    return jsonify(similarity_search(q))

@app.route('/qna', methods=['GET','POST'])
def get_qna():
    req = request.get_json()
    question = req['question']
    question = preprocess_ip_query(question)
    mem_key = req["user"].split(".com")[0]
    # session memory for chat history
    if mem_key+"_chat_history" not in session:
        try:
            session.pop(mem_key+"_chat_history", default=None)
        except:
            print("Clearing session")
        session[mem_key+"_chat_history"] = []
    else:
        print("Using the following memory key", mem_key)
    
    # conversation handler
    resp = get_conversation_chain_rag(get_vector_store()).run({"question":question, "chat_history":session[mem_key+"_chat_history"]})
    op = {}
    session[mem_key+"_chat_history"] += [(question, resp)]
    op["response"] = resp
    op["history"] = session[mem_key+"_chat_history"]

    # Ecomm query generator for recommendations
    prompt = f"IF there is a intent in the below context return items that match \n ##Context: {resp} \n ##Instruction: Generate Search Query to use in Search engine containing Fashion clother and accessories with try to decipher the gender and brand preference from the context and use it \n ## Generate Search Query: "
    query = get_conversation_chain_rag(get_vector_store()).run({"question":prompt, "chat_history":session[mem_key+"_chat_history"]})
    products = similarity_search(query)
    op["recommendations"] = products
    op["product_query"] = query
    return jsonify(op)

@app.route('/clear/session', methods=['DELETE'])
def clean_session():
    session.clear()
    return "Session Cleared"

if __name__ == "__main__":
    dotenv_path = Path('.env')
    load_dotenv()
    app.run(debug=True, host='0.0.0.0',port = 5001)