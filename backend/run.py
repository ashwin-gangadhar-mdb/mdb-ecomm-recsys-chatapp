from flask import Flask, request, jsonify, session
import json
import time
from dotenv import load_dotenv
from flask_cors import CORS
from kafka import KafkaProducer
from pymongo import MongoClient
import langchain

import getpass
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import  ConversationalRetrievalChain, ConversationChain,LLMChain
from langchain import PromptTemplate

from functools import lru_cache
import certifi

import os
import numpy as np
from pathlib import Path


template='You are an assistant to a human, powered by a large language model trained by OpenAI.\n\nYou are designed to be able to assist with a wide range of fashion clothes, beauty products, fashion accessories and casual wear dresses, from answering simple questions to providing in-depth explanations and product recommendation to occassion , theme and situation on a wide range of product categories. As a language model, you are able to generate human-like text based on the input you receive, allowing you to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.\n\nYou are constantly learning and improving, and your capabilities are constantly evolving. You are able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. You have access to some personalized information provided by the human in the Context section below. Additionally, you are able to generate your own text based on the input you receive, allowing you to engage in discussions and provide explanations and descriptions on a wide range of topics.\n\nOverall, you are a powerful tool that can help with a wide range of tasks on this fashion ecommerce website and provide valuable insights and information on a wide range of product categories. When asked to purchase product instruct use to engage with the cards to make a purchase. Whether the human needs help with a specific question or just wants to have a conversation about a particular product/inventory, you are here to assist and recommend products taking my gender into account.\n\nContext:\nMyntra Products and Categories\n \nCurrent conversation:\n{chat_history}\nLast line:\nHuman: {question} and my gender is {gender}\nYou:'

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = 'super secret key'
client = MongoClient(os.getenv("MDB_CONNECTION_STR"), tlsCAFile=certifi.where())
db = client["search"]
col = client['sample']['kafkatest']

@lru_cache
def get_openai_emb_transformers():
    embeddings = OpenAIEmbeddings()
    return embeddings

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

@lru_cache
def get_conversation_chain_conv():
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.7)
    chain = ConversationChain(llm=llm)
    return chain

def similarity_search(question,filter=[{"text":{"query":"Men", "path": "gender"}}],k=150):
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
            compound["must"]  += [fil]
        knnBeta["filter"] = {"compound": compound}
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
    {"$group":{
        "_id": "$title",
        "id": {"$first": "$title"},
        "title": {"$first": "$title"},
        "price": {"$first": "$price"},
        "atp": {"$first": "$atp"},
        "baseColour": {"$addToSet": "$baseColour"},
        "gender": {"$addToSet": "$gender"},
        "mfg_brand_name": {"$first": "$mfg_brand_name"},
        "link": {"$addToSet": "$link"},
        "score": {"$avg": "$score"}
    }},
    {"$project":{"_id": 0}}]
    res = list(collection.aggregate(pipeline))
    return res

def get_user_profile(uid):
    collection = client['search']['UserProfile']
    res = list(collection.find({"email": uid+".com"},{"_id": 0 , "first_name": 1, "last_name": 1, "address": 1, "sex": 1}))
    op = {}
    if len(res)>0:
        op = res[0]
        if op["sex"] == "Male":
            op["sex"] = "Men"
        elif op["sex"] == "Female":
            op["sex"] = "Women"
    return op

def preprocess_ip_query(text):
    text = text.replace("product", "fashion clothes")
    text = text.replace("products", "fashion clothes")
    text = text.replace("pruchase", "recommend products")
    text = text.replace("buy", "recommend products")
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
    return jsonify(similarity_search(q,filter=[]))

def parse_query(val):
    op = ""
    foo = []
    if len(val.split("\n")[1:-1])>1:
        foo = val.split("\n")[1:-1]
    elif len(val.split(",")[1:-1])>1:
        foo = val.split(",")[1:-1]
    if len(foo)>0:
        for i,ele in enumerate(foo):
            if ele !="":
                op += " "+ele.replace(str(i)+".","").strip()
        return op
    else:
        return val
    
@app.route('/qna', methods=['GET','POST'])
def get_qna():
    req = request.get_json()
    question = req['question']
    question = preprocess_ip_query(question)
    mem_key = req["user"].split(".com")[0]
    user_profile = get_user_profile(mem_key)
    # session memory for chat history
    if mem_key+"_chat_history" not in session:
        try:
            session.pop(mem_key+"_chat_history", default=None)
        except:
            print("Clearing session")
        session[mem_key+"_chat_history"] = []
    else:
        print("Using the following memory key", mem_key)

    
    # restric history to last three messages
    if len(session[mem_key+"_chat_history"])>4:
        session[mem_key+"_chat_history"] = session[mem_key+"_chat_history"][-4:]
    
    if "sex" in user_profile:
        gender = user_profile["sex"]
    else:
        gender = ""
    
    if "first_name" in user_profile:
        name = user_profile["first_name"]
    else:
        name = "Guest"
    
    if "address" in user_profile:
        address = user_profile["address"]
    else:
        address = ""

    # initiate conversaiton
    if len(session[mem_key+"_chat_history"]) < 1 or len(session[mem_key+"_chat_history"])==4:
        resp = get_conversation_chain_rag(get_vector_store()).run({"question":f"Greetings!!! I am {name} and I am {gender} and live in {address}", "chat_history":session[mem_key+"_chat_history"], "gender": gender})
        session[mem_key+"_chat_history"] += [(f"Greetings!!! I am {name} and I am looking for {gender}s products to purchase and live in {address}", resp)]
    # conversation handler
    resp = get_conversation_chain_rag(get_vector_store()).run({"question":question, "chat_history":session[mem_key+"_chat_history"], "gender": gender})
    op = {}
    session[mem_key+"_chat_history"] += [(question, resp)]
    op["response"] = resp
    op["history"] = session[mem_key+"_chat_history"]

    # Ecomm query generator for recommendations
    prompt = f"Identify the top keywords related to fashion e-commerce that will drive the most relevant traffic to our website and increase search engine visibility. Gather data on search volume, competition, and related keywords. The keywords should be relevant to our target audience and align with our content marketing strategy. Give SEO or product queries only as output not descriptive suggestion on products. Pick the keywords from the context below and give only 5 search queries as output \n ##Context: {resp}"
    query = get_conversation_chain_conv().predict(input=prompt) ##.run({"question":prompt, "chat_history":session[mem_key+"_chat_history"]})
    
    check_query_prompt = f"Given the response from the LLM from previous stage. Can we use this reponse to query The search engine. Answer with Yes or No only \n ##Context: {query}"
    check = get_conversation_chain_conv().predict(input=check_query_prompt)
    query = parse_query(query)
    if "YES" in check.upper():
        filter_query = []
        # if len(user_profile.keys())>0 and "sex" in user_profile:
        #     filter_query += [{"text": {"query": user_profile["sex"], "path": "gender"}}]
        products = similarity_search(query,filter=filter_query)
        op["recommendations"] = products
        op["product_query"] = query
    else:
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