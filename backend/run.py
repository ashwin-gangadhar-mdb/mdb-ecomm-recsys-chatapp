from flask import Flask, request, jsonify, session
import json
from dotenv import load_dotenv
from flask_cors import CORS
from kafka import KafkaProducer
from pymongo import MongoClient

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.memory import ConversationBufferWindowMemory,ConversationSummaryBufferMemory
from langchain.chains import  ConversationalRetrievalChain, ConversationChain,LLMChain

from functools import lru_cache
import certifi

import os
import numpy as np
from pathlib import Path

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


def get_conversation_chain_rag(filter_query):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.7)
    retriever = get_vector_store().as_retriever(search_type='similarity', search_kwargs={'k': 200, "pre_filter" : {"compound": {"must": filter_query}}})
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5)
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    return qa_chain

@lru_cache
def get_conversation_chain_conv():
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.7)
    chain = ConversationChain(llm=llm, memory=ConversationBufferWindowMemory(k=5))
    return chain

def similarity_search(question,filter=[{"text":{"query":"Men", "path": "gender"}}],k=200):
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
        "id": {"$first": "$id"},
        "title": {"$first": "$title"},
        "price": {"$first": "$price"},
        "atp": {"$first": "$atp"},
        "baseColour": {"$addToSet": "$baseColour"},
        "gender": {"$addToSet": "$gender"},
        "mfg_brand_name": {"$first": "$mfg_brand_name"},
        "link": {"$first": "$link"},
        "articleType": {"$first": "$articleType"},
        "score": {"$avg": "$score"}
    }},
    {"$sort": {"score": -1}},
    {"$project":{"_id": 0}}]
    res = list(collection.aggregate(pipeline))
    buckets = {}
    for ele in res:
        if ele["articleType"] in buckets:
            buckets[ele["articleType"]] += [ele]
        else:
            buckets[ele["articleType"]] = [ele]
    counter = {}
    op = []
    for k in buckets.keys():
        op += buckets[k][:5]
    op = sorted(op,key=lambda ele: -1*ele["score"])
    return op

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
    if "history" in req:
        history = req["history"]
    else:
        history = []
    mem_key = req["user"].split(".com")[0]
    user_profile = get_user_profile(mem_key)

    # chat history
    if type(history)==list and len(history)>0:
        session[mem_key+"_chat_history"] = history 
    else:
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

    # prepare user profile filter query
    filter_query = []
    if len(user_profile.keys())>0 and "sex" in user_profile:
        filter_query += [{"text": {"query": user_profile["sex"], "path": "gender"}}]
    if len(user_profile.keys())>0 and "age_group" in user_profile:
        filter_query += [{"text": {"query": user_profile["age_group"], "path": "ageGroup"}}]

    # initiate conversaiton
    if len(session[mem_key+"_chat_history"]) < 1 or len(session[mem_key+"_chat_history"])==4:
        resp = get_conversation_chain_rag(filter_query).run({"question":f"Greetings!!! I am {name} and I am {gender} and live in {address}", "chat_history":session[mem_key+"_chat_history"]})
        session[mem_key+"_chat_history"] += [(f"Greetings!!! I am {name} and I am looking for {gender}s products to purchase and live in {address}", resp)]
    
    # conversation handler 
    lllm = get_conversation_chain_rag(filter_query)
    resp = lllm.run({"question":question, "chat_history":session[mem_key+"_chat_history"]})
    op = {}
    session[mem_key+"_chat_history"] += [(question, resp)]
    op["response"] = resp
    op["history"] = session[mem_key+"_chat_history"]

    # Ecomm query generator for recommendations   
    check_query_prompt = f"Given the response from the LLM from previous stage. Can we use this reponse to query The search engine. Answer with Yes or No only \n ##Response from LLM: {resp}"
    check = get_conversation_chain_conv().predict(input=check_query_prompt, history=op["history"]) # .predict({"input":check_query_prompt, "history":[]})
    query = parse_query(resp)
    print("Result of the LLM check for recommendations",check)
    if "YES" in check.upper():
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