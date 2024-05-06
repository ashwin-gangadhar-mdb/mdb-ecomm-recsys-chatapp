from flask import Flask, request, jsonify, session
import json
from dotenv import load_dotenv
from flask_cors import CORS
from kafka import KafkaProducer
from pymongo import MongoClient

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory

from reco_util import get_product_reco_status, get_product_recommendations, similarity_search, get_conversation_chain_conv

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


def summarize(chat_history: str):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)


@lru_cache
def get_openai_emb_transformers():
    embeddings = OpenAIEmbeddings()
    return embeddings

@lru_cache
def get_vector_store():
    col = db["catalog_final_myn"]
    vs = MongoDBAtlasVectorSearch(collection=col,embedding=get_openai_emb_transformers(),index_name="vector_index",embedding_key="openAIVec", text_key="title")
    return vs

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

def get_sorted_results(product_recommendations):
    all_titles = [rec['title'] for rec in product_recommendations['products']]
    col = db["catalog_final_myn"]
    results = list(col.find({"title": {"$in":all_titles}}, {"_id": 0 , "id":1, "title": 1, "price": 1, "baseColour": 1, "articleType": 1, "gender": 1, "link" : 1, "mfg_brand_name": 1}))
    sorted_results = []
    for title in all_titles:
        for result in results:
            if result['title'] == title:
                sorted_results.append(result)
                break
    return sorted_results

@app.route("/check", methods=["GET", "POST"])
def check_response():
    req = request.get_json()
    ip = req["ip"]
    check_query_prompt = f"This Is AI Bot that categories if the given response from previous stage of LLM as valid response to be transformed through vector embedding and perform a vector search on ecommerce fashion database \n Answer with Yes or No only \n ##Response from LLM: {ip}"
    check = get_conversation_chain_conv().predict(input=check_query_prompt)
    return jsonify({"status": check})

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

    # prepare user profile filter query in new vector search format
    filter_query_new = []
    if len(user_profile.keys())>0 and "sex" in user_profile:
        filter_query_new += [{"gender": user_profile["sex"]}]
    if len(user_profile.keys())>0 and "age_group" in user_profile:
        filter_query_new += [{"ageGroup": user_profile["age_group"]}]

    # initiate conversaiton
    if len(session[mem_key+"_chat_history"]) < 1:
        question = f"User profile Context to use to response : Greetings!!! I am {name} and I am {gender} and live in {address} \n ##User: {question} \n ## AI Assistant:"
        # resp = get_conversation_chain_rag(filter_query).run({"question":f"Greetings!!! I am {name} and I am {gender} and live in {address}", "chat_history":session[mem_key+"_chat_history"]})
        # session[mem_key+"_chat_history"] += [(f"Greetings!!! I am {name} and I am looking for {gender}s products to purchase and live in {address}", resp)]
    

    # conversation handler 
    llm = get_conversation_chain_conv()

    history = ChatMessageHistory()
    for message in session[mem_key+"_chat_history"]:
        history.add_user_message(message[0])
        history.add_ai_message(message[1])
    memory = ConversationSummaryMemory.from_messages(
        llm=llm,
        chat_memory=history,
        return_messages=True
    )

    status = get_product_reco_status(question, memory.buffer)
    op = {}
    if status['relevancy_status']:
        product_recommendations = get_product_recommendations(question, memory.buffer, filter_query_new,\
                                     reco_queries=status["recommendations"])

        
        session[mem_key+"_chat_history"] += [(question, product_recommendations['message'])]
        op["response"] = product_recommendations['message']
        sorted_results = get_sorted_results(product_recommendations)
        products = sorted_results
        op["recommendations"] = products
        op["product_query"] = status["recommendations"]
    else:
        response = llm.invoke(question)
        op["response"] = response.content
        
    # op["history"] = session[mem_key+"_chat_history"]
    return jsonify(op)

@app.route('/clear/session', methods=['DELETE'])
def clean_session():
    session.clear()
    return "Session Cleared"

if __name__ == "__main__":
    dotenv_path = Path('.env')
    load_dotenv()
    app.run(debug=True, host='0.0.0.0',port = 5001)
