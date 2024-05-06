from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

from pymongo import MongoClient
from typing import List
import certifi
import os
from dotenv import load_dotenv

load_dotenv()

from functools import lru_cache

client = MongoClient(os.environ["MDB_CONNECTION_STR"], tlsCAFile=certifi.where())
db = client["search"]
# llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.7, max_tokens=1024)

@lru_cache
def get_openai_emb_transformers():
    embeddings = OpenAIEmbeddings()
    return embeddings

@lru_cache
def get_vector_store():
    col = db["catalog_final_myn"]
    vs = MongoDBAtlasVectorSearch(collection=col,embedding=get_openai_emb_transformers(),index_name="vector_index",embedding_key="openAIVec", text_key="title")
    return vs

@lru_cache(10)
def get_conversation_chain_conv():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, max_tokens=2048)
    # chain = ConversationChain(llm=llm, memory=ConversationBufferWindowMemory(k=5))
    return llm


# Define your desired data structure.
class ProductRecoStatus(BaseModel):
    relevancy_status: bool = Field(description="Product recommendation status is conditioned on the fact if the context of input query is to purchase a fashion clothing and or fashion accessories.")
    recommendations: List[str] = Field(description="list of recommended product titles based on the input query context and if recommendation_status is true.")


class Product(BaseModel):
    # id: str = Field(description="Unique identifier for the product.")
    title: str = Field(description="Title of the product.")
    # price: float = Field(description="Price of the product.")
    # atp: int = Field(description="Availability to purchase the product.")
    baseColour: List[str] = Field(description="List of base colours of the product.")
    gender: List[str] = Field(description="List of genders the product is targeted for.")
    # link: str = Field(description="Link to the product image.")
    articleType: str = Field(description="Type of the article.")
    # score: float = Field(description="Score assigned to the product by the recommendation model.")
    mfg_brand_name: str = Field(description="Manufacturer or brand name of the product.")


class Recommendations(BaseModel):
    products: List[Product] = Field(description="List of recommended products.")
    message: str = Field(description="Message to the user and context of the chat history summary.")


reco_status_parser = JsonOutputParser(pydantic_object=ProductRecoStatus)

reco_status_prompt = PromptTemplate(
    template="You are AI assistant tasked at identifying if there is a product purchase intent in the query and providing suitable fashion recommendations.\n{format_instructions}\n{query}\n\
        #Chat History Summary: {chat_history}\n\nBased on the context of the query, please provide the relevancy status and list of recommended products.",
    input_variables=["query", "chat_history"],
    partial_variables={"format_instructions": reco_status_parser.get_format_instructions()},
)

reco_parser = JsonOutputParser(pydantic_object=Recommendations)
reco_prompt = PromptTemplate(
    input_variables=["question", "recommendations", "chat_history"],
    partial_variables={"format_instructions": reco_parser.get_format_instructions()},
    template="\n User query:{question} \n Chat Summary: {chat_history} \n Rank and suggest me suitable products for creating grouped product recommendations given all product recommendations below feature atleast one product for each articleType \n {recommendations} \n show output in {format_instructions} for top 10 products"
)

def get_product_reco_status(query: str, chat_history: List[str]):
    llm = get_conversation_chain_conv()
    chain = reco_status_prompt | llm | reco_status_parser
    resp = chain.invoke({"query": query, "chat_history": chat_history})
    return resp

def get_product_recommendations(query: str, chat_history: List[str], filter_query: dict, reco_queries: List[str]):
    vectorstore = get_vector_store()
    retr = vectorstore.as_retriever(search_kwargs={"k": 10, "pre_filter":filter_query})
    all_recommendations = []
    for reco_query in reco_queries:
        all_recommendations += retr.get_relevant_documents(reco_query)
    llm = get_conversation_chain_conv()
    chain = reco_prompt | llm | reco_parser
    resp = chain.invoke({"question": query, "chat_history": chat_history, "recommendations": [v.page_content for v in all_recommendations]})
    return resp


def similarity_search(question,filter=[{"gender":"Men"}],k=200):
    collection = client['search']['catalog_final_myn']
    query_vector = get_openai_emb_transformers().embed_query(question)
    vectorSearchQuery = {
            "index": "vector_index",
            "queryVector": query_vector,
            "path": "openAIVec",
            "limit": k,
            "numCandidates": 200,
        }
    if len(filter)>0:
        vectorSearchQuery["filter"] = {"$and":filter}
    pipeline = [
    {
        "$vectorSearch": vectorSearchQuery
    },
    {
        "$addFields":{
        "score":{
            '$meta': 'vectorSearchScore'
            }
        }
    },
    {
        "$match":{
        "score":{
            "$gt":0.8
        }
        }
    },
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
    op = []
    for k in buckets.keys():
        op += buckets[k][:5]
    op = sorted(op,key=lambda ele: -1*ele["score"])
    return op