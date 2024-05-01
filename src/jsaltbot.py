import chromadb
import json
import os
import re

from pathlib import Path
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from typing import List

from llama_index.core import Document, PromptTemplate, SimpleDirectoryReader, Settings, StorageContext, VectorStoreIndex, get_response_synthesizer
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.readers.json import JSONReader
from llama_index.vector_stores.chroma import ChromaVectorStore

ROOT = Path(__file__).parents[1].resolve()

# sentence transformers -- lightweight embedding model for now; astrollama too big
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = OpenAI(model="gpt-4-vision-preview")

Settings.chunk_size = 1024
Settings.embed_model = embed_model
Settings.llm = llm

qa_prompt = PromptTemplate(
    "Paper chunks are below:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the arXiv paper chunks, answer the query like a professional research astronomer. "
    "Only cite referenced papers if they support your answer; otherwise don't reference the "
    "arXiv paper chunks at all. Always cite using the arXiv identifier of the supporting paper " 
    "(YYMM.XXXXX) and don't use references from within the text chunks. If you find multiple "
    "papers that are relevant but have differing opinions, then use the most recent results. "
    "If you don't know the answer then say 'I cannot answer.' Answer in 100 words or less.\n"
    "Query: {query_str}\n"
    "Answer: "
)

# Slack bot stuff...
app = App(token=os.environ["SLACK_BOT_TOKEN"])

def create_query_handler(query_engine):
    """Factory method that handles the Slack bot while loading the appropriate
    query_engine instance and returns the event handler
    """
    
    def query_event_handler(event, say):
        try:
            channel_id = event['channel']
            thread_ts = event['ts']
            full_user_query = event['text']

            # strip out the bot name mention from the query
            user_query = re.sub(r"<@[^>]+>", "", full_user_query).strip()

            # user_query = body["event"]["blocks"][0]["elements"][0]["elements"][1]["text"]
            if user_query:
                print(f"Query received: {user_query}\n")
                response = query_engine.custom_query(user_query)
                print(f"Response: {response}\n")

                # this part goes to Slack
                say(response, channel=channel_id, thread_ts=thread_ts)
        except Exception as e:
            print("Error: %s"%e)

    return query_event_handler

def process_json_data(directory_path: str | os.PathLike) -> List[Document]:
    """Helper function to convert arXiv data into a list of LlamaIndex Documents
    """
    processed_docs = []
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r') as file:
                    doc = json.load(file)
                    for document_key in doc:
                        if "abstract" in doc[document_key] and (doc[document_key]["abstract"] != ""): # ensure all nodes have some valid string..
                            document = Document(
                                doc_id=document_key,
                                text=doc[document_key]["abstract"], 
                                metadata={'doc_id': document_key, 'filepath': file_path, 'yymm': filename}
                            )
                            processed_docs.append(document)
    return processed_docs

def format_node_data(node: TextNode) -> str:
    """Convert LlamaIndex Document (TextNode) objects into RAG context string.
    """
    
    doc_id = node.metadata["doc_id"]
    filepath = node.metadata["filepath"]
    yymm = node.metadata["yymm"]
    
    abs = node.text
    
    with open(filepath, 'r') as file:
        doc = json.load(file)
        conc = doc[doc_id]["conclusions"]

    formatted_context = (
        f"{doc_id}\n\n"
        f"(Year: {'19' + yymm[:2] if int(yymm[:4]) > 9000 else '20'+yymm[:2]})\n\n"
        f"ABSTRACT: {abs}\n\n"
        f"CONCLUSIONS: {conc}"
    )

    return formatted_context

class ArxivRetrievalQueryEngine(CustomQueryEngine):
    """arXiv/astroph RAG query engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: OpenAI
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)

        context_str = "\n\n".join([format_node_data(n) for n in nodes])
        response = self.llm.complete(
            qa_prompt.format(context_str=context_str, query_str=query_str)
        )

        return str(response)

def build_query_engine(index, k_retrieve=3, response_mode="refine") -> ArxivRetrievalQueryEngine:
    """convenience function that returns the query engine defined in the 
    above class. 

    Params
    k_retrieve: int
        number of astro-ph documents to retrieve
    response_mode: str -- must be "refine" or "compact"
        how should the LLM integrate the context?

    `llm` and `qa_prompt` are global vars
    """
    retriever = index.as_retriever(similarity_top_k=k_retrieve)
    synthesizer = get_response_synthesizer(llm=llm, text_qa_template=qa_prompt, response_mode=response_mode)
    
    query_engine = ArxivRetrievalQueryEngine(
        retriever=retriever, 
        response_synthesizer=synthesizer, 
        llm=llm, 
        qa_prompt=qa_prompt
    )

    return query_engine

if __name__ == "__main__":

    print(f"Working in {ROOT} directory")
    db = chromadb.PersistentClient(path=f"{ROOT}/chroma_db")
    chroma_collection = db.get_or_create_collection("astroph-abs_bge-small-en")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    n_docs = chroma_collection.count()

    # if empty, create collection and index
    if n_docs == 0:
        docs_path = f"{ROOT}/arxiv-astro-ph"
        arxiv_docs = process_json_data(docs_path)

        print(f"Building vector store for {len(arxiv_docs)} documents...")
        
        # for building vector store 
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            arxiv_docs, storage_context=storage_context, embed_model=embed_model, 
        )
    # otherwise, build index
    else:
        print(f"Loading {n_docs} documents...")
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )

    # build query engine (using global vars for llm and qa_prompt... see top!)
    query_engine = build_query_engine(index, k_retrieve=3, response_mode="refine")
    
    # initialize the Slack listener function
    handler = create_query_handler(query_engine)
    app.event("app_mention")(handler)

    # start Slack app
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()