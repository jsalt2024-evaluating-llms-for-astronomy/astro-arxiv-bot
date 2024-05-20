import argparse
import chromadb
import json
import logging
import os
import re
import torch

from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, conlist
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from transformers import AutoTokenizer
from typing import List

from llama_index.core import (
    Document,
    PromptTemplate,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core import Settings
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

ROOT = Path(__file__).parents[1].resolve()
hf_token = os.environ["HF_TOKEN"]
app = App(token=os.environ["SLACK_BOT_TOKEN"])
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Running the LLM RAG-powered Slack bot, AstroArXivBot."
    )

    parser.add_argument(
        "--local-llm",
        action=argparse.BooleanOptionalAction,
        help="Use LLaMA-3-8B-Instruct (local LLM) instead of OpenAI",
    )
    parser.add_argument(
        "-k",
        "--top-k-papers",
        type=int,
        help="Number of top papers to retrieve",
        default=3,
    )

    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging verbosity level (default: INFO)",
    )

    args = parser.parse_args()
    return args


def initialize_models(local_llm=True):

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model

    if local_llm:
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        logging.info(f"Using LOCAL MODEL: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

        stopping_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        llm = HuggingFaceLLM(
            model_name=model_name,
            model_kwargs={
                "token": hf_token,
                "torch_dtype": torch.bfloat16,
            },
            generate_kwargs={
                "do_sample": False,
                "pad_token_id": tokenizer.eos_token_id,
            },
            tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
            tokenizer_kwargs={"token": hf_token},
            stopping_ids=stopping_ids,
        )
        Settings.llm = llm
        Settings.tokenizer = tokenizer
    else:
        gpt_model_name = "gpt-4o"
        llm = OpenAI(model=gpt_model_name, temperature=0.0)
        logging.info(f"Using GPT model: {gpt_model_name}")
        Settings.llm = llm


def setup_logging(verbosity, log_filename):

    log_level = getattr(logging, verbosity.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )


qa_prompt = PromptTemplate(
    "You are an astronomy research assistant that can access arXiv paper chunks as context "
    "to answer user queries."
    "Some of the context may contain LaTeX code, but please answer "
    "the query in natural language like a professional research astronomer. "
    "Match the level of specificity or generality as the query. "
    "ALWAYS cite ALL relevant papers using EXACTLY the citation style in the context, "
    "in parentheses: `(<https://arxiv.org/abs/1406.2364|1406.2364>)`."
    "The current year is 2024. Unless directed others, prioritize MORE RECENT results based on the paper YEAR. "
    "Answer in 100 words or fewer. "  # maybe more like 50 for Llama-3...
    "If the query is not related to astronomy in any way, or if none of the papers can help "
    "you answer this, then say 'I cannot answer'.\n\n"
    "The arXiv astro-ph papers context string are below:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Query: {query_str}\n"
    "Answer: "
)


@app.event("app_mention")
@app.event("message")
def handle_query_event(event, say, client):
    try:
        logging.debug(f"Received event: {event}")

        event_type = event["type"]
        channel_id = event["channel"]
        ts = event["ts"]
        thread_ts = (
            event.get("thread_ts") or ts
        )  # Use thread_ts if present, else fallback to ts
        user = event["user"]
        full_user_query = event["text"]

        # strip out the bot name mention from the query
        user_query = re.sub(r"<@[^>]+>", "", full_user_query).strip()

        # check if this message is in a thread where the bot has already replied
        if event.get("thread_ts") and event["thread_ts"] != event["ts"]:
            # log the message without replying
            logging.info(
                f"Message in thread from user: {user}, text: {full_user_query}, thread timestamp: {thread_ts}"
            )
        else:
            # handle direct messages
            if event.get("channel_type") == "im":
                if user_query:
                    # log the direct message
                    logging.info(
                        f"Direct message received. User: {user}, Channel: {channel_id}, Event type: {event_type}, Thread timestamp {thread_ts}\n{user_query}"
                    )

                    response = query_engine.custom_query(user_query)

                    # send the response to Slack in the same thread
                    reply = say(
                        response,
                        channel=channel_id,
                        thread_ts=thread_ts,
                        unfurl_links=False,
                    )

                    # pre-populate reply with two emoji reactions
                    client.reactions_add(
                        channel=channel_id,
                        timestamp=reply["ts"],
                        name="thumbsup",
                    )
                    client.reactions_add(
                        channel=channel_id,
                        timestamp=reply["ts"],
                        name="thumbsdown",
                    )
                    logging.info(f"Response ({reply['ts']})\n{response}\n")

            # handle mentions in channels -- same as above
            elif event_type == "app_mention":
                if user_query:
                    logging.info(
                        f"Mention received. User: {user}, Channel: {channel_id}, Event type: {event_type}, Thread timestamp {thread_ts}\n{user_query}"
                    )

                    response = query_engine.custom_query(user_query)

                    reply = say(
                        response,
                        channel=channel_id,
                        thread_ts=thread_ts,
                        unfurl_links=False,
                    )

                    client.reactions_add(
                        channel=channel_id,
                        timestamp=reply["ts"],
                        name="thumbsup",
                    )
                    client.reactions_add(
                        channel=channel_id,
                        timestamp=reply["ts"],
                        name="thumbsdown",
                    )
                    logging.info(f"Response ({reply['ts']})\n{response}\n")

    except Exception as e:
        logging.error(f"Error: {e}")


@app.event("reaction_added")
@app.event("reaction_removed")
def handle_reaction_added(event, say):
    try:
        logging.debug(f"Received event: {event['type']}")

        event_type = event["type"]
        message_ts = event["item"]["ts"]
        reaction = event["reaction"]
        user = event["user"]

        logging.info(
            f"{event_type}: {reaction}, Message timestamp: {message_ts}, User: {user}"
        )
    except Exception as e:
        logging.error(f"Error: {e}")


def process_json_data(directory_path: str | os.PathLike) -> List[Document]:
    """Helper function to convert arXiv data into a list of LlamaIndex Documents"""
    processed_docs = []
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r") as file:
                    doc = json.load(file)
                    for document_key in doc:
                        if "abstract" in doc[document_key] and (
                            doc[document_key]["abstract"] != ""
                        ):  # ensure all nodes have some valid string..
                            document = Document(
                                doc_id=document_key,
                                text=doc[document_key]["abstract"],
                                metadata={
                                    "doc_id": document_key,
                                    "filepath": file_path,
                                    "yymm": filename,
                                },
                            )
                            processed_docs.append(document)
    return processed_docs


class ArXivPaper(BaseModel):
    """Data model for arXiv astro-ph paper."""

    year: int
    arxiv_id: str = Field(..., description="arXiv identifier")
    summary: str = Field(..., description="one sentence summary of the paper")


def model_validate_arxiv_paper(formatted_context: str, max_retry=5) -> ArXivPaper:
    """A model validation program that prompts (and forces) some formatted_context
    (i.e. output of `format_node_data()`) to adhere to the ArXivPaper model.

    Honestly the JSON completion for LLaMA-3 is somewhat unreliable so it may be better to
    either patch in openai here, or simply skip this until we have a better option.
    """
    validated_output = LLMTextCompletionProgram.from_defaults(
        output_cls=ArXivPaper,
        llm=Settings.llm,
        prompt_template_str=(
            "Please extract the following text into a structured ArXivPaper object: {formatted_context}. "
            "Make sure it is valid python dict that does not include extra characters like newlines. "
            "Provide a concise summary (one sentence). "
            "Extract the arXiv id like '1501.00001' or 'astro-ph9905116'; "
            "it should NOT contain the word 'arXiv' in it. "
        ),
    )

    retries = 0
    while retries < max_retry:
        try:
            return validated_output(formatted_context=formatted_context)
        except Exception as e:
            retries += 1
            logging.warning(f"Retry #{retries}", e)
    return None


def make_arxiv_link(arxiv_id):
    return f"https://arxiv.org/abs/{arxiv_id if '.' in arxiv_id else arxiv_id.replace('astro-ph', 'astro-ph/')}"


def format_node_data(node: TextNode, improved_validation=True) -> str:
    """Convert LlamaIndex Document (TextNode) objects into RAG context string.
    Uses the model_validate_arxiv_paper() structured extraction if `improved_validation`
    is set to True.
    """

    doc_id = node.metadata["doc_id"]
    filepath = node.metadata["filepath"]
    yymm = node.metadata["yymm"]

    abstract = node.text

    with open(filepath, "r") as file:
        doc = json.load(file)
        conclusions = doc[doc_id]["conclusions"]

    arxiv_id = doc_id.replace("_arXiv", "")

    formatted_context = (
        f"CITATION: '<{make_arxiv_link(arxiv_id)}|{arxiv_id}>'\n\n"
        f"YEAR: {'19' + yymm[:2] if int(yymm[:4]) > 9000 else '20' + yymm[:2]}\n\n"
        f"ABSTRACT: {abstract}\n\n"
        f"CONCLUSIONS: {conclusions}\n"
    )

    # CURRENTLY NOT RELIABLE ENOUGH TO USE
    if improved_validation:
        output = model_validate_arxiv_paper(formatted_context)

        if output is None:
            return formatted_context

        improved_formatted_context = (
            f"CITATION: '<{make_arxiv_link(output.arxiv_id)}|{output.arxiv_id}>'\n\n"
            f"YEAR: {output.year}\n\n"
            f"SUMMARY: {output.summary}\n\n"
            f"ABSTRACT: {abstract}\n\n"
            f"CONCLUSIONS: {conclusions}\n"
        )
        return improved_formatted_context
    else:
        return formatted_context


class ArxivRetrievalQueryEngine(CustomQueryEngine):
    """arXiv/astroph RAG query engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: OpenAI
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)

        # converts documents -> formatted strings
        formatted_contexts = [
            format_node_data(n, improved_validation=False) for n in nodes
        ]

        context_str = "\n\n".join(formatted_contexts)
        response = Settings.llm.complete(
            qa_prompt.format(context_str=context_str, query_str=query_str)
        )

        return str(response)


def build_query_engine(
    index, top_k_papers=5, response_mode="refine"
) -> ArxivRetrievalQueryEngine:
    """convenience function that returns the query engine defined in the
    above class.

    Params
    top_k_papers: int
        number of astro-ph documents to retrieve
    response_mode: str -- must be "refine" or "compact"
        how should the LLM integrate the context?

    `qa_prompt` is a global vars
    """
    retriever = index.as_retriever(similarity_top_k=top_k_papers)
    synthesizer = get_response_synthesizer(
        llm=Settings.llm, text_qa_template=qa_prompt, response_mode=response_mode
    )

    query_engine = ArxivRetrievalQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
        llm=Settings.llm,
        qa_prompt=qa_prompt,
    )

    return query_engine


if __name__ == "__main__":

    args = parse_arguments()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{ROOT}/logs/log_{timestamp}.log"
    setup_logging(args.verbosity, log_filename)

    initialize_models(local_llm=args.local_llm)

    logging.info(f"Working in {ROOT} directory")
    db = chromadb.PersistentClient(path=f"{ROOT}/chroma_db")
    chroma_collection = db.get_or_create_collection("astroph-abs_bge-small-en")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    n_docs = chroma_collection.count()

    # if empty, create collection and index
    if n_docs == 0:
        docs_path = f"{ROOT}/arxiv-astro-ph"
        arxiv_docs = process_json_data(docs_path)

        logging.info(f"Building vector store for {len(arxiv_docs)} documents...")

        # for building vector store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            arxiv_docs,
            storage_context=storage_context,
            embed_model=Settings.embed_model,
        )
    # otherwise, build index
    else:
        logging.info(f"Loading {n_docs} documents...")
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=Settings.embed_model
        )

    # build query engine (using global var for qa_prompt... see top!)
    query_engine = build_query_engine(
        index, top_k_papers=args.top_k_papers, response_mode="compact"
    )

    # start Slack app
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
