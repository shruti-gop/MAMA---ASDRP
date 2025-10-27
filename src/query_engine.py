from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import asyncio
import warnings
import logging
import time
import math
from statistics import mean

from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.chat_engine.types import ChatMode

FILE_NAME = "Minimal_Intelligence_Orchestration_Report.pdf"  # rename if needed
STORAGE_DIR = "../storage/mama_orchestration"

mama_engine = None
query_stats = []


#setup agent 1 - query engine
async def setup():
    """Initialize the query engine for the MAMA project."""
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)

    llm = Ollama(model="llama3.2:3b-instruct-q8_0", temperature=0.01)
    embedding = OllamaEmbedding(model_name="mxbai-embed-large")

    Settings.llm = llm
    Settings.embed_model = embedding

    try:
        #load existing vector index if available
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        mama_index = load_index_from_storage(storage_context=storage_context)
    except:
        #build new index if not found
        mama_report = SimpleDirectoryReader(
            input_files=[f"./data/{FILE_NAME}"]
        ).load_data()
        mama_index = VectorStoreIndex.from_documents(mama_report)
        mama_index.storage_context.persist(persist_dir=STORAGE_DIR)

    global mama_engine
    mama_engine = mama_index.as_chat_engine(chat_mode=ChatMode.BEST)


#metric tracker
def compute_entropy(source_nodes):
    # """Approximate entropy from context node distribution."""
    if not source_nodes:
        return 0.0
    n = len(source_nodes)
    p = 1 / n
    return -sum(p * math.log(p) for _ in range(n))


#query with metrics
async def query_engine(query_str):
    # """Ask the MAMA Query Engine a question and log performance metrics."""
    global mama_engine, query_stats

    start_time = time.time()
    response = await mama_engine.achat(query_str.strip())
    elapsed = time.time() - start_time

    context_count = len(response.source_nodes)
    entropy = compute_entropy(response.source_nodes)

    query_stats.append({
        "query": query_str,
        "response": response.response,
        "context_count": context_count,
        "entropy": entropy,
        "latency": elapsed
    })

    print(f"\n Agent Response:\n\t{response.response}")
    print(f"\n Metrics — Contexts: {context_count} | Entropy: {entropy:.3f} | Latency: {elapsed:.2f}s")


#main loop
async def main():
    await setup()
    query_str = input("==> What question do you have about the Minimal Intelligence Orchestration Report? ") #might not be needed

    while len(query_str.strip()) > 0:
        await query_engine(query_str)
        query_str = input("\n==> Ask another question (or press Enter to quit): ")

    if query_stats:
        print("\n Summary Metrics")
        print(f"Average Latency: {mean([q['latency'] for q in query_stats]):.2f}s")
        print(f"Average Entropy: {mean([q['entropy'] for q in query_stats]):.3f}")
        print(f"Average Context Count: {mean([q['context_count'] for q in query_stats]):.1f}")


if __name__ == "__main__":
    asyncio.run(main())
