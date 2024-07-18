import os
import logging
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

load_dotenv()
embed_fn = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
llm_chat = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv('OPENAI_API_KEY'))

# chat_history = []
num_chunks = 10
def load_local_vectordb_using_qdrant():
    try:
        cl_name = os.getenv('collection_name')
        qdrant_client = QdrantClient(
            url=os.getenv('qdrant_url'),
            api_key=os.getenv('qdrant_api_key'),
        )
        qdrant_store = Qdrant(qdrant_client, cl_name, embed_fn)
        return qdrant_store
    except Exception as e:
        log.error(f"Error while loading vectordb: {str(e)}")
        raise

def retri_answer(query, vectordb, chat_history):
    try:
        rephrase_question_template = """from given history rephrase the question:
        Question: {query}
        chat_history: {chat_history}
        """
        
        output = StrOutputParser()
        question_prompt = PromptTemplate(template=rephrase_question_template, input_variables=["query", "chat_history"])
        rephrase_chain = question_prompt | llm_chat | output
        question = rephrase_chain.invoke({"query": query, "chat_history": chat_history})
        # retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})
        retriever = vectordb.similarity_search(query)
        context =[]
        for i in range (len(retriever)):
            context.append(retriever[i].page_content)

        _template = """make appropriate answer of this context. manswer must be a string:
        context:{context}
        """

        answer_prompt = PromptTemplate(template=_template, input_variables=["context"])
        retrival_chain = answer_prompt | llm_chat | output
        response = retrival_chain.invoke({"context": context})        
        
        # chat_history.append({"query": question, "response": response})
        
        return response , retriever[0].metadata['source_url']
    except Exception as e:
        log.error(f"Error during retrieval: {str(e)}")
        raise
