from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import logging

logging.basicConfig(level=logging.DEBUG)

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question as thoroughly as possible.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    logging.debug("Setting custom prompt...")
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    logging.debug("Creating RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def load_llm():
    logging.debug("Loading language model...")
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=1024,  # Allow more tokens for a longer response
        temperature=0.7  # Increase temperature for more diverse responses
    )
    return llm

def qa_bot():
    logging.debug("Initializing QA bot...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    logging.debug("Loading FAISS database...")
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logging.error(f"Error loading FAISS database: {e}")
        return None
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def final_result(query):
    logging.debug(f"Handling query: {query}")
    qa_result = qa_bot()
    if qa_result is None:
        logging.error("QA bot failed to initialize.")
        return "QA bot initialization failed."
    response = qa_result({'query': query})
    return response

@cl.on_chat_start
async def start():
    logging.debug("Chat started.")
    chain = qa_bot()
    if chain is None:
        logging.error("Failed to initialize QA bot.")
        await cl.Message(content="Failed to initialize QA bot. Please try again later.").send()
        return
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to DiabetesBot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if chain is None:
        logging.error("QA bot not initialized.")
        await cl.Message(content="Failed to initialize QA bot. Please try again later.").send()
        return
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    try:
        res = await chain.ainvoke(message.content, callbacks=[cb])
        answer = res["result"]

        await cl.Message(content=answer).send()
    except Exception as e:
        logging.error(f"Error during query handling: {e}")
        await cl.Message(content="An error occurred while processing your query.").send()

if __name__ == "__main__":
    try:
        cl.run(port=8000)  # Specify your port here
    except Exception as e:
        logging.error(f"Failed to run chainlit server: {e}")
