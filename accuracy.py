import difflib
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    print("Setting custom prompt...")
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    print("Creating RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# Loading the model
def load_llm():
    print("Loading language model...")
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# QA Model Function
def qa_bot():
    print("Initializing QA bot...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    print("Loading FAISS database...")
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading FAISS database: {e}")
        return None
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# Enhanced accuracy evaluation function with detailed debugging
def evaluate_accuracy(qa_bot_func):
    # Initialize the QA bot
    qa = qa_bot_func()
    
    # Define your test dataset
    test_dataset = [
        {'question': 'What is diabetes?', 'answer': 'Diabetes is a chronic condition characterized by high levels of sugar (glucose) in the blood.'},
        {'question': 'What are the main types of diabetes?', 'answer': 'The main types of diabetes are Type 1, Type 2, and gestational diabetes.'},
        {'question': 'What is Type 1 diabetes?', 'answer': 'Type 1 diabetes is a condition in which the immune system attacks and destroys insulin-producing beta cells in the pancreas.'},
        {'question': 'What is Type 2 diabetes?', 'answer': 'Type 2 diabetes is a condition in which the body becomes resistant to insulin or does not produce enough insulin.'},
        {'question': 'What is gestational diabetes?', 'answer': 'Gestational diabetes is a type of diabetes that develops during pregnancy and usually goes away after the baby is born.'},
        {'question': 'What are the symptoms of diabetes?', 'answer': 'Symptoms of diabetes include increased thirst, frequent urination, extreme fatigue, blurred vision, and slow-healing sores.'},
        {'question': 'How is diabetes diagnosed?', 'answer': 'Diabetes is diagnosed through blood tests such as the fasting blood glucose test, oral glucose tolerance test, and A1C test.'},
        {'question': 'What is the A1C test?', 'answer': 'The A1C test measures the average blood glucose levels over the past 2-3 months.'},
        {'question': 'Can diabetes be cured?', 'answer': 'There is no cure for diabetes, but it can be managed through medication, lifestyle changes, and monitoring blood sugar levels.'}
    ]

    correct_answers = 0
    total_questions = len(test_dataset)

    for item in test_dataset:
        question = item['question']
        true_answer = item['answer']
        
        response = qa.invoke({'query': question})
        bot_answer = response['result']
        
        # Print the question, true answer, and bot's answer for debugging
        print(f"Question: {question}")
        print(f"True Answer: {true_answer}")
        print(f"Bot Answer: {bot_answer}")
        print("-" * 40)
        
        # Use difflib to check for similarity
        similarity = difflib.SequenceMatcher(None, bot_answer.strip().lower(), true_answer.strip().lower()).ratio()
        if similarity > 0.8:  # Consider answers similar if the similarity is above 80%
            correct_answers += 1

    accuracy = correct_answers / total_questions
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    # Evaluate accuracy
    evaluate_accuracy(qa_bot)

    # Start Chainlit bot
    cl.run()
