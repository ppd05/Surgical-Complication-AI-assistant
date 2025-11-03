from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

def initialize_llm(api_key):
    #Initialize the Gemini 2.0 Flash-Lite model.
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=api_key, temperature=0.0)

def generate_response_chain(llm, retriever):
    SYSTEM_PROMPT = (
        "You are an expert Surgical Complication AI Assistant. Your task is to provide "
        "accurate, concise, and structured clinical summaries based ONLY on the provided "
        "CONTEXT from the surgical knowledge base. Do not use external knowledge. "
        "If the context does not contain the answer, state that you cannot find the information."
        
        "Format your answer strictly with the following clinical sections (use Markdown formatting):"
        "\n\n### 1. Diagnosis & Presentation (Signs/Imaging)"
        "\n### 2. Etiology & Risk Factors"
        "\n### 3. Management Protocol"
        "\n\nAt the end, include a short summary of the source document(s) used for grounding."
    )
    # --------------------------------------------------
    
    # Using chatprompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context: {context}\n\nQuestion: {question}"),
        ]
    )

    # RAG chain structure using RunnableSequence
    rag_chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def generate_response(rag_chain, query):
    try:
        # The chain expects a dictionary with the key "question"
        response = rag_chain.invoke({"question": query})
        return response
    except Exception as e:
        return f"An error occurred during generation: {e}"