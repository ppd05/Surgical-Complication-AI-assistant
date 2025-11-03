from langchain_core.documents import Document
import json
import os

def load_json_knowledgebase(filepath):
    #Surgical knowledge base is JSON, loads the JSON file.
    if not os.path.exists(filepath):
        print(f"Error: Knowledge base file not found at {filepath}")
        return {"surgeries": []}
    
    with open(filepath, 'r') as f:
        return json.load(f)

def prepare_documents(kb_data):
    #Transform the JSON knowledge base into a list of LangChain Document objects.
    
    #CRITICAL UPDATE: Concatenates all clinical fields (etiology, risk factors, 
    #diagnostics, and protocol) into the document's content for apt retrieval.
    
    documents = []
    
    for surgery in kb_data.get("surgeries", []):
        surgery_name = surgery.get("surgery_name")
        category = surgery.get("category")
        
        for complication in surgery.get("complications", []):
            comp_name = complication.get("name")
            etiology = complication.get("etiology", "N/A")
            risk_factors = complication.get("risk_factors", "N/A")
            diagnostic_criteria = complication.get("diagnostic_criteria", "N/A")
            protocol = complication.get("protocol", "N/A")
            
            # --- CONCATENATE ALL CLINICAL DATA FOR RAG CONTEXT
            content = (
                f"Surgical Complication: {comp_name} during/after {surgery_name}. "
                f"Clinical Context (Etiology): {etiology}. "
                f"Key Risk Factors: {risk_factors}. "
                f"Diagnostic Criteria (Signs/Labs/Imaging): {diagnostic_criteria}. "
                f"Management Protocol: {protocol}"
            )
            
            doc = Document(
                page_content=content,
                metadata={
                    "surgery": surgery_name,
                    "complication": comp_name,
                    "category": category,
                    "etiology": etiology,
                    "risk_factors": risk_factors,
                    "diagnostic_criteria": diagnostic_criteria,
                    "protocol": protocol,
                    "source": f"{surgery_name} - {comp_name}" # Used for tracing source
                }
            )
            documents.append(doc)
            
    return documents