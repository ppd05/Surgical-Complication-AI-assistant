import streamlit as st
import os
import re
from dotenv import load_dotenv

#Importing Core Components from Source Files 
from src.data_loader import load_json_knowledgebase, prepare_documents
from src.vectorstore_manager import build_faiss_retriever
from src.ai_engine import initialize_llm, generate_response_chain, generate_response 
# -----------------------------------------------

# Loading environment variables
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY") 
KB_PATH = "data/surgical_knowledge.json"

st.set_page_config(
    page_title="Surgical Complication AI Assistant", 
    page_icon="⚕️", 
    layout="wide"
)
st.title("⚕️ Surgical Complication AI Assistant")

# PLACEHOLDER CONSTANTS
PH_SURGERY = "--- Select a Surgery ---"
PH_COMPLICATION = "--- Select a Complication ---"


# CACHING AND INITIALIZATION

@st.cache_resource
def initialize_resources(api_key, kb_path):
    #Initializes the data, LLM, Retriever, and RAG chain.
    #Uses st.cache_resource to run only once.

    if not api_key:
        st.error("🚨 GOOGLE_API_KEY not found. Please set it in your .env file.")
        st.stop()
    
    # 1. Load data
    try:
        kb_data = load_json_knowledgebase(kb_path)
        docs = prepare_documents(kb_data)

        # 2. Initialize LLM and Retriever
        # Using a flash model for speed and efficiency in RAG
        llm = initialize_llm(api_key) 
        retriever = build_faiss_retriever(docs)

        # 3. Build the LCEL RAG Chain
        rag_chain = generate_response_chain(llm, retriever)
        
        return kb_data, rag_chain
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        st.stop()


# --- HELPER FUNCTION FOR SINGLE SEARCH RETRIEVAL ---

def search_knowledge_base(query, kb_data):
    #Searches the entire knowledge base for a query match in surgery or complication names."""
    if not query:
        return None, None, None

    search_term = query.lower()
    
    # Iterate through all surgeries
    for surgery in kb_data["surgeries"]:
        surgery_name = surgery["surgery_name"]
        
        # Check if surgery name matches the query
        if search_term in surgery_name.lower():
            # If the user searches "colectomy", just return the first complication for quick view
            first_comp = surgery["complications"][0]
            return surgery_name, first_comp["name"], surgery

        # Iterate through complications within that surgery
        for complication in surgery["complications"]:
            comp_name = complication["name"]
            
            # Check if complication name matches the query
            if search_term in comp_name.lower():
                return surgery_name, comp_name, surgery
                
    return None, None, None

# HELPER FUNCTION FOR SUMMARY DISPLAY

def get_protocol_summary(kb_data):
    #Generates a markdown list of all available protocols (Surgeries and their Complications)."""
    summary = ""
    for surgery in kb_data["surgeries"]:
        comp_names = [c["name"] for c in surgery["complications"]]
        # Use a bulleted list for surgeries
        summary += f"- **{surgery['surgery_name']}** (`{surgery.get('category', 'General')}`)\n"
        # Nested list for complications
        summary += f"  - Complications: {', '.join(comp_names)}\n"
    return summary


# APP

try:
    knowledge_base_data, rag_chain_instance = initialize_resources(API_KEY, KB_PATH)
    
    # Store essential data for display/filtering
    if 'kb_data' not in st.session_state:
        st.session_state.kb_data = knowledge_base_data
        st.session_state.surgery_names = [s["surgery_name"] for s in st.session_state.kb_data["surgeries"]]
    
except Exception:
    # If initialization failed, the error message is already displayed in the cached function
    st.stop()


#SIDEBAR:Addtional info

st.sidebar.header("Application Status")
st.sidebar.markdown(
    """
    This assistant provides clinical information
    based on a pre-loaded knowledge base.

    ### **Disclaimer:**  
    ***This Application is created for educational and research purposes only.  
    For any medical-related help, please consult a medical professional.***
    """
)
st.sidebar.divider()
st.sidebar.markdown(
    """
    **RAG Model:** `gemini-2.0-flash-lite`
    
    **Scope:** Etiology, Risk Factors, Diagnosis, and Management Protocols.
    """
)


#MAIN CONTENT: Tabs for Organization

tab1, tab2 = st.tabs(["📚 Knowledge Base Explorer", "💬 AI Assistant Chat"])

# Initialize variables for the display block
selected_surgery_name = None
selected_comp_name = None
selected_surgery = None
is_valid_selection = False

with tab1:
    st.header("Browse Comprehensive Clinical Protocols")
    
    # Show the list of available protocols as requested by the user
    with st.expander("Click here to view all available Surgeries and Complications"):
        st.markdown(get_protocol_summary(st.session_state.kb_data))
    
    st.markdown("---")
    
    #MODE SELECTION: Allows user to choose between search and dropdowns
    protocol_mode = st.radio(
        "Select Protocol Retrieval Mode:",
        ["💡 Instant Search (Fast)", "📚 Protocol Explorer (Dropdowns)"],
        horizontal=True
    )
    
    st.divider()

    # --- MODE LOGIC ---
    if protocol_mode == "💡 Instant Search (Fast)":
        st.info("💡 **Instant Search:** Type any keyword (e.g., 'leak', 'cholecystectomy', 'bleeding') to immediately find the corresponding protocol.")
        
        # SINGLE SEARCH INPUT
        search_query = st.text_input(
            "Search Protocol by Surgery or Complication Name",
            placeholder="e.g., Anastomotic Leak, Bile Duct Injury, Colectomy",
            key="quick_search"
        )
        
        # EXECUTION
        if search_query:
            selected_surgery_name, selected_comp_name, selected_surgery = search_knowledge_base(
                search_query, st.session_state.kb_data
            )
            is_valid_selection = (selected_surgery_name is not None and selected_comp_name is not None)
            
        elif not search_query:
            st.info("Start typing into the search bar above to instantly find a protocol.")
            
    
    elif protocol_mode == "📚 Protocol Explorer (Dropdowns)":
        st.info("Use the dropdown menus to explore the entire structured knowledge base by selecting the Surgery, then the Complication.")
        
        # DROP DOWN EXPLORER (Restored from previous versions)
        col_sel_1, col_sel_2 = st.columns(2)
        
        surgery_options = [PH_SURGERY] + st.session_state.surgery_names
        
        with col_sel_1:
            selected_surgery_name = st.selectbox(
                "1. Select Surgery Protocol", 
                surgery_options,
                index=0,
                key="dd_surgery"
            )
        
        if selected_surgery_name != PH_SURGERY:
            selected_surgery = next(
                (s for s in st.session_state.kb_data["surgeries"] if s["surgery_name"] == selected_surgery_name), 
                None
            )

            if selected_surgery:
                comp_names_list = [c["name"] for c in selected_surgery["complications"]]
                comp_options = [PH_COMPLICATION] + comp_names_list
                
                with col_sel_2:
                    selected_comp_name = st.selectbox(
                        "2. Select Specific Complication", 
                        comp_options,
                        index=0,
                        key="dd_complication"
                    )
                    
                if selected_comp_name != PH_COMPLICATION:
                    is_valid_selection = True
            
    #  DATA RETRIEVAL AND DISPLAY LOGIC (Common to both modes) 
    
    # Default placeholder data structure
    default_data = {
        'etiology': 'Select a mode and find a protocol above.', 
        'risk_factors': 'Select a mode and find a protocol above.', 
        'diagnostic_criteria': 'Select a mode and find a protocol above.', 
        'protocol': 'Select a mode and find a protocol above to view the full management protocol.', 
        'references': []
    }
    
    selected_comp_data = default_data
    
    if is_valid_selection:
        # If a match is found (either via search or dropdown), retrieve the specific complication data
        selected_comp_data = next(
            (c for c in selected_surgery["complications"] if c["name"] == selected_comp_name),
            default_data
        )
        
        st.subheader(f"Detailed Protocol: {selected_comp_name}")
        st.caption(f"Applies to: **{selected_surgery_name}**")
        
    elif (protocol_mode == "💡 Instant Search (Fast)") and ('search_query' in locals() and search_query and not is_valid_selection):
        st.warning(f"No protocol found matching '{search_query}'. Please try different keywords or switch to the Explorer mode.")
    # Note: If Explorer mode is active but not selected, the default_data placeholders will show.


    #  Display Clinical Fields (Updated to logical flow: Risk -> Problem -> Confirmation) --
    col_r, col_e, col_d = st.columns(3) 
    
    with col_r:
        st.metric(label="💡 WHY IS THIS PATIENT AT RISK?(Pre-existing conditions,procedural factors)", value=" ")
        st.markdown(selected_comp_data.get('risk_factors', 'N/A'))

    with col_e:
        st.metric(label="💥 WHAT IS THE PROBLEM?(Mechanism of injury,pathophysiology)", value=" ")
        st.markdown(selected_comp_data.get('etiology', 'N/A'))

    with col_d:
        st.metric(label="✅ HOW DO WE CONFIRM IT?(Signs,symptoms,diagnostic tests)", value=" ")
        st.markdown(selected_comp_data.get('diagnostic_criteria', 'N/A'))
    
    st.markdown("---")
    
    # Management Protocol (Large Section)
    col_protocol, col_references = st.columns([3, 1])

    with col_protocol:
        st.subheader("📋 Management Protocol Steps")
        protocol_text = selected_comp_data.get('protocol', 'N/A')
        
        # Only format protocol if a valid selection has been made
        if is_valid_selection:
            formatted_protocol = ""
            
            # --- FIXED LOGIC START: Use regex to reliably split on "Step X: " ---
            # Regex to find the content of each step. The pattern looks for 'Step X: ' followed by 
            # content, stopping just before the next 'Step Y: ' or the end of the string.
            step_content_matches = re.findall(r'Step \d+: (.*?)(?=Step \d+: |$)', protocol_text, re.DOTALL)
            
            if step_content_matches:
                # We now have a list where each element is the content of one step.
                for i, step in enumerate(step_content_matches):
                    clean_step = step.strip()
                    # Use standard Markdown numbered list (1. 2. 3. etc.)
                    formatted_protocol += f"{i + 1}. **{clean_step}**\n"
            else:
                # Fallback if the regex doesn't match (e.g., if data format changes)
                formatted_protocol = protocol_text 
            # --- FIXED LOGIC END ---

            st.markdown(formatted_protocol)
        else:
            st.markdown(protocol_text) # This will show the placeholder message
            

    with col_references:
        st.subheader("🔗 References")
        if is_valid_selection and selected_comp_data.get("references"):
            for ref in selected_comp_data["references"]:
                # Corrected to use the reference text for link display
                st.markdown(f"- [Standard Treatment Guidelines (MoHFW)]({ref})")
        else:
            st.markdown("*No references provided.*")

with tab2:
    st.header("Ask the RAG Assistant")
    st.info("The assistant answers questions based *only* on the indexed surgical protocols, providing structured information (Diagnosis, Etiology, Risk Factors, and Management).")
    
    query = st.text_area(
        "Type your clinical question here:", 
        placeholder="e.g., What are the signs of anastomotic leak after colectomy? Or, what causes Bile Duct Injury?",
        height=100
    )
    
    if st.button("Generate Structured Clinical Response", type="primary"):
        if query:
            with st.spinner("Analyzing knowledge base and generating structured response..."):
                # Use the cached RAG chain instance
                response = generate_response(rag_chain_instance, query) 
            
            st.subheader("💡 AI Generated Response")
            # Using st.markdown now, as the AI response contains Markdown headers from the structured prompt
            st.markdown(response)
        else:
            st.warning("Please enter a question to get a response.")