import streamlit as st
import pandas as pd
import google.generativeai as genai
import re
from PIL import Image
import requests
import markdown
import time

# PDF PROCESSING
import PyPDF2

# M. MEETARBHAN
# 7/23/2024


# === TITLE ====
st.set_page_config(
    page_title="BDOGPT",
    page_icon="🐦",
    layout="wide",
)

#------------------------------------------------------------
# HEADER
st.title('🐦BDO-GPT')
st.caption("With ❤️ by your Intern")

#------------------------------------------------------------
# FUNCTIONS
def extract_graphviz_info(text: str) -> list[str]:
    graphviz_info = text.split('```')
    return [graph for graph in graphviz_info if ('graph' in graph or 'digraph') and ('{' in graph and '}' in graph)]

def append_message(message: dict) -> None:
    st.session_state.chat_session.append({'user': message})
    return

# add_message(role, content) adds messages to the session state
# Messages include both model responses and user prompts
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    with st.chat_message(role):
        st.markdown(content)

# load_model() loads the default gemini-pro model for general tasks
@st.cache_resource
def load_model() -> genai.GenerativeModel:
    model = genai.GenerativeModel('gemini-pro')
    return model

# load_modelvision() is a simple function that returns the vision model used to 
# intepret images
@st.cache_resource
def load_modelvision() -> genai.GenerativeModel:
    model = genai.GenerativeModel('gemini-pro-vision')
    return model


# download_html(response) converts the response to Markdown to be saved
def download_html(response):
    markdown_content = f"# Produced by BDO GPT\n\n{response}"
    html_content = markdown.markdown(markdown_content)
    return html_content

# extract_from_pdf(pdf_file) consumes a PdfFile and uses the PyPDF2 library to
# extract all text from the pdf 
def extract_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    extracted_text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        extracted_text +=   page.extract_text() + "\n"
    return extracted_text
    

#=================================================================
# CONFIGURATION
# TODO : SET API KEY AS AN ENVIRONMENT VARIABLE
genai.configure(api_key='AIzaSyDkmaWadxYJGAgWdMVpB-qfPyhrjctrZcI')

model = load_model()
vision = load_modelvision()

if 'chat' not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = []

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'welcome' not in st.session_state:
    welcome = model.generate_content(f'''
    You are BDOGPT! Say hello to the user. Provide a random fact of the day but specify it.
    Random Fact : 
    ''')
    welcome.resolve()
    st.session_state.welcome = welcome

    with st.chat_message('ai'):
        st.write(st.session_state.welcome.text)
else:
    with st.chat_message('ai'):
        st.write(st.session_state.welcome.text)

if len(st.session_state.chat_session) > 0:
    count = 0
    for message in st.session_state.chat_session:
        if message['user']['role'] == 'model':
            with st.chat_message('ai'):
                st.write(message['user']['parts'])
                graphs = extract_graphviz_info(message['user']['parts'])
                if len(graphs) > 0:
                    for graph in graphs:
                        st.graphviz_chart(graph, use_container_width=False)
                        with st.expander("View text"):
                            st.code(graph, language='dot')

                html_content = download_html(message['user']['parts'])
                st.download_button(
                    label="Save Response",
                    data=html_content,
                    file_name=f"response_{count}.html",
                    mime="text/html"
                )
        else:
            with st.chat_message('user'):
                st.write(message['user']['parts'][0])
                if len(message['user']['parts']) > 1:
                    st.image(message['user']['parts'][1], width=200)
        count += 1

# =================== TODO : REMOVE DOMAIN SPECIFIC TOGGLE IN DEPLOYMENT ========================
# Domain Specific Button Toggle
if 'domainType' not in st.session_state:
    st.session_state.domainType = False

def toggle_domain_type():
    st.session_state.domainType = not st.session_state.domainType

#CdomainButton = st.button("AML-CFT Mode", key=None, help='Chat w/ AML PDFs', type="primary", on_click=toggle_domain_type)

#st.caption(f"AML-CFT Mode: {st.session_state.domainType}")
#==================================================================================================

# =================================== SIDEBAR ================================================
with st.sidebar:
    image_attachment = st.checkbox("Attach image", value=False, help="Attach an image and let BDOGPT help")
    txt_attachment = st.checkbox("Attach text file", value=False, help="Attach a Text file and let BDOGPT help")
    csv_excel_attachment = st.checkbox("Attach CSV or Excel", value=False, help="Attach a CSV or Excel file and let BDOGPT help")
    graphviz_mode = st.checkbox("Graphviz mode", value=False, help="Generates graphs from your prompts or Data")
    pdf_attachment = st.checkbox("Upload PDF", value=False, help="Upload a PDF to query")

    if image_attachment:
        image = st.file_uploader("Upload your image", type=['png', 'jpg', 'jpeg'])
        url = st.text_input("Or paste your image url")
    else:
        image = None
        url = ''

    if txt_attachment:
        txtattachment = st.file_uploader("Upload your text file", type=['txt'])
    else:
        txtattachment = None

    if csv_excel_attachment:
        csvexcelattachment = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])
    else:
        csvexcelattachment = None

    if pdf_attachment:
        pdfattachment = st.file_uploader("Upload your PDF file", type=['pdf'])
    else:
        pdfattachment = None

# =========================================================================================================

prompt = st.chat_input("How can BDO help you today?")

if prompt:
    txt = ''
    if txtattachment:
        txt = txtattachment.getvalue().decode("utf-8")
        txt = '   Text file: \n' + txt

    if csvexcelattachment:
        try:
            df = pd.read_csv(csvexcelattachment)
        except:
            df = pd.read_excel(csvexcelattachment)
        txt += '   Dataframe: \n' + str(df)
    
    if pdfattachment:
        txt = extract_from_pdf(pdfattachment)
        txt = ' PDF File : \n' + txt

    if graphviz_mode:
        txt += '   Generate a graph with graphviz in .dot \n'

# Conditional Check : If length of text exceeds 8000, truncatenate and display with ... at the end
    if len(txt) > 8000:
        txt = txt[:8000] + '...'
    if image or url != '':
        if url != '':
            img = Image.open(requests.get(url, stream=True).raw)
        else:
            img = Image.open(image)
        prmt = {'role': 'user', 'parts': [prompt + txt, img]}
    else:
        prmt = {'role': 'user', 'parts': [prompt + txt]}

    append_message(prmt)

    with st.spinner('Wait, BDO is cooking...'):
        if len(prmt['parts']) > 1:
            response = vision.generate_content(prmt['parts'], stream=True, safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_LOW_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_LOW_AND_ABOVE",
                },
            ])
            response.resolve()
        else:
            response = st.session_state.chat.send_message(prmt['parts'][0])

        try:
            append_message({'role': 'model', 'parts': response.text})
        except Exception as e:
            append_message({'role': 'model', 'parts': f'{type(e).__name__}: {e}'})

        st.rerun()

# Handle query_rag for text input
if prompt and st.session_state.domainType:
    start_time = time.time()
    add_message("user", prompt)
    response_placeholder = st.empty()

    response_text = ""
    for chunk in query_rag(prompt, st.session_state.domainType):
        response_text += chunk + " "
        response_placeholder.markdown(f"**Assistant:** {response_text.strip()}")

    add_message("assistant", response_text.strip())
    time_taken = time.time() - start_time

    st.success(f"Success :) Time Taken :{time_taken}", icon="🔥")
