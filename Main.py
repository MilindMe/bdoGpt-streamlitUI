import streamlit as st
import pandas as pd
import google.generativeai as genai
import re
from PIL import Image
import requests
import markdown

# PDF PROCESSING
import PyPDF2

st.set_page_config(
    page_title="BDOGPT",
    page_icon="🐦",
    layout="wide",
)

#------------------------------------------------------------
#HEADER
st.title('🐦BDO-GPT')
st.caption("With ❤️ by your Intern")

#------------------------------------------------------------
#FUNCTIONS
def extract_graphviz_info(text: str) -> list[str]:
    graphviz_info  = text.split('```')
    return [graph for graph in graphviz_info if ('graph' in graph or 'digraph') and ('{' in graph and '}' in graph)]

def append_message(message: dict) -> None:
    st.session_state.chat_session.append({'user': message})
    return

@st.cache_resource
def load_model() -> genai.GenerativeModel:
    model = genai.GenerativeModel('gemini-pro')
    return model

@st.cache_resource
def load_modelvision() -> genai.GenerativeModel:
    model = genai.GenerativeModel('gemini-pro-vision')
    return model

def download_html(response):
    markdown_content = f"# Produced by BDO GPT\n\n{response}"
    html_content = markdown.markdown(markdown_content)
    return html_content

#=================================================================
#CONFIGURATION
genai.configure(api_key='AIzaSyDkmaWadxYJGAgWdMVpB-qfPyhrjctrZcI')

model = load_model()
vision = load_modelvision()

if 'chat' not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = []

#=================================================================
#CHAT

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'welcome' not in st.session_state :
    welcome  = model.generate_content(f'''
    Give the User a Welcome greeting and list the things you can do such as describe images, query pdfs,
                                      make graphs using Graphiz, and talk to excel documents:
    Hello! I am BDOGPT, a Generative AI assistant designed to help you describe images, query documents,
    and make your life better :)
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

# Sidebar
with st.sidebar:
    image_attachment = st.toggle("Attach image", value=False, help="Activate this mode to attach an image and let the chatbot read it")
    txt_attachment = st.toggle("Attach text file", value=False, help="Activate this mode to attach a text file and let the chatbot read it")
    csv_excel_attachment = st.toggle("Attach CSV or Excel", value=False, help="Activate this mode to attach a CSV or Excel file and let the chatbot read it")
    graphviz_mode = st.toggle("Graphviz mode", value=False, help="Generates graphs from your prompts or Data")
    pdf_attachment = st.toggle("Upload PDF", value=False, help="Upload a PDF to query")

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

# ADDED PDF UPLOAD 
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

    if graphviz_mode:
        txt += '   Generate a graph with graphviz in .dot \n'

    if len(txt) > 5000:
        txt = txt[:5000] + '...'
    if image or url != '':
        if url != '':
            img = Image.open(requests.get(url, stream=True).raw)
        else:
            img = Image.open(image)
        prmt  = {'role': 'user', 'parts':[prompt+txt, img]}
    else:
        prmt  = {'role': 'user', 'parts':[prompt+txt]}

    append_message(prmt)

    with st.spinner('Wait a moment, I am thinking...'):
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
            append_message({'role': 'model', 'parts':response.text})
        except Exception as e:
            append_message({'role': 'model', 'parts':f'{type(e).__name__}: {e}'})

        st.rerun()
