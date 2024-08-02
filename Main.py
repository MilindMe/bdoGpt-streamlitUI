import streamlit as st
import pandas as pd
import google.generativeai as genai
import re
from PIL import Image
import requests
import markdown
import time
import os
from dotenv import load_dotenv

# PDF PROCESSING
import PyPDF2

# WORD PROCESSING
from docx import Document

# M. MEETARBHAN
# 7/23/2024
 
# ENVIRONMENT VARIABLE FOR API KEY
load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')

# === TITLE ====
st.set_page_config(
    page_title="BDOGPT",
    page_icon="🐦",
    layout="wide",
)

#------------------------------------------------------------
# HEADER
st.title('🐦‍⬛BDO-GPT')
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
    model = genai.GenerativeModel('gemini-1.5-flash')
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
        extracted_text += page.extract_text() + "\n"
    return extracted_text

# extractAudio_to_text() consumes a .mp3 file and calls the Model to convert
# the audio file to text
def extractAudio_to_text(audio_file_path):
    audio_file = genai.upload_file(path=audio_file_path)
    prompt = """
    Listen carefully to the following audio file and transcribe the contents.
    Produce a relatively lengthy summary.
    """
    response = vision.generate_content([prompt, audio_file])
    # Removes the temporary audio file path after processing 
    os.remove(audio_file_path)
    return response.text

# extract_from_word() consumes a word document and parses the text from it
def extract_from_word(doc_file):
    doc = Document(doc_file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

#=================================================================
# CONFIGURATION
# TODO : SET API KEY AS AN ENVIRONMENT VARIABLE
#genai.configure(api_key='AIzaSyDkmaWadxYJGAgWdMVpB-qfPyhrjctrZcI')

# === Billable Key === 
genai.configure(api_key='AIzaSyDkmaWadxYJGAgWdMVpB-qfPyhrjctrZcI')
# ================
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
    st.subheader("📁 Upload files")
    st.divider()
    image_attachment = st.checkbox("Image", value=False, help="Attach an image and let BDOGPT help")
    txt_attachment = st.checkbox("Text file", value=False, help="Attach a Text file and let BDOGPT help")
    csv_excel_attachment = st.checkbox("CSV or Excel", value=False, help="Attach a CSV or Excel file and let BDOGPT help")
    pdf_attachment = st.checkbox(" PDF", value=False, help="Upload a PDF to query")

    # -------------------------------------------------------------------------------------
    # Word Document Attachment
    # TODO : Merge all attachment buttons and let program dynamically determine procedures
    # -------------------------------------------------------------------------------------
    word_attachment = st.checkbox("Word Doc", value=False, help="Upload a Word Document and let BDOGPT help")

    # M.Meetarbhan ============================
    # Audio File Processing Feature
    # 7/23/2024 
    audio_attachment = st.checkbox("Audio File", value=False, help="Upload an Audio File and let BDOGPT cook up a Summary")
    # =========================================

    st.divider()
    graphviz_mode = st.toggle("Graphviz mode", value=False, help="Generates graphs from your prompts or Data")

    if image_attachment:
        image = st.file_uploader("Upload your image", type=['png', 'jpg', 'jpeg', 'pdf'])
        url = st.text_input("Or paste your image url")
    else:
        image = None
        url = ''

    # Text Upload
    if txt_attachment:
        txtattachment = st.file_uploader("Upload your text file", type=['txt'])
    else:
        txtattachment = None

    # Excel Upload
    if csv_excel_attachment:
        csvexcelattachment = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])
    else:
        csvexcelattachment = None

    # Pdf Upload
    if pdf_attachment:
        pdfattachments = st.file_uploader("Upload your PDF file", accept_multiple_files=True, type=['pdf'])
    else:
        pdfattachments = None

    # Word Document Upload    
    if word_attachment:
        wordAttachment = st.file_uploader("Upload your word file", accept_multiple_files=True, type=['docx'])
    else:
        wordAttachment = None

    # Audio Upload
    if audio_attachment:
        audio_attachment = st.file_uploader("Upload Audio File", accept_multiple_files=False, type=['mp3'])
    else: 
        audio_attachment = None
    


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
            txt += '    Datagrame: \n' + str(df)
        except:
            xls = pd.ExcelFile(csvexcelattachment)
            all_sheets_df = {sheet: xls.parse(sheet) for sheet in xls.sheet_names}
            for sheet_name, sheet_df in all_sheets_df.items():
                txt += f'   Sheet: {sheet_name}\n' + str(sheet_df) + '\n\n'
    
    if pdfattachments:
        if len(pdfattachments) > 2:
            st.error("You can upload a maximum of 2 PDFs.")
        else:
            pdf_texts = []
            for index, pdfattachment in enumerate(pdfattachments):
                pdf_title = pdfattachment.name
                pdf_text = extract_from_pdf(pdfattachment)
                pdf_texts.append(f"FILE {index + 1} ({pdf_title}):\n{pdf_text}\n")
            txt += '\n'.join(pdf_texts)

# Word Document
    if wordAttachment:
        word_texts = []
        for index, word_attachment in enumerate(wordAttachment):
            word_title = word_attachment.name
            word_text = extract_from_word(word_attachment)
            word_texts.append(f"FILE {index + 1} ({word_title}):\n{word_text}\n")
        txt += '\n'.join(word_texts)

    # AUDIO ATTACHMENT FEATURE ==============================================
    # The Audio Attachment needs to save the audio file locally in a temporary folder to provide to the Gemini API

    if audio_attachment:
        # TODO : IMPLEMENT ERROR HANDLING FOR UNSUPPORTED TYPE
        # FUNCTION SHOULD THROW AN ERROR LIKE : EXPECTED A BLOB DICT OR IMAGE TYPE BUT GOT {type(audio_file)}
        file_path = os.path.join("temp", audio_attachment.name)
        with open(file_path, "wb") as f:
            f.write(audio_attachment.getbuffer())
        txt = ' Audio File Summary : \n' + extractAudio_to_text(file_path)

    # =======================================================================


    if graphviz_mode:
        txt += '   Generate a graph with graphviz in .dot \n'

# Check : If length of text exceeds 8000, truncatenate and display with ... at the end
# TODO : DYNAMIC WORD COUNT ADJUSTMENT 
    #if len(txt) > 100000:
     #   txt = txt[:100000] + '...'

    if image or url != '':
        if url != '':
            img = Image.open(requests.get(url, stream=True).raw)
        else:
            img = Image.open(image)
        prmt = {'role': 'user', 'parts': [prompt + txt, img]}
    else:
        prmt = {'role': 'user', 'parts': [prompt + txt]}

    append_message(prmt)

    with st.spinner('Please Wait, BDO is cooking...'):
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
