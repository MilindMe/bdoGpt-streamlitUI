import streamlit as st
import pandas as pd
import google.generativeai as genai
import re
from PIL import Image
import requests
import PyPDF2

st.set_page_config(
    page_title="BDOGPT_gemini",
    page_icon="https://seeklogo.com/images/G/google-ai-logo-996E85F6FD-seeklogo.com.png",
    layout="wide",
)

#------------------------------------------------------------
#HEADER
st.markdown('''
🐦BDO-GPT''', unsafe_allow_html=True)
st.caption("With ❤️ by your Intern")

#------------------------------------------------------------
#LANGUAGE
langcols = st.columns([0.2,0.8])
with langcols[0]:
    lang = st.selectbox('Select your language', ('English',))

if 'lang' not in st.session_state:
    st.session_state.lang = lang
st.divider()

#------------------------------------------------------------
#FUNCTIONS
def extract_graphviz_info(text: str) -> list[str]:
    """
    The function `extract_graphviz_info` takes in a text and returns a list of graphviz code blocks found in the text.

    :param text: The `text` parameter is a string that contains the text from which you want to extract Graphviz information
    :return: a list of strings that contain either the word "graph" or "digraph". These strings are extracted from the input
    text.
    """

    graphviz_info  = text.split('```')

    return [graph for graph in graphviz_info if ('graph' in graph or 'digraph') and ('{' in graph and '}' in graph)]

def append_message(message: dict) -> None:
    """
    The function appends a message to a chat session.

    append_message(message) is a function that takes in a dictionary (message) that represents a chat message.
    The dictionary contains info on the user that sent the message and message content.
    # Consumes dictionary
    # Return : Nil
    """
    st.session_state.chat_session.append({'user': message})
    return

@st.cache_resource
def load_model() -> genai.GenerativeModel:
    """
    The function `load_model()` returns an instance of the `genai.GenerativeModel` class initialized with the model name
    'gemini-pro'.
    :return: an instance of the `genai.GenerativeModel` class.
    """
    model = genai.GenerativeModel('gemini-pro')
    return model

@st.cache_resource
def load_modelvision() -> genai.GenerativeModel:
    """
    The function `load_modelvision` loads a generative model for vision tasks using the `gemini-pro-vision` model.
    :return: an instance of the `genai.GenerativeModel` class.
    """
    model = genai.GenerativeModel('gemini-pro-vision')
    return model

def read_pdf(file) -> str:
    """
    The function `read_pdf` takes in a file object and returns the text extracted from the PDF file.

    :param file: The `file` parameter is a file object representing the PDF file to be read.
    :return: The extracted text from the PDF file as a string.
    """
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

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

if 'welcome' not in st.session_state or lang != st.session_state.lang:
    st.session_state.lang = lang
    welcome  = model.generate_content(f'''
    Give a welcome greeting to the user and suggest what they can do
    (You can describe images, answer questions, read text files, read tables, generate graphs with graphviz, etc.)
    You are a chatbot in a chat application created in Streamlit and Python. generate the answer in {lang}''')
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
        else:
            with st.chat_message('user'):
                st.write(message['user']['parts'][0])
                if len(message['user']['parts']) > 1:
                    st.image(message['user']['parts'][1], width=200)
        count += 1

cols = st.columns(5)

with cols[0]:
    image_attachment = st.toggle("Attach image", value=False, help="Activate this mode to attach an image and let the chatbot read it")

with cols[1]:
    txt_attachment = st.toggle("Attach text file", value=False, help="Activate this mode to attach a text file and let the chatbot read it")
with cols[2]:
    csv_excel_attachment = st.toggle("Attach CSV or Excel", value=False, help="Activate this mode to attach a CSV or Excel file and let the chatbot read it")
with cols[3]:
    graphviz_mode = st.toggle("Graphviz mode", value=False, help="Activate this mode to generate a graph with graphviz in .dot from your message")
with cols[4]:
    pdf_attachment = st.toggle("Attach PDF", value=False, help="Activate this mode to attach a PDF file and let the chatbot read it")

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
        txt += '   PDF file: \n' + read_pdf(pdfattachment)

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
