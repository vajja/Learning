import streamlit as st
import pandas as pd
from llm_connections import LLMClient
from qwen_audio import QwenSpeech2Text

global provider, model
global prev_provider, prev_model
global temperature, prev_temperature, speech_status, speech_text
# global llm_client
# llm_client = None
provider = None
model = None
prev_model = None
prev_provider = None
temperature = 0.05
prev_temperature = 0.05
speech_status = False
speech_text = ""

with st.sidebar:

    provider = st.selectbox(
        "Choose model",
        ("OpenAI", "Claude", "XAI"),
    )

    st.write("You selected:", provider)

    if provider == "OpenAI":

        confusion_matrix = pd.DataFrame(
            {
                "model": ["gpt-4o-mini", "gpt-4o-mini-realtime-preview", "gpt-5-mini",
                          "gpt-5-chat-latest", "gpt-5.1-chat-latest", "gpt-5.1-codex",
                          "gpt-5.1-codex-max"],
                "Input": ["$0.15", "$0.60", "$0.25", "$1.25", "$1.25", "$1.25", "$1.25"],
                "Output": ["$0.60", "$2.40", "$2.00", "$10.00", "$10.00", "$10.00", "$10.00"]
            },
            index=[1, 2, 3, 4, 5, 6, 7],
        )
        st.table(confusion_matrix)
        model = st.selectbox("Select a model",
                             ("gpt-4o-mini", "gpt-4o-mini-realtime-preview", "gpt-5-mini",
                          "gpt-5-chat-latest", "gpt-5.1-chat-latest", "gpt-5.1-codex",
                          "gpt-5.1-codex-max"))
    elif provider=="Claude":
        confusion_matrix = pd.DataFrame(
            {
                "model": ["Claude Haiku 3.5", "Claude Haiku 4.5", "Claude Sonnet 4", "Claude Sonnet 4.5",
                          "Claude Opus 4.5", "Claude Opus 4.6"],
                "Input": ["$0.80", "$1.0", "$3.00", "$3.00", "$5.00", "$5.00"],
                "Output": ["$4.00", "$5.0", "$15.00", "$15.00", "$25.00", "$25.00"]
            },
            index=[1, 2, 3, 4, 5, 6],
        )
        st.table(confusion_matrix)
        model = st.selectbox("Select a model",
                             ("claude-3-5-haiku-latest", "claude-haiku-4-5-20251001", "claude-sonnet-4", "claude-sonnet-4-5",
                              "claude-opus-4-5", "claude-opus-4-6"))
    elif provider=="XAI":
        st.header("XAI")
        confusion_matrix = pd.DataFrame(
            {
                "model": ["grok-3-mini", "grok-4-fast-non-reasoning", "grok-4-fast-reasoning", "grok-4-1-fast-non-reasoning",
                          "grok-4-1-fast-reasoning", "grok-code-fast-1"],
                "Input": ["$0.30", "$0.20", "$0.20", "$0.20", "$0.20", "$0.20"],
                "Output": ["$0.50", "$0.50", "$0.50", "$0.50", "$0.50", "$1.50"]
            },
            index=[1, 2, 3, 4, 5, 6],
        )
        st.table(confusion_matrix)
        model = st.selectbox("Select a model",
                             ("grok-3-mini", "grok-4-fast-non-reasoning",
                              "grok-4-fast-reasoning", "grok-4-1-fast-non-reasoning",
                              "grok-4-1-fast-reasoning", "grok-code-fast-1"))

    temperature = st.slider("Temperature", 0.0, 1.0, 0.05)


tab1, tab2 = st.tabs(["Speech2text", "text"])

# st.header("Custom LLM's")


# React to user input
@st.cache_resource
def get_client():
    # global llm_client
    # if llm_client is None:
    llm_client = LLMClient()
    llm_client.connect_openai(model="gpt-4o-mini", temperature=0.05)
    llm_client.connect_anthropic(model="claude-3-5-haiku-latest", temperature=0.05)
    llm_client.connect_grok(model="grok-4-1-fast-reasoning", temperature=0.05)
    return llm_client

@st.cache_resource
def get_speech2text():
    qwen_instance = QwenSpeech2Text.get_instance()
    return qwen_instance

models = {"OpenAI": {"model": "gpt-4o-mini", "temperature": 0.05},
          "Claude": {"model": "claude-3-5-haiku-latest", "temperature": 0.05},
          "XAI":{"model": "grok-4-1-fast-reasoning", "temperature": 0.05}}

get_client()
prev_provider = "OpenAI"
prev_model = "gpt-4o-mini"


def update_model(provider, model, temperature):
    llm_client = get_client()
    # if llm_client is None:
    #     connect_clients()
    if provider=="OpenAI":
        print(f"updated open ai mode: {model} and temperature: {temperature}")
        llm_client.connect_openai(model=model, temperature=temperature)
    elif provider=="Claude":
        print(f"updated Claude ai mode: {model} and temperature: {temperature}")
        llm_client.connect_anthropic(model=model, temperature=temperature)
    elif provider=="XAI":
        print(f"updated Grok ai mode: {model} and temperature: {temperature}")
        llm_client.connect_grok(model=model, temperature=temperature)

def query_model(provider, query):
    llm_client = get_client()
    # if llm_client is None:
    #     connect_clients()
    output = "Error"
    if provider=="OpenAI":
        print("OpenAI query")
        output = llm_client.query_openai(query)
    elif provider=="Claude":
        print("Claude query")
        output = llm_client.query_anthropic(query)
    elif provider=="XAI":
        print("XAI Query")
        output = llm_client.query_grok(query)
    return output
if "messages" not in st.session_state:
    st.session_state.messages = []


with tab1:
    st.header("Speech to text")
    audio_value = st.audio_input("Record a voice message")
    global audio_inp
    audio_inp = False
    if audio_value:
        st.audio(audio_value)
        with open("streamlit_exp/test_streamlit.wav", "wb") as f:
            f.write(audio_value.getbuffer())
        st.success("Audio saved as test_streamlit.wav")
        qwen_instance = get_speech2text()
        speech_text = qwen_instance.speech_text("streamlit_exp/test_streamlit.wav")
        speech_status = True
        st.write(speech_text)
        # audio_inp = True
with tab2:
    st.header("chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("What is up?"):
        # audio_value = st.audio_input("Record a voice message")
        # if prompt == "use speech":
        #     prompt = speech_text
        audio_inp = False
        speech_text = ""
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        print(f"Query: {prompt}")
        update = False
        print(f"provider: {provider}, model: {model}, temperature: {temperature}")
        print(f"prev_provider: {prev_provider}, prev_model: {prev_model}, prev_temperature: {prev_temperature}")
        if provider!=prev_provider or prev_model!=model or prev_temperature!=temperature:
            prev_model = model
            prev_provider = provider
            update = True
        if update:
            if models[prev_provider]["model"]!=prev_model or \
                    models[prev_provider]["temperature"]!=temperature:
                # updating
                models[prev_provider]["model"] = prev_model
                models[prev_provider]["temperature"] = temperature
                update_model(prev_provider, prev_model, temperature)
        output = "Error"
        with st.chat_message("system"):
            output = query_model(prev_provider, prompt)
            st.markdown(output)
        print(f"Output: {output}")
        with open("current_results.txt", "w+") as f:
            f.write("\n\n\n")
            f.write(f"{output}")
        st.session_state.messages.append({"role": "System", "content": output})

# 1: 10 - nanny
# 2:

