from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3)
)
def get_chatbot_response(chatbot, prompt):
    try:
        return chatbot.predict(input=prompt)
    except Exception as e:
        if "429" in str(e):
            st.warning("Rate limit reached. Waiting before retrying...")
            time.sleep(2)  # Wait for 2 seconds before retry
            raise e
        raise e

def initialize_chatbot():
    load_dotenv()
    
    # Check if API key is set
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file!")
    
    llm = ChatGoogleGenerativeAI(
        model='gemini-1.5-pro',
        temperature=0.6,
        max_completion_tokens=2048,
        google_api_key=api_key
    )
    
    # Initialize conversation memory
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()
    
    # Create conversation chain with memory
    conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=True
    )
    
    return conversation

# Custom CSS for WhatsApp-like chat interface
st.markdown("""
<style>
.user-message {
    background-color: #dcf8c6;
    padding: 12px 16px;
    border-radius: 10px;
    margin: 5px 0;
    max-width: 70%;
    float: right;
    clear: both;
    color: #000000;
    font-size: 14px;
    line-height: 1.4;
    box-shadow: 0 0.5px 1px rgba(0,0,0,0.1);
}
.assistant-message {
    background-color: #ffffff;
    padding: 12px 16px;
    border-radius: 10px;
    margin: 5px 0;
    max-width: 70%;
    float: left;
    clear: both;
    color: #000000;
    font-size: 14px;
    line-height: 1.4;
    box-shadow: 0 0.5px 1px rgba(0,0,0,0.1);
    border: 1px solid #e9e9e9;
}
.chat-container {
    padding: 10px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    background-color: #f0f2f5;
    min-height: 400px;
    border-radius: 10px;
}
.message-container {
    width: 100%;
    overflow: hidden;
    margin-bottom: 10px;
    padding: 0 10px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hello! I'm Pinjal's AI Assistant. How can I help you today? Feel free to ask me any questions!"
    })

# App title and description
st.title("🤖 Pinjal Chatbot")
st.write("Welcome! This is a chatbot powered by pinjal patel own model.")

try:
    chatbot = initialize_chatbot()

    # Create a container for the chat
    chat_container = st.container()
    
    with chat_container:
        # Display chat history with WhatsApp-like styling
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="message-container">
                    <div class="user-message">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message-container">
                    <div class="assistant-message">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("What would you like to talk about?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display the user message with styling
        st.markdown(f"""
        <div class="message-container">
            <div class="user-message">{prompt}</div>
        </div>
        """, unsafe_allow_html=True)

        # Get bot response
        try:
            with st.spinner("Thinking..."):
                bot_response = get_chatbot_response(chatbot, prompt).strip()
                
                # Display the assistant response with styling
                st.markdown(f"""
                <div class="message-container">
                    <div class="assistant-message">{bot_response}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                
                # Force refresh to show updated messages
                st.rerun()
        except Exception as e:
            st.error("The chatbot is currently experiencing high traffic. Please try again in a few moments.")
            st.error(f"Error details: {str(e)}")

except Exception as e:
    st.error(f"Error: {str(e)}")
    if "429" in str(e):
        st.warning("You've reached the API rate limit. Please wait a few moments before trying again.")
    else:
        st.warning("Please make sure your API key is correctly set in the .env file")