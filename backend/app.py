# app.py - API Backend
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma
import os
import json
import uuid
import asyncio
import uvicorn
from typing import List, Optional, Dict, Any, AsyncGenerator
from config import VOYAGE_API_KEY, ANTHROPIC_API_KEY, VOYAGE_MODEL, CLAUDE_MODEL, VECTOR_DB_PATH

# Create FastAPI app
app = FastAPI(title="ZorroBot API")

# Add CORS middleware to allow requests from the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File to store conversations
CONVERSATIONS_FILE = "conversations.json"

# Enhanced persistence functions
def load_conversations():
    try:
        if os.path.exists(CONVERSATIONS_FILE):
            with open(CONVERSATIONS_FILE, 'r') as f:
                data = json.load(f)
                print(f"Loaded conversations from file: {len(data.get('conversations', {}))} conversations")
                return data
        else:
            print(f"Conversations file not found, creating default")
    except Exception as e:
        print(f"Error loading conversations: {str(e)}")
    
    # Default structure with a welcome message
    default_data = {
        "active_conversation": "default",
        "conversations": {
            "default": {
                "messages": [
                    {"role": "assistant", "content": "Hello! I'm ZorroBot, your Zorro Trading Platform assistant. How can I help you today? You can ask me about trading strategies, functions, indicators, or upload a Zorro script for analysis."}
                ],
                "script_content": None,
                "script_name": None,
                "last_sources": []
            }
        }
    }
    save_conversations(default_data)
    return default_data


# Save conversations to file with error handling
def save_conversations(data):
    try:
        with open(CONVERSATIONS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved conversations to file: {len(data.get('conversations', {}))} conversations")
    except Exception as e:
        print(f"Error saving conversations: {str(e)}")

# Get the current active conversation with proper fallback
def get_active_conversation():
    data = load_conversations()
    active_id = data["active_conversation"]
    
    # Ensure the active conversation exists
    if active_id not in data["conversations"]:
        print(f"Active conversation {active_id} not found, falling back to default")
        if "default" in data["conversations"]:
            active_id = "default"
            data["active_conversation"] = "default"
            save_conversations(data)
        else:
            # Create a default conversation if needed
            data["conversations"]["default"] = {
                "messages": [
                    {"role": "assistant", "content": "Hello! I'm ZorroBot, your Zorro Trading Platform assistant. How can I help you today? You can ask me about trading strategies, functions, indicators, or upload a Zorro script for analysis."}
                ],
                "script_content": None,
                "script_name": None,
                "last_sources": []
            }
            data["active_conversation"] = "default"
            save_conversations(data)
            active_id = "default"
    
    return data["conversations"][active_id]

# Save changes to the active conversation
def update_active_conversation(updates):
    data = load_conversations()
    active_id = data["active_conversation"]
    data["conversations"][active_id].update(updates)
    save_conversations(data)

# Model for chat messages
class Message(BaseModel):
    role: str
    content: str

# Model for chat requests
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

# Model for chat responses
class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: List[Dict[str, Any]]

# Model for new conversation
class NewConversation(BaseModel):
    title: str

# Create and cache the retrieval chain (only initialize once)
qa_chain = None

def initialize_chain():
    """Initialize the retrieval chain"""
    global qa_chain
    
    if qa_chain is not None:
        return qa_chain
    
    # Create embedding function
    embeddings = VoyageAIEmbeddings(
        voyage_api_key=VOYAGE_API_KEY,
        model=VOYAGE_MODEL,
    )
    
    # Load vector database
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )
    
    # Custom prompt template
    template = """You are ZorroBot, an expert assistant specialized in the Zorro algorithmic trading platform.
Use the following pieces of context to answer the question at the end.
If you don't know the answer or if the answer isn't in the context, say "I don't have enough information about that in my documentation." 
Don't try to make up an answer.
Always provide code examples when appropriate.
Format any code blocks with proper markdown (```c ... ```).
If the context mentions related functions or topics, suggest them at the end of your answer.

Context:
{context}

Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create LLM
    llm = ChatAnthropic(
        anthropic_api_key=ANTHROPIC_API_KEY,
        model=CLAUDE_MODEL,
        temperature=0.1,
        streaming=True,  # Enable streaming for text animation
    )
    
    # Create chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

# Initialize the chain when the app starts
@app.on_event("startup")
async def startup_event():
    initialize_chain()
    # Make sure conversations file exists
    load_conversations()

# API Endpoints

@app.get("/api/messages")
async def get_messages():
    """Get all chat messages for the active conversation"""
    active_conversation = get_active_conversation()
    return {"messages": active_conversation["messages"]}

async def generate_streaming_response(query: str) -> AsyncGenerator[str, None]:
    """Generate a streaming response from the LLM"""
    chain = initialize_chain()
    
    # Get the raw LLM to stream the response
    llm = chain.llm
    
    # Get script context if available
    active_conversation = get_active_conversation()
    script_context = ""
    if active_conversation["script_content"] is not None:
        script_context = f"Here's a Zorro script that the user has uploaded for analysis:\n```c\n{active_conversation['script_content']}\n```\n"
    
    # Combine script context and user input
    full_prompt = script_context + query
    
    # Create response generator
    async for chunk in llm.astream(full_prompt):
        yield f"data: {json.dumps({'text': chunk.content})}\n\n"
        await asyncio.sleep(0.01)  # Small delay for text animation effect

# Change this POST endpoint to a GET endpoint
@app.get("/api/chat/stream")
async def stream_chat(message: str):
    """Stream a chat response chunk by chunk for text animation"""
    # Add user message to chat history
    active_conversation = get_active_conversation()
    active_conversation["messages"].append({"role": "user", "content": message})
    update_active_conversation({"messages": active_conversation["messages"]})
    
    # Create a streaming response
    return StreamingResponse(
        generate_streaming_response(message),
        media_type="text/event-stream"
    )

@app.post("/api/chat", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    """Process a chat message and return complete response (non-streaming)"""
    # Get the chain
    chain = initialize_chain()
    
    # Add user message to chat history
    active_conversation = get_active_conversation()
    active_conversation["messages"].append({"role": "user", "content": request.message})
    update_active_conversation({"messages": active_conversation["messages"]})
    
    # Get script context if available
    script_context = ""
    if active_conversation["script_content"] is not None:
        script_context = f"Here's a Zorro script that the user has uploaded for analysis:\n```c\n{active_conversation['script_content']}\n```\n"
    
    # Combine script context and user input
    full_prompt = script_context + request.message
    
    # Generate response
    response = await chain.ainvoke({"query": full_prompt})
    answer = response["result"]
    sources = response["source_documents"]
    
    # Process sources to make them JSON serializable
    processed_sources = []
    for doc in sources:
        source_data = {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        processed_sources.append(source_data)
    
    # Add assistant response to chat history
    active_conversation["messages"].append({"role": "assistant", "content": answer})
    update_active_conversation({
        "messages": active_conversation["messages"],
        "last_sources": processed_sources
    })
    
    # Get the current conversation ID
    data = load_conversations()
    conversation_id = data["active_conversation"]
    
    return {"answer": answer, "conversation_id": conversation_id, "sources": processed_sources}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a Zorro script file"""
    # Check file type
    if not file.filename.endswith(('.c', '.cpp', '.txt')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .c, .cpp, and .txt files are allowed.")
    
    # Read file content
    content = await file.read()
    active_conversation = get_active_conversation()
    
    # Update the active conversation
    update_active_conversation({
        "script_content": content.decode(),
        "script_name": file.filename
    })
    
    return {"message": f"File {file.filename} uploaded successfully", "filename": file.filename}

@app.get("/api/script")
async def get_script():
    """Get the currently uploaded script"""
    active_conversation = get_active_conversation()
    
    if active_conversation["script_content"] is None:
        return {"script_content": None, "script_name": None}
    
    return {
        "script_content": active_conversation["script_content"], 
        "script_name": active_conversation["script_name"]
    }

@app.post("/api/clear")
async def clear_chat():
    """Clear the chat history and uploaded script for the active conversation"""
    update_active_conversation({
        "messages": [
            {"role": "assistant", "content": "Hello! I'm ZorroBot, your Zorro Trading Platform assistant. How can I help you today? You can ask me about trading strategies, functions, indicators, or upload a Zorro script for analysis."}
        ],
        "script_content": None,
        "script_name": None,
        "last_sources": []
    })
    
    return {"message": "Chat history cleared"}

@app.get("/api/conversations")
async def get_conversations():
    """Get all conversations"""
    data = load_conversations()
    return {
        "active_id": data["active_conversation"],
        "conversations": [
            {
                "id": conv_id,
                "title": f"Conversation {i+1}" if conv_id == "default" else conv_id,
                "messages": conv["messages"]
            }
            for i, (conv_id, conv) in enumerate(data["conversations"].items())
        ]
    }

@app.post("/api/conversations")
async def create_conversation(request: NewConversation):
    """Create a new conversation"""
    # Generate a unique ID
    conversation_id = str(uuid.uuid4())
    
    # Load existing conversations
    data = load_conversations()
    
    # Add new conversation
    data["conversations"][conversation_id] = {
        "messages": [
            {"role": "assistant", "content": "Hello! I'm ZorroBot, your Zorro Trading Platform assistant. How can I help you today? You can ask me about trading strategies, functions, indicators, or upload a Zorro script for analysis."}
        ],
        "script_content": None,
        "script_name": None,
        "last_sources": []
    }
    
    # Set as active conversation
    data["active_conversation"] = conversation_id
    
    # Save changes
    save_conversations(data)
    
    return {"id": conversation_id, "title": request.title}

@app.post("/api/conversations/{conversation_id}/activate")
async def activate_conversation(conversation_id: str):
    """Set the active conversation"""
    data = load_conversations()
    
    if conversation_id not in data["conversations"]:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    data["active_conversation"] = conversation_id
    save_conversations(data)
    
    return {"message": f"Conversation {conversation_id} activated"}

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    data = load_conversations()
    
    if conversation_id not in data["conversations"]:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Cannot delete the default conversation
    if conversation_id == "default":
        raise HTTPException(status_code=400, detail="Cannot delete the default conversation")
    
    # Delete the conversation
    del data["conversations"][conversation_id]
    
    # If the deleted conversation was active, switch to default
    if data["active_conversation"] == conversation_id:
        data["active_conversation"] = "default"
    
    save_conversations(data)
    
    return {"message": f"Conversation {conversation_id} deleted"}

@app.get("/api/examples")
async def get_examples():
    """Get example questions"""
    examples = [
        "How do I create a trend-following strategy?",
        "Explain how OptimalF works",
        "What are the main functions for backtesting?",
        "How to access historical price data?",
        "Explain Walk-Forward Optimization"
    ]
    
    return {"examples": examples}

# Run the app using Uvicorn
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)