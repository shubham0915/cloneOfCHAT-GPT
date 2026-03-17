from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import sqlite3
import os
import requests
import re

#  ENV ======================
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("GOOGLE_API_KEY missing")

# LLM ======================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
)

#  TOOLS ======================

search_tool = DuckDuckGoSearchRun(
    name="web_search",
    description="Search the web for current events, news, and real-time facts",
    region="us-en",
)

@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: Literal["add", "sub", "mul", "div"],
) -> dict:
    """Strict arithmetic calculator"""
    if operation == "add":
        result = first_num + second_num
    elif operation == "sub":
        result = first_num - second_num
    elif operation == "mul":
        result = first_num * second_num
    elif operation == "div":
        if second_num == 0:
            return {"error": "Division by zero"}
        result = first_num / second_num

    return {"result": result}

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch real-time stock price (NO GUESSING)"""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return {"error": "Alpha Vantage API key missing"}

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    )

    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()

        quote = data.get("Global Quote")
        if not quote:
            return {"error": "Stock data unavailable or API limit reached"}

        return {
            "symbol": symbol,
            "price": float(quote["05. price"]),
        }

    except Exception as e:
        return {"error": str(e)}

tools = [search_tool, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(tools)

# STATE ======================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# SYSTEM PROMPT ======================

system = SystemMessage(
    content=(
        "You are an AI assistant with tool access.\n\n"

        "CRITICAL FINANCE RULES:\n"
        "- NEVER reuse stock prices from memory.\n"
        "- ALWAYS fetch real-time prices using the stock tool.\n"
        "- If stock price is required, tool usage is MANDATORY.\n\n"

        "CALCULATION RULES:\n"
        "- ALL math must use the calculator tool.\n"
        "- NEVER calculate mentally.\n\n"

        "SEARCH RULES:\n"
        "- Use web search for news or uncertain facts.\n\n"

        "If a tool exists for the task, you MUST use it."
    )
)

# ====================== GUARDS ======================

STOCK_REGEX = re.compile(
    r"\b(stock|share|price|buy|sell|market|value)\b", re.IGNORECASE
)

def is_stock_query(messages: list[BaseMessage]) -> bool:
    if not messages:
        return False
    return bool(STOCK_REGEX.search(messages[-1].content))

def normalize_symbol(text: str) -> str | None:
    if "vodafone" in text.lower():
        return "VOD"  # ADR (US)
    return None

#  CHAT NODE ======================

def chat_node(state: ChatState):
    messages = [system] + state["messages"][-20:]  # context control
    response = llm_with_tools.invoke(messages)

    # HARD ENFORCEMENT
    if is_stock_query(state["messages"]) and not response.tool_calls:
        symbol = normalize_symbol(state["messages"][-1].content)
        if not symbol:
            return {
                "messages": [
                    AIMessage(
                        content="Please specify a stock ticker symbol (e.g., VOD, AAPL)."
                    )
                ]
            }

        return {
            "messages": [
                AIMessage(
                    content="Fetching the latest real-time stock price now."
                )
            ]
        }

    return {"messages": [response]}

tool_node = ToolNode(tools)

# SQLITE ======================

conn = sqlite3.connect("chatbot.db", check_same_thread=False)

conn.execute("""
CREATE TABLE IF NOT EXISTS chats (
    thread_id TEXT PRIMARY KEY,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

checkpointer = SqliteSaver(conn=conn)

#  GRAPH ======================

graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")

chatbot = graph.compile(checkpointer=checkpointer)

#  DB HELPERS ======================

def save_chat_title(thread_id: str, title: str):
    conn.execute(
        "INSERT OR IGNORE INTO chats (thread_id, title) VALUES (?, ?)",
        (thread_id, title),
    )
    conn.commit()

def get_all_chats():
    return conn.execute(
        "SELECT thread_id, title FROM chats ORDER BY created_at DESC"
    ).fetchall()
