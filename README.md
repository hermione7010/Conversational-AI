
Conversational-AI

# Memory-Enabled Conversational AI using Amazon Bedrock, LangChain, and LangGraph

## Overview

This project demonstrates how to build a context-aware, memory-enabled conversational assistant powered by Claude 3 (Sonnet) through Amazon Bedrock. Using LangChain and LangGraph, the assistant can maintain long-running sessions, stream live responses, and intelligently manage state across conversations.

It is designed to simulate human-like, context-rich conversations that persist across multiple interactions — ideal for e-commerce bots, AI assistants, or support agents.

---

## Features

* Maintains full conversation history and context
* Checkpointing support using LangGraph and AWS
* Runs Claude 3 Sonnet via Amazon Bedrock
* Graph-based conversational flow with modular nodes
* Real-time response streaming
* Extensible with custom tools (e.g., shoe search)

---

## Architecture

The system is built as a state machine using LangGraph:

* `load_context`: Injects background context into the state
* `extract_messages`: Retrieves user messages
* `llm`: Sends messages to Claude via Amazon Bedrock
* Session management: Saves and restores sessions with `BedrockSessionSaver`
* Checkpointing: Tracks and resumes session progress

---

## Installation

Install required packages using pip:

```bash
pip install -r requirements.txt
```

---

## Prerequisites

* An AWS account with access to Amazon Bedrock and Claude 3 Sonnet
* AWS credentials configured locally (`~/.aws/credentials`)
* Claude 3 Sonnet model access enabled in your region (e.g., `us-east-1`)

---

## Usage

1. Run the script:

```bash
python bedrock.py
```

2. Interact via the terminal. Type your messages and get real-time responses.
3. Type `quit`, `exit`, or `q` to end the session.

---

## Example

```
User: What's the best running shoe for trails?
Assistant: For trail running, I’d recommend looking into shoes with aggressive tread and great stability like the Salomon Speedcross 5.
```

---

## Extending the Assistant

* Add custom tools using the `@tool` decorator (e.g., `search_shoes`)
* Insert new nodes into the graph for logic branching or external API calls

---

## Project Goals

This system is designed to:

* Enable long, contextual AI conversations
* Simulate persistent, human-like assistants
* Provide a serverless, scalable architecture using Bedrock

---

## License

This project is provided for educational/demo purposes under the MIT License.
