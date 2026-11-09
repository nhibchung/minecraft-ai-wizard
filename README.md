# Minecraft AI Wizard

An interactive 3D browser game featuring an AI-powered NPC that answers Minecraft-related questions using Large Language Models (LLMs) and retrieval-based knowledge grounding.

## 🎮 Live Demo

[**Play Minecraft AI Wizard**](https://minecraft-ai-wizard.onrender.com/)

*Note: The backend is hosted on Render's free tier and may require some time to wake up after inactivity.*

---

## Overview

Minecraft AI Wizard combines Unity WebGL, a Python backend, information retrieval, and LLM inference to create an interactive AI-powered gaming experience.

Players can explore a 3D environment and interact with an AI Wizard to ask questions about Minecraft.

## Architecture

The application follows this high-level workflow:

1. **Unity WebGL Frontend:** Provides the interactive 3D game environment and NPC interface.
2. **Python Backend:** Handles incoming chat requests and coordinates retrieval and LLM inference.
3. **Knowledge Retrieval:** Uses BM25 to retrieve relevant Minecraft knowledge.
4. **LLM Inference:** Uses Groq API with the GPT-OSS-20B model to generate contextual responses.
5. **Response Delivery:** Returns generated answers to the Unity frontend.

## Technology Stack

| Component | Technology |
|---|---|
| Game Engine | Unity |
| Frontend | Unity WebGL |
| Backend | Python, FastAPI |
| Retrieval | BM25, LangChain |
| LLM | GPT-OSS-20B |
| Inference Provider | Groq |
| Deployment | Render |

## Key Features

- Interactive 3D browser-based environment.
- AI-powered NPC conversations.
- Retrieval-based knowledge grounding using Minecraft Wiki content.
- LLM integration through an external inference API.
- In-memory response caching for repeated questions.

## Running Locally

1. Clone the repository.
2. Install the Python dependencies.
3. Configure the required environment variables.
4. Start the FastAPI application.

Required environment variables:

- `GROQ_API_KEY`
- `HF_TOKEN`

Refer to the application configuration and dependency files for implementation details.

## Project Motivation

This project was built to explore practical LLM application development, including information retrieval, prompt engineering, backend integration, API consumption, and deployment.
