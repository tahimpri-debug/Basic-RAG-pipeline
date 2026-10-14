# Basic RAG Assistant

A command-line RAG (Retrieval-Augmented Generation) system that answers questions from your own text file. It finds the most relevant parts of the document and gives them to an LLM, so the answer is based on your data instead of the model's general knowledge.

Built with **LangChain**, **Hugging Face embeddings**, **Chroma**, and **LangChain Expression Language (LCEL)**.

## Features

- Loads any `.txt` file as the knowledge source
- Splits the text into chunks with a recursive text splitter
- Creates embeddings locally with a free Hugging Face model (no embedding API cost)
- Stores vectors in a persistent Chroma database, so the file is embedded only once
- Retrieves the 3 most relevant chunks for each question
- Generates a short, factual answer with an LLM (OpenAI or OpenRouter)
- Simple chat loop in the terminal

## Tech stack

| Part | Tool |
|------|------|
| Language | Python 3 |
| Framework | LangChain (LCEL) |
| Chunking | `RecursiveCharacterTextSplitter` (chunk size 1000, overlap 200) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face, runs locally) |
| Vector store | Chroma |
| LLM | `ChatOpenAI` (works with OpenAI or OpenRouter) |

## How it works

```
knowlage.txt
   -> TextLoader (load the file)
   -> RecursiveCharacterTextSplitter (cut into chunks)
   -> HuggingFaceEmbeddings (turn chunks into vectors)
   -> Chroma (store and search vectors)

Your question
   -> retriever (find the top 3 matching chunks)
   -> prompt template (insert chunks + question)
   -> LLM (write the answer)
   -> StrOutputParser (plain text)
```

On the first run the document is chunked, embedded, and saved into the `chroma_db` folder. On later runs the saved database is loaded directly, so startup is fast.

## Requirements

- Python 3.10 or newer
- An API key for OpenAI or OpenRouter (the embeddings are free and run locally)

## Setup

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd chunking-and-embeding
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project folder (see `.env.example`):

   **Using OpenAI:**
   ```
   OPENAI_API_KEY=your_openai_key
   ```

   **Using OpenRouter:**
   ```
   OPENAI_API_KEY=your_openrouter_key
   OPENAI_BASE_URL=https://openrouter.ai/api/v1
   MODEL_NAME=openai/gpt-4o-mini
   ```

## Usage

Run the assistant:

```bash
python calling_and_enbeding.py
```

The first run downloads the embedding model (about 90 MB) and builds the database, so it takes longer. After that it starts quickly.

Example:

```
=== RAG Assistant ===
Ask about the document. Type 'exit' to quit.

You: What time does Daniel wake up?
AI: Daniel wakes up at 5:30 every morning.

You: exit
Goodbye!
```

## Using your own document

1. Replace the text in `knowlage.txt` with your own content (or change the file path in the code)
2. **Delete the `chroma_db` folder**
3. Run the script again

The database is only built when the `chroma_db` folder does not exist. If you edit the text file without deleting the folder, the old data is still used. The same applies if you change the embedding model.

## Project structure

```
calling_and_enbeding.py   # main script
knowlage.txt              # knowledge source (your document)
requirements.txt          # Python dependencies
.env.example              # template for required environment variables
.gitignore                # keeps .env, .venv, and chroma_db out of Git
README.md                 # this file
```

Created locally and not pushed to GitHub: `.env` (your real API key), `.venv/` (virtual environment), and `chroma_db/` (generated database).

## Security note

Never commit your `.env` file. It is listed in `.gitignore`. If an API key is ever exposed, revoke it and create a new one right away.

## What I learned building this

- Chunking a document and why chunk overlap matters
- Turning text into embeddings and storing them in a vector database
- Retrieving similar chunks for a question
- Connecting retrieval and generation with an LCEL pipeline
- Keeping secrets safe with environment variables

## Status

Working basic RAG pipeline. Possible next steps: support PDF files, show the source chunks with each answer, and add a way to re-index when the document changes.
