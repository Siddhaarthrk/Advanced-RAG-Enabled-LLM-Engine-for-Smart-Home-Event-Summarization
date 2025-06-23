# Advanced-RAG-Enabled-LLM-Engine-for-Smart-Home-Event-Summarization

## Prerequisites

Ensure you have the following installed (If not present, install them):

* Python 3.8+
* SQLite3
* pip (Python package manager)

## Installation Steps

### 1. Clone the Repository

```sh
git@github.com:Siddhaarthrk/Advanced-RAG-Enabled-LLM-Engine-for-Smart-Home-Event-Summarization.git
```

### 2. Setup the Environment

#### Open the PowerShell or terminal in the cloned directory:

```sh
cd Advanced-RAG-Enabled-LLM-Engine-for-Smart-Home-Event-Summarization/
```

#### Create a Virtual Environment (Optional but Recommended):

```sh
python -m venv venv
```

To activate the environment:

```sh
venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies

#### Upgrade pip (if required):

```sh
python -m pip install --upgrade pip
```

#### Install all required dependencies:

```sh
pip install -r requirements.txt
```

#### If you need to manually install dependencies, use:

```sh
pip install python-dotenv sqlalchemy langchain_community langchain_ollama faiss-cpu sentence-transformers sqlparse flask
```

---

## Install and Set Up Ollama

Ollama is required to run the LLaMA and SQLCoder models. Follow these steps:

1. Install Ollama by following the instructions on [Ollama's website](https://ollama.ai/).
2. Once installed, download the required models in the command prompt/terminal:

```sh
ollama pull llama3.1:8b-instruct-q4_0
ollama pull pxlksr/defog_sqlcoder-7b-2:F16
```

---

## Running the Application

### 1. Ensure the Database and Vector Index Exist

### 2. Start the Flask Application:

```sh
python app.py
```

### 3. The server will start on:

```
http://127.0.0.1:5000/
```

## Usage

* Open `http://127.0.0.1:5000/` in your browser.
* Enter a natural language question about smart home data.
* The system will return structured insights and SQL queries.

---

## Required Python Packages

The project requires the following Python packages:

* `os`, `re`, `time`, `json` (Standard Python modules)
* `dotenv` (for loading environment variables)
* `sqlalchemy` (for SQL database interaction)
* `langchain_community.utilities.SQLDatabase`
* `langchain_ollama.ChatOllama`
* `sqlite3` (built-in, for SQLite database)
* `sqlparse` (for SQL validation)
* `faiss` (for vector database indexing)
* `sentence-transformers` (for embedding models)
* `flask` (for the web API)

## Sample Output and Workflow of this Project is included in this Repo as Sample IO and Worflow docx
