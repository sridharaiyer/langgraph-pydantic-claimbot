# LangGraph Pydantic ClaimBot

A Streamlit LLM Agent application for generating auto insurance claims using LangGraph and Pydantic.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd langgraph-pydantic-claimbot
    ```
2. **Project Setup Guide (with `uv` on macOS)**
    Follow these instructions to clone the repo, install the right Python version, and set up your virtual environment using [`uv`](https://github.com/astral-sh/uv), a fast and modern Python packaging toolchain.

    Install `uv` using Homebrew:

    ```bash
    brew install uv
    ```
    Or using `curl`:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
3. **Initialize the project and install python venv and dependencies**
    ```bash
    # In the project root directory
    uv python install 3.12.7
    uv venv --python=3.12.7 .venv
    source .venv/bin/activate
    uv sync # This installs all the dependencies from the pyproject.toml in the .venv

4.  **Set the OpenAI API key:**

    ```bash
    export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    ```

    Replace `"YOUR_OPENAI_API_KEY"` with your actual OpenAI API key.
5.  **Run the Streamlit application:**

    ```bash
    streamlit run chatbot.py
    ```

## Project Structure

*   `chatbot.py`: The main Streamlit application file.
*   `langgraph_workflow.py`: Defines the LangGraph workflow.
*   `nodes.py`: Contains the LangGraph nodes for analyzing messages and generating responses.
*   `agents.py`: Defines the Pydantic AI agents for intent detection and claim extraction.
*   `models.py`: Defines the Pydantic models for intent and claim data.
*   `prettify.py`: Contains utility functions for formatting output.
*   `pyproject.toml`: Specifies the project dependencies.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
*   `README.md`: This file, providing an overview of the project.

## Usage

1.  Run the Streamlit application using the command `streamlit run chatbot.py`.
2.  The application will open in your web browser.
3.  Enter your query in the chat input field and press Enter.
4.  The AI assistant will process your request and display the response.
5.  You can view the processing steps by expanding the "Show Processing Steps" section in the chat messages.
