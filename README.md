# Med Oryx - Medical Research Agent

MedOryx is a medical research Agent powered by BioMCP and Haystack. 

## Overview

This application leverages large language models and structured biomedical data sources to provide comprehensive research assistance. Med Oryx can help you:

- Search and analyze **clinical trials** from ClinicalTrials.gov (including protocols, outcomes, locations)
- Find **genomic variant** information from MyVariant.info
- Access **scientific literature** through PubMed/PubTator3
- Answer complex biomedical questions using natural language

## Technologies

- **[BioMCP](https://github.com/genomoncology/biomcp)**: Provides access to biomedical data sources
- **[Haystack](https://github.com/deepset-ai/haystack)**: AI orchestration framework
- **[Anthropic Claude](https://www.anthropic.com/claude)**: Large language model for natural language understanding
- **[Streamlit](https://streamlit.io/)**: Web application framework

## Setup

### Prerequisites

- Python 3.11+
- Anthropic API key
- [uv](https://github.com/astral-sh/uv) - Modern Python package manager

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/med-oryx.git
   cd med-oryx
   ```

2. Install dependencies with uv:
   ```
   uv pip install .
   ```

3. Set your Anthropic API key as an environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

1. Start the application:
   ```
   uv run streamlit run app.py
   ```

2. Open your browser and navigate to the provided URL (typically http://localhost:8501)

3. Enter your medical research questions in the chat interface

### Example Questions

The agent can answer questions like:
- Are there any studies on new approaches to treating acne?
- What are some ongoing trails for never-smokers with lung cancer?
- Tell me about clinical trials for migraine prevention.
- Are there trials for managing hay fever symptoms?
