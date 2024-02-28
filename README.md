# Paper Similarity Search

**Description**

This script combines the power of Arxiv, Google Custom Search, OpenAI's language models, and embedding technology to provide a streamlined paper discovery and similarity search tool.

**Features**

* **Arxiv Search:** Fetches relevant papers from Arxiv based on your search query. Stores results in a CSV file for easy access.
* **Semantic Similarity:** Calculates semantic relatedness between your query and fetched research papers using OpenAI embeddings.
* **Google Custom Search:** Provides additional search results from a wider web context using Google's Custom Search Engine (CSE). 
* **Streamlit Integration:** Creates a user-friendly web interface for search and result display.

**Installation**

1. pip install -r requirements.txt 

**Setting Up Environment Variables**

1. Create a `.env` file in the project directory. 
2. Obtain API keys for:
    * OpenAI ([https://beta.openai.com/account/api-keys](https://beta.openai.com/account/api-keys))
    * Google Custom Search Engine ([https://developers.google.com/custom-search](https://developers.google.com/custom-search))
3. Add the following to your `.env` file:

   ```
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_CSE_KEY=your_cse_api_key
   GOOGLE_CSE_ID=your_cse_engine_id
   ```

**Usage**

1. Run `streamlit run app.py` (assuming your main script is named `app.py`)
2. Start searching! 

**Additional Notes**

* The script stores fetched results from Arxiv in CSV files within an 'arxiv' directory for convenient retrieval of past searches. 
* The Streamlit interface provides options to search Arxiv directly or use Google CSE, as well as load past searches. 
