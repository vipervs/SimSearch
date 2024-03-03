import os
import arxiv
import pandas as pd
import json
import streamlit as st
from csv import writer
from openai import OpenAI
from scipy import spatial
from dotenv import load_dotenv
from glob import glob
import requests

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function for making an embedding request
def embedding_request(text):
    response = client.embeddings.create(
        input=text, model="text-embedding-3-small")
    return response

# Function to calculate relatedness between two vectors
def relatedness_function(a, b):
    return 1 - spatial.distance.cosine(a, b)

# Function to search arXiv for academic papers
def arxiv_search(query):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=10
    )
    result_list = []
    with open(f"arxiv/{query}.csv", "w") as f_object:
        writer_object = writer(f_object)
        for result in client.results(search):
            result_dict = {
                "title": result.title,
                "summary": result.summary,
                "article_url": [x.href for x in result.links][0],
                "pdf_url": [x.href for x in result.links][1],
                "published": result.published.strftime("%Y-%m-%d")
            }
            result_list.append(result_dict)
            title_embedding = embedding_request(result.title).data[0].embedding
            row = [
                result.title,
                result.summary,
                result_dict["published"],
                result_dict["pdf_url"],
                title_embedding
            ]
            writer_object.writerow(row)
    return result_list

# Function to search Google Custom Search
def google_custom_search(query):
    api_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'key': os.getenv('GOOGLE_CSE_KEY'),
        'cx': os.getenv('GOOGLE_CSE_ID')
    }
    headers = {'Accept': 'application/json'}
    response = requests.get(api_url, params=params, headers=headers)
    response.raise_for_status()
    json_data = response.json()
    items = json_data.get("items", [])
    results = []
    with open(f'cse/{query}.csv', "w") as f_object:
        csv_writer = writer(f_object) 
        for item in items:
            title = item["title"]
            link = item["link"]
            snippet = item.get("snippet")
            title_embedding = embedding_request(title).data[0].embedding
            result = {
                "title": title, 
                "link": link, 
                "snippet": snippet,
                "title_embedding": title_embedding
            }
            csv_writer.writerow([title, link, snippet, json.dumps(title_embedding)]) 
            results.append(result)
    return results

# Function to rank titles based on relatedness
def titles_ranked_by_relatedness(query, source):
    query_embedding = embedding_request(query).data[0].embedding

    if source == "arXiv":
        df = pd.read_csv(f'arxiv/{query}.csv', header=None)  
        strings_and_relatedness = [
            (row[0], row[1], row[2], row[3], relatedness_function(query_embedding, json.loads(row[4]))) 
            for i, row in df.iterrows()
        ]
        strings_and_relatedness.sort(key=lambda x: x[4], reverse=True)
    elif source == "CSE":
        df = pd.read_csv(f'cse/{query}.csv', header=None)  
        strings_and_relatedness = [ 
            (row[0], row[1], row[2], relatedness_function(query_embedding, json.loads(row[3]))) 
            for i, row in df.iterrows()
        ]
        strings_and_relatedness.sort(key=lambda x: x[3], reverse=True)
    else:
        raise ValueError(f"Invalid source: {source}")  # Handle unknown sources

    return strings_and_relatedness

# Function to fetch articles and return summaries
def fetch_articles_and_return_summary(description, source):
    if source == "arXiv":
        arxiv_search(description)
    elif source == "CSE":
        google_custom_search(description)
    else: 
        raise ValueError(f"Invalid source: {source}")

    return titles_ranked_by_relatedness(description, source) 

# Definition of tools for chat completion
tools = [
    {
        "type": "function",
        "function": {
            "name": "fetch_articles_and_return_summary",
            "description": "Use this function fetch papers and provide a summary for users",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "5 - 10 Boolean keywords that can be used for a scientific search based on the user's query"
                    }
                },
                "required": ["keywords"]
            }
        }
    },
]

st.set_page_config(page_title="✨ Paper Similarity Search 🔬")
st.title("✨ Paper Similarity Search 🔬")

search_engine = st.selectbox("Select Search Engine:", ["arXiv", "CSE"])
with st.form('search_form'):
    query = st.text_area('Enter text:', max_chars=100)
    if st.form_submit_button('Search'):
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ],
            model="gpt-3.5-turbo",
            tools=tools
        )
        print(f"User Query: {query}")
        print(f"Chat completion: {chat_completion}")
        tool_call = chat_completion.choices[0].message.tool_calls[0]
        function_name = tool_call.function.name
        arguments = tool_call.function.arguments

        if function_name == "fetch_articles_and_return_summary":
            keywords = json.loads(arguments)['keywords']
            if search_engine == "arXiv":
                st.subheader("ArXiv Results")
                with st.spinner("Searching arXiv Database..."):
                    results = fetch_articles_and_return_summary(keywords, source="arXiv") 
                for i, result in enumerate(results, start=1):
                    title, summary, published, url, score = result  
                    st.subheader(f"Result {i}: {title}")
                    st.write(f"Summary: {summary}")
                    st.write(f"Published: {published}")
                    st.write(f"URL: {url}")
                    st.write(f"Relatedness Score: {score:.2f}")
                    st.write("---") 
            elif search_engine == "CSE":
                st.subheader("Google CSE Results")
                with st.spinner("Searching Google CSE..."):
                    results = fetch_articles_and_return_summary(keywords, source="CSE")
                for i, result in enumerate(results, start=1):
                    title, snippet, url, score = result 
                    st.subheader(f"Result {i}: {title}")
                    st.write(f"Snippet: {snippet}")
                    st.write(f"URL: {url}")
                    st.write(f"Relatedness Score: {score:.2f}") 
                    st.write("---") 

# Sidebar sections
past_searches = glob('arxiv/*.csv') + glob('cse/*.csv')  # Combine search lists
past_searches_with_folder = [(os.path.dirname(file), os.path.basename(file)) for file in past_searches]
past_search_options = [(folder, file) for folder, file in past_searches_with_folder]
past_search = st.sidebar.radio("Past Searches:", past_search_options, format_func=lambda option: f"{option[0]}/{option[1]}")

if st.sidebar.button('Load Past Search'):
    selected_folder, selected_filename = past_search  # Unpack the tuple
    if selected_folder == 'arxiv':  # Compare the folder 
        source = 'arXiv'
        query = selected_filename.replace('.csv', '')  # Use just the filename
        results = titles_ranked_by_relatedness(query, source) 
        for i, result in enumerate(results, start=1):
            title, summary, published, url, score = result  
            st.subheader(f"Result {i}: {title}")
            st.write(f"Summary: {summary}")
            st.write(f"Published: {published}")
            st.write(f"URL: {url}")
            st.write(f"Relatedness Score: {score:.2f}")
            st.write("---")
    elif selected_folder == 'cse':
        source = 'CSE'
        query = selected_filename.replace('.csv', '')  # Use just the filename
        results = titles_ranked_by_relatedness(query, source) 
        for i, result in enumerate(results, start=1):
            title, link, snippet, score = result  
            st.subheader(f"Result {i}: {title}")
            st.write(f"Snippet: {snippet}")
            st.write(f"URL: {link}")
            st.write(f"Relatedness Score: {score:.2f}")
            st.write("---") 

if st.sidebar.button('Delete Selected Search'):
    selected_folder, selected_filename = past_search  # Unpack tuple
    path = os.path.join(selected_folder, selected_filename)  # Construct path
    os.remove(path)
    st.rerun()