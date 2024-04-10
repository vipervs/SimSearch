import os
import arxiv
import pandas as pd
import json
import streamlit as st
from csv import writer
import ollama
from scipy import spatial
from dotenv import load_dotenv
from glob import glob
import requests
import anthropic

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Function for making an embedding request
def embedding_request(text):
    response = ollama.embeddings(model='nomic-embed-text:latest', prompt=text)
    if isinstance(response, list):
        if isinstance(response[0], dict):
            embedding = response[0].get('embedding')
        else:
            embedding = response[0]
    elif isinstance(response, dict):
        embedding = response.get('embedding')
    else:
        raise ValueError("Unexpected response format from ollama.embeddings")
    #print(f"Ollama API response: {response}")
    return embedding

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
    with open(f"arxiv/{query}.csv", "w", newline='') as f_object:
        writer_object = writer(f_object)
        query_embedding = embedding_request(query)
        for result in client.results(search):
            title_embedding = embedding_request(result.title)
            relatedness_score = relatedness_function(query_embedding, title_embedding)
            result_dict = {
                "title": result.title,
                "summary": result.summary,
                "article_url": [x.href for x in result.links][0],
                "pdf_url": [x.href for x in result.links][1],
                "published": result.published.strftime("%Y-%m-%d"),
                "relatedness_score": relatedness_score
            }
            result_list.append(result_dict)
            writer_object.writerow([result.title, result.summary, result_dict["published"], result_dict["pdf_url"], title_embedding, relatedness_score])
    return result_list

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

            # Calculate relatedness
            query_embedding = embedding_request(query)
            text_for_embedding = f"{title} {snippet}"
            embedding = embedding_request(text_for_embedding)
            relatedness_score = relatedness_function(query_embedding, embedding) 

            result = {
                "title": title, 
                "link": link, 
                "snippet": snippet,
                "embedding": embedding,
                "relatedness_score": relatedness_score
            }
            csv_writer.writerow([title, link, snippet, json.dumps(embedding), relatedness_score])
            results.append(result)
    return results

# Function to rank titles based on relatedness
def titles_ranked_by_relatedness(query, source):
    query_embedding = embedding_request(query)

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

st.set_page_config(page_title="Paper Similarity Search 🔬")
st.title("Paper Similarity Search 🔬")

search_engine = st.selectbox("Select Search Engine:", ["arXiv", "CSE"])

with st.form('search_form'):
    query = st.text_area('Enter text:', max_chars=100)
    if st.form_submit_button('Search'):
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=50,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Generate keywords for the following query in a boolean format suitable for a scientific search and return only the keywords,\n\n<query>{query}</query>"
                        }
                    ]
                }
            ]
        )
        keywords = message.content[0].text
        #print(f"Generated Keywords: {keywords}")
        if search_engine == "arXiv":
            st.header(f"📚 ArXiv Results: {keywords}")
            with st.spinner("Searching arXiv Database..."):
                results = arxiv_search(keywords)
            for i, result in enumerate(results, start=1):
                title, summary, published, url, score = result['title'], result['summary'], result['published'], result['pdf_url'], result['relatedness_score']
                st.subheader(f"Result {i}: {title}")
                st.write(f"Summary: {summary}")
                st.write(f"Published: {published}")
                st.write(f"URL: {url}")
                st.write(f"Relatedness Score: {score:.2f}")
                st.write("---")
        elif search_engine == "CSE":
            st.header(f"📚 Google CSE Results: {keywords}")
            with st.spinner("Searching Google CSE..."):
                results = google_custom_search(keywords)
            for i, result in enumerate(results, start=1):
                title, snippet, url, score = result['title'], result['snippet'], result['link'], result['relatedness_score']
                st.subheader(f"Result {i}: {title}")
                st.write(f"Snippet: {snippet}")
                st.write(f"URL: {url}")
                st.write(f"Relatedness Score: {score:.2f}")
                st.write("---")

# Sidebar sections
st.sidebar.header("Past Searches 📚")
past_searches = glob('arxiv/*.csv') + glob('cse/*.csv')
past_searches_with_folder = [(os.path.dirname(file), os.path.basename(file)) for file in past_searches]
past_search_options = [(folder, file) for folder, file in past_searches_with_folder]

# Group searches by source
searches_by_source = {}
for file_path in past_searches:
    folder = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    source = folder.split('/')[-1]  # Extract source from folder name
    if source not in searches_by_source:
        searches_by_source[source] = []
    searches_by_source[source].append((folder, file_name))

for source, searches in searches_by_source.items():
    with st.sidebar.expander(source):
        for folder, file_name in searches:
            search_label = f"{folder}/{file_name}"
            col1, col2 = st.columns([8, 2])
            with col1:
                if st.checkbox(search_label, key=search_label):
                    if folder == 'arxiv':
                        query = file_name.replace('.csv', '')
                        # Trigger a reload by updating a Streamlit session state variable
                        if 'load_arxiv_results' not in st.session_state:
                            st.session_state['load_arxiv_results'] = query
                    elif folder == 'cse':
                        query = file_name.replace('.csv', '')
                        # Trigger a reload for CSE
                        if 'load_cse_results' not in st.session_state:
                            st.session_state['load_cse_results'] = query
            with col2:
                if st.button("🗑️", key=f"delete_{search_label}"):
                    file_path = os.path.join(folder, file_name)
                    try:
                        os.remove(file_path)
                        st.success(f"Deleted: {search_label}")
                        st.rerun()  # Rerun the app to update the sidebar
                    except FileNotFoundError:
                        st.warning(f"File not found: {search_label}")

# Main window logic to display results based on session state
if 'load_arxiv_results' in st.session_state:
    query = st.session_state['load_arxiv_results']
    file_path = os.path.join('arxiv', query + '.csv') 
    try:
        df = pd.read_csv(file_path, header=None)
        df = df.sort_values(by=5, ascending=False)  
        st.header(f"📚 ArXiv Results: {query}")
        for i, row in df.iterrows():
            title, summary, published, url, embedding, score = row[0], row[1], row[2], row[3], row[4], row[5]
            st.subheader(f"Result {i + 1}: {title}")
            st.write(f"Summary: {summary}")
            st.write(f"Published: {published}")
            st.write(f"URL: {url}")
            st.write(f"Relatedness Score: {score:.2f}")
            st.write("---")
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        st.warning(f"Error loading past search: {e}")
    finally:
        del st.session_state['load_arxiv_results']  # Clear the state after results are displayed

if 'load_cse_results' in st.session_state:
    query = st.session_state['load_cse_results']
    file_path = os.path.join('cse', query + '.csv') 
    try:
        df = pd.read_csv(file_path, header=None)
        df = df.sort_values(by=4, ascending=False) 
        st.header(f"📚 CSE Results: {query}")
        for i, row in df.iterrows():
            title, link, snippet, embedding, score = row[0], row[1], row[2], row[3], row[4] 
            st.subheader(f"Result {i + 1}: {title}")
            st.write(f"Snippet: {snippet}")
            st.write(f"URL: {link}")
            st.write(f"Relatedness Score: {score:.2f}") 
            st.write("---")
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        st.warning(f"Error loading past search: {e}")
    finally:
        del st.session_state['load_cse_results'] 
