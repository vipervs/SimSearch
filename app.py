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

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embedding_request(text):
    response = client.embeddings.create(
        input=text, model="text-embedding-3-small")
    return response

def relatedness_function(a, b):
    return 1 - spatial.distance.cosine(a, b)

def arxiv_search(query):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=10
    )
    result_list = []

    with open(f"arxiv/{query}.csv", "w") as f_object:
        writer_object = writer(f_object)
        f_object.close() 

    for result in client.results(search):
        result_dict = {}
        result_dict.update({"title": result.title})
        result_dict.update({"summary": result.summary})
        result_dict.update({"article_url": [x.href for x in result.links][0]})
        result_dict.update({"pdf_url": [x.href for x in result.links][1]})
        result_dict.update({"published": result.published.strftime("%Y-%m-%d")})

        result_list.append(result_dict)

        title_embedding = embedding_request(result.title).data[0].embedding

        row = [
            result.title,
            result.summary,
            result_dict["published"],
            result_dict["pdf_url"],
            title_embedding
        ]

        with open(f'arxiv/{query}.csv', "a") as f_object:  
            writer_object = writer(f_object)
            writer_object.writerow(row)
            f_object.close()

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
    results = [{"title": item["title"], "link": item["link"]} for item in items]
    return results[:20]  # Limit to 10 results

def titles_ranked_by_relatedness(query):
  query_embedding = embedding_request(query).data[0].embedding
  df = pd.read_csv(f'arxiv/{query}.csv', header=None)  
  strings_and_relatedness = [
      (row[0], row[1], row[2], row[3], relatedness_function(query_embedding, json.loads(row[4]))) for i, row in df.iterrows()
  ]

  strings_and_relatedness.sort(key=lambda x: x[4], reverse=True)

  return strings_and_relatedness

def fetch_articles_and_return_summary(description):
    arxiv_search(description)
    return titles_ranked_by_relatedness(description)

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
                        "description": "Some keywords that can be used for a Arxiv search based on the user's query"
                    }
                },
                "required": ["keywords"]
            }
        }
    },
]

st.title("Paper Similarty Search")
query = st.text_input("Search Query")
files = glob('arxiv/*.csv')
past_search = st.sidebar.radio("Last searches: ", [os.path.basename(file).replace('.csv', '') for file in files])

search_engine = st.sidebar.selectbox("Select Search Engine:", ["arXiv", "CSE"])

if st.button('Search'):
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

    tool_call = chat_completion.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = tool_call.function.arguments

    if function_name == "fetch_articles_and_return_summary": 
        keywords = json.loads(arguments)['keywords']

        if search_engine == "arXiv":
            results = fetch_articles_and_return_summary(keywords)
            for i, result in enumerate(results, start=1):
                title, summary, published, url, score = result
                st.subheader(f"Result {i}: {title}")
                st.write(f"Summary: {summary}")
                st.write(f"Published: {published}")
                st.write(f"URL: {url}")
                st.write(f"Relatedness Score: {score:.2f}")
                st.write("---") 

        elif search_engine == "CSE":
            with st.spinner("Searching Google CSE..."):
                google_results = google_custom_search(keywords)

            st.subheader("Google CSE Results")  
            for i, result in enumerate(google_results, start=1):
                st.subheader(f"Result {i}: {result['title']}")
                st.write(f"URL: {result['link']}")
                st.write("---") 

if st.sidebar.button('Load Past Search'): # Loads the results of a previous search
    results = titles_ranked_by_relatedness(past_search.replace('arxiv/', '').replace('.csv', ''))
    for i, result in enumerate(results, start=1):
        title, summary, published, url, score = result  
        st.subheader(f"Result {i}: {title}")
        st.write(f"Summary: {summary}")
        st.write(f"Published: {published}")
        st.write(f"URL: {url}")
        st.write(f"Relatedness Score: {score:.2f}")
        st.write("---")

if st.sidebar.button('Delete Selected Search'):
    os.remove(os.path.join('arxiv', past_search + '.csv')) 
    st.rerun() 

