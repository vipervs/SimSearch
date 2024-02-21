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

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embedding_request(text):
    response = client.embeddings.create(
        input=text, model="text-embedding-3-small")
    return response

def relatedness_function(a, b):
    return 1 - spatial.distance.cosine(a, b)

def get_articles(query):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=10
    )
    result_list = []

    with open(f"arxiv/{query}.csv", "w") as f_object:  # Save the file in the 'arxiv' folder
        writer_object = writer(f_object)
        f_object.close()

    for result in client.results(search):
        result_dict = {}
        result_dict.update({"title": result.title})
        result_dict.update({"summary": result.summary})
        result_dict.update({"article_url": [x.href for x in result.links][0]})
        result_dict.update({"pdf_url": [x.href for x in result.links][1]})
        result_list.append(result_dict)

        title_embedding = embedding_request(result.title).data[0].embedding

        row = [
            result.title,
            result.summary,
            result_dict["pdf_url"],
            title_embedding
        ]

        with open(f'arxiv/{query}.csv', "a") as f_object:  # Save the file in the 'arxiv' folder
            writer_object = writer(f_object)
            writer_object.writerow(row)
            f_object.close()

    return result_list

def titles_ranked_by_relatedness(query):
  query_embedding = embedding_request(query).data[0].embedding
  df = pd.read_csv(f'arxiv/{query}.csv', header=None)  # Load the file from the 'arxiv' folder
  strings_and_relatedness = [
      (row[0], row[1], row[2], relatedness_function(query_embedding, json.loads(row[3]))) for i, row in df.iterrows()
  ]

  strings_and_relatedness.sort(key=lambda x: x[3], reverse=True)

  return strings_and_relatedness

def fetch_articles_and_return_summary(description):
    get_articles(description)
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

st.title("arXiv Paper Similarty Search")

query = st.text_input("Search Query")

# List all .csv files in the 'arxiv' directory
files = glob('arxiv/*.csv')

past_search = st.sidebar.radio("Last searches: ", files)

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
        results = fetch_articles_and_return_summary(json.loads(arguments)['keywords'])
        for i, result in enumerate(results, start=1):
            title, summary, url, score = result  # Unpack the result assuming it's a tuple
            st.subheader(f"Result {i}: {title}")
            st.write(f"Summary: {summary}")
            st.write(f"URL: {url}")
            st.write(f"Relatedness Score: {score:.2f}")
            st.write("---")

if st.sidebar.button('Load Past Search'):
    results = titles_ranked_by_relatedness(past_search.replace('arxiv/', '').replace('.csv', ''))
    for i, result in enumerate(results, start=1):
        title, summary, url, score = result  # Unpack the result assuming it's a tuple
        st.subheader(f"Result {i}: {title}")
        st.write(f"Summary: {summary}")
        st.write(f"URL: {url}")
        st.write(f"Relatedness Score: {score:.2f}")
        st.write("---")

if st.sidebar.button('Delete Selected Search'):
    os.remove(past_search)
