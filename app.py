import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import numpy as np
import nltk
import os

# Load environment variables
load_dotenv(".env")

# Install NLTK library
nltk.download("punkt")

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Function to calculate chat completion
def chat_completion(
    model,
    messages,
    max_tokens=500,
    temperature=0.7,
    top_p=1,
    stream=False,
    on_stream=None,
):
    completion = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
    )

    if stream:
        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                on_stream(chunk.choices[0].delta.content)
                response = response + chunk.choices[0].delta.content

        return response

    return completion.choices[0].message.content.strip(), completion


# Range: 0 - 1
def average_cosine_similarity(outputs):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(outputs)
    cos_sim_matrix = cosine_similarity(tfidf_matrix)

    # Exclude diagonal elements and calculate the average similarity
    np.fill_diagonal(cos_sim_matrix, 0)
    avg_sim = np.sum(cos_sim_matrix) / (cos_sim_matrix.size - len(outputs))

    return avg_sim


# Range: 0 - 1
def average_bleu_scores(outputs):
    scores = []
    smoothie = SmoothingFunction().method4  # Using smoothing method 4 as an example
    for i, candidate in enumerate(outputs):
        references = [outputs[:i] + outputs[i + 1 :]]
        tokenized_candidate = word_tokenize(candidate)
        tokenized_references = [word_tokenize(ref) for ref in references[0]]
        score = sentence_bleu(
            tokenized_references, tokenized_candidate, smoothing_function=smoothie
        )
        scores.append(score)

    return sum(scores) / len(scores)


def average_jaccard_similarity(outputs):
    scores = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            set1 = set(outputs[i].split())
            set2 = set(outputs[j].split())
            intersection = set1.intersection(set2)
            union = set1.union(set2)
            score = len(intersection) / len(union)
            scores.append(score)

    return sum(scores) / len(scores)


# def average_levenshtein_distance(outputs):
#     scores = []
#     for i in range(len(outputs)):
#         for j in range(i+1, len(outputs)):
#             dist = levenshtein_distance(outputs[i], outputs[j])
#             max_len = max(len(outputs[i]), len(outputs[j]))
#             normalized_dist = dist / max_len
#             scores.append(normalized_dist)

#     return np.mean(scores)

# print(average_levenshtein_distance(outputs))


def average_sequence_matcher_distance(outputs):
    scores = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            matcher = SequenceMatcher(None, outputs[i], outputs[j])
            scores.append(1 - matcher.ratio())  # Subtract from 1 to represent distance

    return np.mean(scores)


# Streamlit app
st.title("Prompt Fuzz Testing")

n = st.number_input("Generation count", value=5, min_value=1)
model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"], index=0)
temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=1.0)
top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=1.0)
max_tokens = st.slider("Max Tokens", min_value=0, max_value=5000, value=500)

# Initialize messages in session state if not present
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": ""}]

# Use session state for initial system message input
initial_message = st.text_input(
    "Enter your initial system message", st.session_state["messages"][0]["content"]
)
# Update the initial message in session state
st.session_state["messages"][0]["content"] = initial_message

# Initialize a counter in session state for generating unique keys if not present
if "message_counter" not in st.session_state:
    st.session_state["message_counter"] = 0

# Existing code for adding a new message
# add_message = st.button("Add another message")
# if add_message:
#     # Increment the counter to ensure a unique key for the new message
#     st.session_state["message_counter"] += 1
#     unique_key = st.session_state["message_counter"]
#     new_role = st.selectbox(
#         "Select role for new message",
#         ("system", "user", "assistant"),
#         key=f"new_role_{unique_key}",
#     )
#     new_content = st.text_input(
#         "Enter the content for the new message", key=f"new_content_{unique_key}"
#     )
#     # Append new message to session state
#     st.session_state["messages"].append({"role": new_role, "content": new_content})

# Use session state for messages throughout your app
messages = st.session_state["messages"]


# Function to fetch chat completion
def fetch_completion(index):
    completion, _ = chat_completion(
        model, messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )
    return completion


# Generate outputs
if st.button("Generate"):
    with st.spinner("Generating completions... Please wait"):
        outputs = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_completion = {
                executor.submit(fetch_completion, i): i
                for i in range(n)  # Adjusted to use 'n' for dynamic generation count
            }
            for future in concurrent.futures.as_completed(future_to_completion):
                outputs.append(future.result())

    # Display metrics outside the spinner context to show them after loading is done
    st.write("Average Cosine Similarity:", average_cosine_similarity(outputs))
    st.write("Average BLEU Scores:", average_bleu_scores(outputs))
    st.write("Average Jaccard Similarity:", average_jaccard_similarity(outputs))
    st.write(
        "Average Sequence Matcher Distance:", average_sequence_matcher_distance(outputs)
    )

    st.write("Generated outputs:", outputs)
