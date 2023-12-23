import streamlit as st
import textwrap
import pandas as pd
import time
from sentence_transformers import SentenceTransformer, util
from annoy import AnnoyIndex

footer = """
<p style='text-align: center; color: gray;'>Made with inspiration by Om A. Soni</p>
"""

shlok_keys = ['Title', 'Chapter', 'Verse', 'Enlgish Translation']

@st.cache_resource
def load_data():
   hn_filepath = 'Gita.xlsx'
   return pd.read_excel(hn_filepath)

@st.cache_resource
def load_hn_model():
  return SentenceTransformer('all-mpnet-base-v2')

hn_model = load_hn_model()

@st.cache_resource
def build_embeddings(hn_data):
   return [hn_model.encode(hn_data['Enlgish Translation'][i], convert_to_tensor=True).numpy() for i in range(len(hn_data))]

@st.cache_resource
def build_annoy_index(shloka_embeddings):
   embedding_size = len(shloka_embeddings[0])
   annoy_index = AnnoyIndex(embedding_size, metric='angular')
   for i, embedding in enumerate(shloka_embeddings):
       annoy_index.add_item(i, embedding)
   annoy_index.build(18) # 18 trees for faster search
   return annoy_index

# st.write("shree ganeshay namah")

hn_data = load_data()
shloka_embeddings = build_embeddings(hn_data)
annoy_index = build_annoy_index(shloka_embeddings)

st.title("TopShlok Bhagavad Gita Assistant")

st.markdown(footer, unsafe_allow_html=True)

   
query = st.text_input("Ask any question related to the Bhagavad Gita: ")

if st.button('Ask'):

    query_embedding = hn_model.encode(query, convert_to_tensor=True).numpy()

    # Use Annoy Index for efficient similarity search
    similar_indices = annoy_index.get_nns_by_vector(query_embedding, 18)

    # Process and display similar Shlokas
    similarities = []
    for curr_index in similar_indices:
        similarity = util.cos_sim(query_embedding, shloka_embeddings[curr_index])
        curr_shlok_details = {key: hn_data[key][curr_index] for key in shlok_keys}
        similarities.append({"shlok_details": curr_shlok_details, "similarity": similarity})

    # Get the most similar Shloka
    top_result = sorted(similarities, key=lambda x: x["similarity"], reverse=True)[0]
    top_shlok_details = top_result["shlok_details"]
    adhyay_number = top_shlok_details['Chapter'].split(" ")[1]
    shlok_number = top_shlok_details['Verse'].split(" ")[1].split(".")[1]

    st.write("------------------------------")
    st.write(f"{top_shlok_details['Chapter']} , Shloka : {shlok_number}")

    meaning_text = top_shlok_details['Enlgish Translation']
    max_line_length = 80  # Adjust as needed

    wrapped_text = textwrap.fill(meaning_text, width=max_line_length)

    placeholder = st.empty()

    prev_text=''
    for char in wrapped_text:
      prev_text=prev_text+char
      placeholder.text(prev_text)
      time.sleep(0.02)  # Adjust the sleep duration as needed
    st.write("\n------------------------------")


    # Prompt for continuation
    # next_input = input("Type 'jsk' to stop or press Enter to continue: ")
    # if next_input.lower() == "jsk":
    #     st.write("|| Jai Shree Krishna ||")  # English farewell
    #     break

