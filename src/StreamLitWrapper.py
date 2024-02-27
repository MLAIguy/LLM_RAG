import streamlit as st
from langchain_community.vectorstores.faiss import FAISS
from src.app import data_ingestion, get_vector_store, get_embedding_model, get_llama2_llm, get_response_llm
from src.utils import get_boto3_client


class StreamLitWrapper:
    def __init__(self, page_name:str, header: str, question_box_title: str,
                 side_bar_title: str, side_bar_button: str, QA_model_buttons: list[str]):
        self.page_name = page_name
        self.header = header
        self.question_box_title = question_box_title
        self.side_bar_title = side_bar_title
        self.side_bar_button = side_bar_button
        self.QA_model_buttons = QA_model_buttons

    def run(self, data_config: dict, model_config: dict):
        st.set_page_config(self.page_name)
        st.header(self.header)

        bedrock_client = get_boto3_client(service_name='bedrock-runtime')
        bedrock_embedding = get_embedding_model(model_config['embedding_model_id'], client=bedrock_client)

        vector_store_save_path = model_config['vector_store_save_path']
        with st.sidebar:
            st.title(self.side_bar_title)

            if st.button(self.side_bar_button):
                with st.spinner("Processing ..."):
                    docs = data_ingestion(**data_config)
                    get_vector_store(docs, bedrock_embedding, vector_store_save_path)
                    st.success("Done")

        user_question = st.text_input(self.question_box_title)
        for inference_model_config in model_config['inference_config']:
            if st.button(f"{inference_model_config['model_name']} Output"):
                with st.spinner("Processing ..."):
                    faiss_index = FAISS.load_local(vector_store_save_path, bedrock_embedding)
                    llm = get_llama2_llm(
                        model_id=inference_model_config['model_id'],
                        client=bedrock_client,
                        model_kwargs=inference_model_config['model_kwargs']
                    )

                    response = get_response_llm(llm, vectorstore_faiss=faiss_index, query=user_question)
                    st.write(response)
                    st.success("Done")