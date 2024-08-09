from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from ragatouille import RAGPretrainedModel
from pysbd import Segmenter
from typing import List, Tuple, Any

from rag.raptor import recursive_embed_cluster_summarize
import os


def add_to_index(colbert: RAGPretrainedModel, docs: List[str]):
    colbert.add_to_index(docs)


def check_index_exists(INDEX_DIR, idx_name):
    try:
        # List all indexes
        existing_indexes = os.listdir(INDEX_DIR)

        # Check if the index name is in the list of existing indexes
        if idx_name in existing_indexes:
            print(f"The index '{idx_name}' already exists.")
            return True
        print(f"The index '{idx_name}' does not exist.")
        return False
    except Exception as e:
        print(f"An error occurred while checking the index: {e}")
        return False


def create_colbert_index(path, index_name: str, docs: List[str]) -> RAGPretrainedModel:
    try:
        os.mkdir(os.path.join(path, index_name))
        colbert = RAGPretrainedModel.from_pretrained(
            "colbert-ir/colbertv2.0",
            index_root=os.path.join(path, index_name)
        )

        colbert.index(
            collection=docs,
            split_documents=False,
            index_name=index_name,
            # use_faiss=True,
        )

        return colbert
    except Exception as e:
        print(f"Error creating index: {e}")
        raise


def get_colbert_index(path, index_name: str) -> RAGPretrainedModel:
    try:
        return RAGPretrainedModel.from_index(os.path.join(path, index_name, 'colbert', 'indexes', index_name))
    except Exception as e:
        print(f"Error creating index: {e}")
        raise


def raptor_get_docs(
        docs: List[str],
        llm_service,
        provider, model,
        embd: SentenceTransformer,
        chunk_size=2000,
        chunk_overlap=50,
) -> tuple[list[str], int, int]:
    try:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        texts_split = []
        for item in docs:
            texts_split.extend(text_splitter.split_text(item))

        print("Length of texts splits: ", len(texts_split))

        # generate the recursive summaries
        results, input_tokens, output_tokens = recursive_embed_cluster_summarize(texts_split, llm_service, provider, model, embd)

        print(results)

        all_texts = texts_split.copy()

        # Iterate through the results to extract summaries from each level and add them to all_texts
        for level in sorted(results.keys()):
            print(results[level][1]["summaries"])

            # get the summary text
            summaries = results[level][1]["summaries"].tolist()

            # Extend all_texts with the summaries from the current level
            all_texts.extend(summaries)

        return all_texts, input_tokens, output_tokens

    except Exception as e:
        print(f"Error creating index: {e}")
        raise


def segment_text_into_sentences(text: str) -> List[str]:
    """
    Segments the input text into sentences using pySBD.

    Args:
    text (str): The input text to be segmented.

    Returns:
    List[str]: A list of sentences extracted from the input text.
    """
    # Initialize pySBD segmenter
    segmenter = Segmenter(language="en", clean=False, char_span=True)

    # Segment the text into sentences
    sentence_objects = segmenter.segment(text)

    # Extract the sentence strings from the sentence objects
    sentences = [sent.sent for sent in sentence_objects]

    return sentences
