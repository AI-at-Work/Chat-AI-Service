from rag.extractor import get_text_from_pdfs
import prompts.rag_prompt
from rag.vector_store import raptor_get_docs, check_index_exists, create_colbert_index, get_colbert_index, add_to_index
from ragatouille import RAGPretrainedModel
from sentence_transformers import SentenceTransformer


class rag:
    def __init__(self, llm_service, index_dir):
        self.embedding = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.llm_service = llm_service
        self.INDEX_DIR = index_dir

    def find_answers_in_pdf(self, question: str, colbert: RAGPretrainedModel, provider, model):
        # Query Pinecone
        results = colbert.search(query=question, k=10)

        context = [res['content'] for res in results]

        prompt = prompts.rag_prompt.get_question_context_prompt(context, question)

        # Generate an answer using the LLM service
        return self.llm_service.generate(provider, model, prompt, "")

    def perform_rag(self, session_id, pdf_file_path, query, provider, model):
        # get the data from pdfs
        pdf_data_list = get_text_from_pdfs(pdf_file_path)

        # create a bunch of documents
        # 4 min for 21 pages
        docs, input_tokens, output_tokens = raptor_get_docs(pdf_data_list, self.llm_service, provider, model,
                                                            self.embedding, chunk_size=512,
                                                            chunk_overlap=0)

        if not check_index_exists(self.INDEX_DIR, session_id):

            print(docs)

            # create the colbert index
            colbert = create_colbert_index(self.INDEX_DIR, session_id, docs)
        else:
            # get the existing index
            colbert = get_colbert_index(self.INDEX_DIR, session_id)

            # add document to existing index
            add_to_index(colbert, docs)

        answer = self.find_answers_in_pdf(query, colbert, provider, model)
        answer["output_tokens"] += output_tokens
        answer["input_tokens"] += input_tokens

        # now get the answer
        return answer
