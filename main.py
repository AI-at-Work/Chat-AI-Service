import grpc
from concurrent import futures
import logging
import redis
from google.protobuf.timestamp_pb2 import Timestamp
from prompts.chat_conversation_summary_prompt import get_chat_conversation_summary_prompt
from prompts.rag_prompt import get_rag_prompt
import sys
import os
import tiktoken

from rag import rag
import modules.cost.openai_cost
import prompts.chat_conversation_summary_prompt

# Add the server directory to the Python path
server_dir = os.path.dirname(os.path.abspath(__file__)) + '/pb'
sys.path.insert(0, server_dir)

# Import the generated protobuf code
from pb import ai_service_pb2
from pb import ai_service_pb2_grpc

from llm.llm import LLM
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
AI_SERVER_PORT = os.getenv("AI_SERVER_PORT")

# Set token limits
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS"))

# path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, 'docs_index')


class AIService(ai_service_pb2_grpc.AIServiceServicer):
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.llm_service = LLM(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.rag = rag.rag(self.llm_service, INDEX_DIR)

    def get_token_count(self, text, provider, model):
        if provider == "openai":
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        else:
            # Because we support ollama, and it is running locally; we can estimate cost based upon tokens later
            return 0

    def get_token_cost(self, input_tokens, output_tokens, model_name):
        cost = modules.cost.openai_cost.estimate_llm_api_cost(model_name, num_tokens_input=input_tokens,
                                                              num_tokens_output=output_tokens)
        return cost

    def perform_rag(self, session_id, pdf_file, chat_message, provider, model):
        return self.rag.perform_rag(session_id=session_id, pdf_file_path=pdf_file, query=chat_message,
                                    provider=provider, model=model)

    def Process(self, request, context):
        input_tokens = 0
        output_tokens = 0

        logging.info(
            f"Received chat from user {request.user_id}, session {request.session_id}, file: {request.file_name}, "
            f"chat_history:{request.chat_history}, chat_summary: {request.chat_summary}, model: {request.model_name}, "
            f"model_provider: {request.model_provider}\n\n")

        if len(request.file_name) == 0:
            context_prompt = get_chat_conversation_summary_prompt(request.chat_summary,
                                                                  request.chat_history,
                                                                  request.chat_message)
        else:
            doc = self.perform_rag(request.session_id, request.file_name, request.chat_message, request.model_provider,
                                   request.model_name)
            context_prompt = get_rag_prompt(request.chat_summary,
                                            request.chat_history,
                                            request.chat_message,
                                            doc["text"])

            # these tokens are of RAG Raptor Summarization
            input_tokens += doc["input_tokens"]
            output_tokens += doc["output_tokens"]

        # Check input token count
        context_prompt_token = self.get_token_count(context_prompt, request.model_provider, request.model_name)
        tokens_cost = self.get_token_cost(input_tokens + context_prompt_token, output_tokens, request.model_name)
        if input_tokens > MAX_INPUT_TOKENS:
            context.set_details("Input token limit exceeded")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return ai_service_pb2.Response()
        elif tokens_cost > request.balance:
            context.set_details("Insufficient funds")
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            return ai_service_pb2.Response()

        logging.info("Now about to start generating response.")
        try:
            response = self.llm_service.generate(
                model=request.model_name,
                provider=request.model_provider,
                system_prompt=request.session_prompt,
                user_prompt=context_prompt,
                max_tokens=MAX_OUTPUT_TOKENS
            )
            ai_response = response["text"]

            # Now cost of final answer generation
            input_tokens += response["input_tokens"]
            output_tokens += response["output_tokens"]

            session_cost = self.get_token_cost(input_tokens, output_tokens, request.model_name)
        except Exception as e:
            context.set_details(f'Failed to generate response from API: {e}')
            context.set_code(grpc.StatusCode.INTERNAL)
            return ai_service_pb2.Response()

        print("Total session cost: ", session_cost)
        response = ai_service_pb2.Response(
            response_text=ai_response,
            timestamp=Timestamp(),
            cost=session_cost
        )
        response.timestamp.GetCurrentTime()

        return response


def serve():
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ai_service_pb2_grpc.add_AIServiceServicer_to_server(AIService(redis_client), server)
    server.add_insecure_port(f'[::]:{AI_SERVER_PORT}')
    server.start()
    logging.info(f"Server started on port {AI_SERVER_PORT}")
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
