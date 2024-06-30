import grpc
from concurrent import futures
import logging
import openai
import redis
from google.protobuf.timestamp_pb2 import Timestamp
import sys
import os

# Add the server directory to the Python path
server_dir = os.path.dirname(os.path.abspath(__file__)) + '/pb'
sys.path.insert(0, server_dir)

# Import the generated protobuf code
from pb import ai_service_pb2
from pb import ai_service_pb2_grpc

# Connect to Redis
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
openai.api_key = os.getenv("OPENAI_API_KEY")


class AIService(ai_service_pb2_grpc.AIServiceServicer):
    def __init__(self, redis_client):
        self.redis_client = redis_client

    def Process(self, request, context):
        # Log the incoming request
        logging.info(f"Received chat from user {request.user_id}, session {request.session_id}")

        # Prepare the context for OpenAI API
        context_prompt = f"Previous relevant conversation summary:\n {request.chat_summary}\n\n"
        context_prompt += f"Current user message:\n {request.chat_message}\n\n"
        context_prompt += f"Note: You have to answer the user query based upon the summary provided.\n"

        print("Context Prompt: ", context_prompt)

        # Generate response using OpenAI API
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # or any other suitable model
                messages=[
                    {"role": "system", "content": request.session_prompt},
                    {"role": "user", "content": context_prompt}
                ]
            )
            ai_response = response.choices[0].message['content'].strip()
        except Exception as e:
            context.set_details(f'Failed to generate response from OpenAI API: {e}')
            context.set_code(grpc.StatusCode.INTERNAL)
            return ai_service_pb2.Response()

        # Create and return the response
        response = ai_service_pb2.Response(
            response_text=ai_response,
            timestamp=Timestamp()
        )
        response.timestamp.GetCurrentTime()

        return response


def serve():
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD
    )

    # user_id = "6c2f2f31-1cb1-4310-b89d-b370c8718cf6"
    # session_id = "2fad58be-8f52-482c-a698-e0e5574a3aee"
    # chat_message = "shitij"

    # try:
    #     embeddings = generate_embeddings(chat_message)
    #     relevant_chats = search_redis(user_id=user_id, session_id=session_id, embedded_query=embeddings,
    #                                   redis_client=redis_client)
    #     print(relevant_chats)
    # except Exception as e:
    #     print(e)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ai_service_pb2_grpc.add_AIServiceServicer_to_server(AIService(redis_client), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logging.info("Server started on port 50051")
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
