def get_rag_prompt(chat_summary, chat_history, chat_message, doc_data):
    context_prompt = f"""The following is a friendly conversation between a user and an AI Assistant. The AI 
            Assistant is talkative and provides lots of specific details from its context and do what user ask for. If 
            the AI Assistant does not know the answer to a question, it truthfully says it does not know.

            the summary is given in the form of text and past conversation is given in the form of json. Use the provided 
            Summary of conversation and past conversation to answer the query of the user and follow the instructions. The 
            relevant chunks of data to resolve users query is given use them if found relevant to User query (You do not
             need to use these pieces of information if not relevant)

            Relevant Document Parts:
            {doc_data}

            Summary of conversation:
            {chat_summary}

            past conversation:
            {chat_history}

            User: {chat_message}
            Assistant:"""
    return context_prompt


def get_question_context_prompt(context, question):
    prompt = f"""
    Context: {context}

    Question: {question}

    Please provide a concise and accurate answer to the question based on the given context. 
    If the context doesn't contain enough information to answer the question confidently, 
    please state that the information is not available in the given context.

    Answer:
    """
    return prompt
