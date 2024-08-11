def get_chat_conversation_summary_prompt(chat_summary, chat_history, chat_message):
    context_prompt = f"""The following is a friendly conversation between a user and an AI Assistant. The AI 
            Assistant is talkative and provides lots of specific details from its context and do what user ask for. If 
            the AI Assistant does not know the answer to a question, it truthfully says it does not know.

            the summary is given in the form of text and past conversation is given in the form of json. Use the provided 
            Summary of conversation and past conversation to answer the query of the user and follow the instructions. (
            You do not need to use these pieces of information if not relevant)

            Summary of conversation:
            {chat_summary}

            past conversation:
            {chat_history}

            User: {chat_message}
            Assistant:"""
    return context_prompt
