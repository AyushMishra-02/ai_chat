# import openai

# openai.api_key = "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"

# def chat_with_gpt(prompt):
#     response = openai.ChatCompletion.create(
#         model = "gpt-3.5-turbo",
#         messages = [{"role":"user","content":prompt}]
#     )

#     return response.choices[0].message.content.strip()

# if __name__ ==  "__main__":
#     while True:
#         user_input =  input("You: ")    
#         if user_input.lower() in ["quit","exit","bye"]:
#             break

#         response = chat_with_gpt(user_input)
#         print("Chatbot: ",response)

import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # More secure way of storing the API key

def chat_with_gpt(prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=150):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except openai.error.OpenAIError as e:
        return f"Error: {str(e)}"

def main():
    print("Welcome to the AI Chatbot! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye! Have a great day!")
            break

        response = chat_with_gpt(user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()
