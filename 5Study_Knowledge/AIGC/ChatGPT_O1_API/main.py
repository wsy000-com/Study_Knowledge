from config import config
from model.chatgpt_o1 import ChatGPT_o1

if __name__=="__main__":
    chatgpt_o1_obj = ChatGPT_o1(config)
    prompt = "介绍一下ChatGPT"
    chatgpt_o1_obj.print_response_stream(prompt)