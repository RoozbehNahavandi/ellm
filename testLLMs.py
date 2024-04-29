# import os
# import json
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# prompt_templates = {
#     'llama': """[INST] <<SYS>>
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
# <</SYS>>
# {prompt}[/INST]
# """,
#     'yi': """<|user|>
# {prompt}
# <|assistant|>
# """,
#     'judgeLM_7b': '{prompt}',
#     'vicuna-13b-gptq': """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:

# """,
#     'vicuna-13b': """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:

# """,
#     'mistral-7b-gptq': '''<s>[INST] {prompt} [/INST]
# ''',
#     'mistral-7b': '''<s>[INST] {prompt} [/INST]
# ''',
#     'alpaca-30b': "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n"
# }

# argss = {'mistral-7b': {
#         'do_sample': False,
#         'max_new_tokens': 512,
#     }}

# print('we are currently here.')

# model_path = '/users/PAS2138/roozbehn99/ellm/LMs/Mistral-7B-Instruct-v0.2/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/41b61a33a2483885c981aa79e0df6b32407ed873'                                                                                      
# model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             device_map='auto',
#             trust_remote_code=False,
#         )
# print('huray we loaded the model')
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# prompt = """
# You are a player in a Crafter game. I am going to tell you the history of what the player (what he has seen, done, etc,) and I want you 
# """

# # preproc_prompt = preprocess_prompt(prompt)
# preproc_prompt = prompt_templates['mistral-7b'].format(prompt=prompt)
# print('preprocessed =', preproc_prompt)

# inputs = tokenizer(preproc_prompt, return_tensors='pt').input_ids.cuda()
# output = model.generate(inputs=inputs, **argss['mistral-7b'])
# print('and the response is:')
# print(tokenizer.decode(output[0]))




#         # self.history_config_fp = os.path.join(os.path.dirname(__file__), 'history_lm_config.json')
#         # self.history_config = json.load(open(self.history_config_fp))
#         # self.history_model_name = 'mistral-7b'
#         # self.history_model_path = self.history_config[self.history_model_name]
#         # print('anananananana ', self.history_model_path)
#         # self.history_model = AutoModelForCausalLM.from_pretrained(
#         #     self.history_model_path,
#         #     device_map='auto',
#         #     trust_remote_code=False,
#         # )
#         # self.history_tokenizer = AutoTokenizer.from_pretrained(self.history_model_path)




import os
from anthropic import Anthropic
from anthropic.types import MessageParam

client = Anthropic(api_key = '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')


# def estimate_tokens(prompt):
#     count = client.count_tokens(prompt)
#     return count


# def submit_prompt(prompt, system_prompt):
#     with client.messages.stream(
#         model = 'claude-2.1', 
#         system = system_prompt,
#         max_tokens = 1024,
#         messages = [
#             {'role': 'user', 'content': prompt}
#         ]
#     ) as stream:
#         try:
#             for text in stream.text_stream:
#                 print(text, "", flush = True)
#         except Exception as e:
#             print(e)
#             raise e


# if __name__ == '__main__':

#     prompt = 'Introduce yourself and tell me a joke.'
#     system_prompt = ""
#     print(f"Tokens for this request: {estimate_tokens(prompt )}")
#     submit_prompt(prompt, system_prompt= system_prompt)


message = 'You see plant, tree, and skeleton. You are targeting skeleton. You see water, grass, cow, and diamond. You are targeting grass. You have in your inventory plant. You see cow, grass, and tree. You are targeting grass. You havent done anything until now (your history is empty)'
system_prompt = """
We have an agent who is playing in the crafter game.
I am going to tell you the history of what the player has done, seen, and achieved so far alongside the current observation 
and it's current achievemenets. I want you to give me a new history based on the previous history and new information.
Don't add any other information to it beside the achievements and what has already been done.
"""

message = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1024,
    system = system_prompt,
    messages=[
        {'role': 'user', 'content': message}
    ]
)
print(message.content[0].text)


