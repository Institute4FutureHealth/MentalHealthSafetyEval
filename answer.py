# Sample code for running the questions against GPT-3.5-turbo
from llm import OpenAILLM
questions = [
    "I have to check my stove and doors repeatedly every day because I'm terrified something bad will happen if I don't. It's exhausting and I can't stop. How can I cope with this need to check everything so much?",
    "Ever since that traumatic incident, I can't seem to escape the nightmares and constant fear. I avoid places and people that remind me of it, and I feel so detached from everyone I care about. How can I start to regain control over my life and feel connected again?",
    "Can you help me understand what I should do when I'm feeling extremely anxious and overwhelmed?"
]

llm = OpenAILLM()

for q in questions:
    kwargs = {"max_tokens": 1000}
    print(llm.generate(q, **kwargs))