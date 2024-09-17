from sentence_transformers import SentenceTransformer
import pandas as pd
from numpy.linalg import norm
import numpy as np

data = pd.read_csv("Safety_Benchmark_Mental Health -Sheet1new.csv")
data_answer = pd.read_csv("Safety_Benchmark_Mental Health-ChatGPT 3.5.csv")
# Download the punkt tokenizer if not already downloaded
def cosine_similarity(a, b):
    return str(np.dot(a, b)/(norm(a)*norm(b)))

responses = {
      "all-mpnet-base-v2": [],
      "all-MiniLM-L6-v2": []
}

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
model2 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

for index, row in data.iterrows():
    print(f"scoring question: {index} {row['Question']}")

    m = cosine_similarity(model.encode(row["Ideal Response"]), model.encode(data_answer.loc[index][f"ChatGPT Answers"]))
    m1 = cosine_similarity(model2.encode(row["Ideal Response"]), model2.encode(data_answer.loc[index][f"ChatGPT Answers"]))

    try:
        responses["all-mpnet-base-v2"].append(m)
        f = open(f"./result_eval_5_all-mpnet-base-v2.txt", "a")
        f.write(f"\n------\nindex:{index}\n" + "".join(responses["all-mpnet-base-v2"][-1]) + "\n------\n")
        f.close()

    except Exception as e:
        print(f"error {e}")
        f = open(f"./result_eval_5_all-mpnet-base-v2.txt", "a")
        f.write(f"\n------\nindex:{index}\nError\n------\n")
        f.close()

    try:
        responses["all-MiniLM-L6-v2"].append(m1)
        f = open(f"./result_eval_5_all-MiniLM-L6-v2.txt", "a")
        f.write(f"\n------\nindex:{index}\n" + "".join(responses["all-MiniLM-L6-v2"][-1]) + "\n------\n")
        f.close()
    except Exception as e:
        print(f"error {e}")
        f = open(f"./result_eval_5_all-MiniLM-L6-v2.txt", "a")
        f.write(f"\n------\nindex:{index}\nError\n------\n")
        f.close()

df = pd.DataFrame(responses["all-mpnet-base-v2"], columns=[f"all-mpnet-base-v2 Answers"])
df.to_csv(f'Safety_Benchmark_Mental Health-all-mpnet-base-v2 eval_5.csv')

df = pd.DataFrame(responses["all-MiniLM-L6-v2"], columns=[f"all-MiniLM-L6-v2 Answers"])
df.to_csv(f'Safety_Benchmark_Mental Health-all-MiniLM-L6-v2 eval_5.csv')