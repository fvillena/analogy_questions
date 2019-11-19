import json
import os
import pandas as pd
import matplotlib.pyplot as plt

performances = {}
for filename in os.listdir():
    if filename.endswith("result.json"):
         with open(filename) as json_file:
            data = json.load(json_file)
            embedding_name = data["config"]["model"]
            n_questions = data["n_questions"]
            n_correct_questions = data["n_correct_questions"]
            accuracy = data["accuracy"]
            performances[embedding_name] = {
                "n_questions" : n_questions,
                "n_correct_questions" : n_correct_questions,
                "accuracy":accuracy
            }
data = []
for embedding,performance in performances.items():
    for metric,submetric in performance.items():
        for submetric_name,val in submetric.items():
            data.append((embedding[:-4],metric,submetric_name,val))
data = pd.DataFrame(data,columns=["embedding","metric","group","val"])

data[data.metric == "accuracy"].pivot("group", "embedding", "val").plot(kind='bar')
plt.ylabel("Accuracy")
plt.xlabel("Group")
plt.tight_layout()
plt.savefig("figure.pdf")