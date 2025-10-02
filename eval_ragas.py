from __future__ import annotations
import sys
import pandas as pd
from datasets import Dataset
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from ragas import evaluate

# Expected CSV columns: question, ground_truth, contexts (| separated)
# Example:
# question,ground_truth,contexts
# "What is measles?","Measles is a highly contagious viral disease...","WHO measles factsheet|CDC measles overview"

def run_eval(csv_path: str):
    df = pd.read_csv(csv_path)
    df["contexts"] = df["contexts"].apply(
        lambda x: [c.strip() for c in str(x).split("|") if c.strip()]
    )
    dataset = Dataset.from_pandas(df[["question", "contexts", "ground_truth"]])
    result = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
    )
    print(result)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_ragas.py evalset.csv")
        raise SystemExit(1)
    run_eval(sys.argv[1])
