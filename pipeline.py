import argparse
import json
import os
import re


RETRIEVAL_MODES = ("direct", "gar", "rgar", "iterative_rgar")
DEFAULT_RETRIEVAL_MODE = "direct"


class QADataset:

    def __init__(self, data, dir="."):
        self.data = data.lower().split("_")[0]
        with open(os.path.join(dir, "benchmark.json"), "r", encoding="utf-8") as f:
            benchmark = json.load(f)
        if self.data not in benchmark:
            raise KeyError("{:s} not supported".format(data))
        self.dataset = benchmark[self.data]
        self.index = sorted(self.dataset.keys())

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        if type(key) == int:
            return self.dataset[self.index[key]]
        elif type(key) == slice:
            return [self.__getitem__(i) for i in range(self.__len__())[key]]
        else:
            raise KeyError("Key type not supported.")


def extract_answer(content):
    
    match = re.findall(r'(?:answer|Answer).*?([A-Z])', content)
    if match:
        return match[-1]
    
    match_last = re.search(r'([A-Z])(?=[^A-Z]*$)', content)
    if match_last:
        return match_last.group(1)
    return None

def main(args):
    import torch
    from src.RGAR import RGAR

    dataset = QADataset(args.dataset_name, dir=args.dataset_dir)

    rgar = RGAR(
        llm_name=args.llm_name, 
        rag=args.rag, 
        follow_up=args.follow_up,
        retriever_name=args.retriever_name, 
        corpus_name=args.corpus_name, 
        device=args.device,
        cot=args.cot,
        retrieval_mode=args.retrieval_mode,
        iterative_rounds=args.iterative_rounds,
        follow_up_rounds=args.follow_up_rounds,
        follow_up_queries=args.follow_up_queries,
    )

    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} saved results.")
    else:
        results = []

    correct_count = sum(1 for r in results if r["is_correct"])  
    start_idx = len(results) 

    for idx, data in enumerate(dataset[start_idx:], start=start_idx):
        question = data["question"]
        options = data["options"]
        correct_answer = data["answer"]

        answer_json, *_ = rgar.answer(question=question, options=options, k=args.top_k)
        # answer_json, snippets, scores = medrag.answer(question=question, options=options, k=args.top_k)
        print(answer_json) 

        predicted_answer = extract_answer(answer_json)

        if predicted_answer is None:
            print(f"Warning: Could not extract answer for Question {idx + 1}")
            predicted_answer = "N/A" 

        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct_count += 1

        print(f"Question {idx + 1}/{len(dataset)}:")
        print(f"  Correct Answer: {correct_answer}")
        print(f"  Predicted Answer: {predicted_answer}")
        print(f"  {'Correct!' if is_correct else 'Incorrect.'}")

        results.append({
            "question": question,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "raw_output": answer_json, 
            "is_correct": is_correct
        })

        if (idx + 1) % 10 == 0:
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            print(f"Progress saved at {idx + 1} questions.")

        torch.cuda.empty_cache()

    accuracy = correct_count / len(dataset) if len(dataset) > 0 else 0.0
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    torch.cuda.empty_cache()
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Final results saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RGAR on a QA dataset.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to load.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory where the dataset is stored.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the results.")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Name of the LLM model.")
    parser.add_argument("--rag", action="store_true", help="Enable RAG (retrieval-augmented generation).")
    parser.add_argument("--retriever_name", type=str, default="MedCPT", help="Name of the retriever model.")
    parser.add_argument("--corpus_name", type=str, default="Textbooks", help="Name of the corpus to use.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on.")
    parser.add_argument("--cot", action="store_true", help="Enable Chain-of-Thought reasoning.")
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        choices=RETRIEVAL_MODES,
        default=DEFAULT_RETRIEVAL_MODE,
        help="Retrieval mode: direct | gar | rgar | iterative_rgar.",
    )
    parser.add_argument("--iterative_rounds", type=int, default=2, help="Number of rounds for iterative_rgar mode.")
    parser.add_argument("--follow_up", action="store_true", help="Enable i-medrag follow-up ask/answer loop.")
    parser.add_argument("--follow_up_rounds", type=int, default=2, help="Number of follow-up rounds in i-medrag mode.")
    parser.add_argument("--follow_up_queries", type=int, default=3, help="Queries generated per follow-up round in i-medrag mode.")
    parser.add_argument("--top_k", type=int, default=32, help="Number of top results to retrieve.")
    args = parser.parse_args()
    main(args)
