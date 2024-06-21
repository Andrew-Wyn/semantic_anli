from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import random

models = {}
tokenizer = {}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def initialize_models():
    global models
    global tokenizers

    model_name_base = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    model_name_large = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    model_name_large_2 = "Joelzhang/deberta-v3-large-snli_mnli_fever_anli_R1_R2_R3-nli"

    tokenizer_base = AutoTokenizer.from_pretrained(model_name_base)
    model_base = AutoModelForSequenceClassification.from_pretrained(model_name_base)

    tokenizer_large = AutoTokenizer.from_pretrained(model_name_large)
    model_large = AutoModelForSequenceClassification.from_pretrained(model_name_large)

    tokenizer_large_2 = AutoTokenizer.from_pretrained(model_name_large_2)
    model_large_2 = AutoModelForSequenceClassification.from_pretrained(
        model_name_large_2
    )

    models = {
        "base": model_base.to(device),
        "large": model_large.to(device),
        "large2": model_large_2.to(device),
    }
    tokenizers = {
        "base": tokenizer_base,
        "large": tokenizer_large,
        "large2": tokenizer_large_2,
    }


if __name__ == "__main__":
    initialize_models()

    chosen_model = random.choice(list(models.keys()))
    print(f"Chosen Model: {chosen_model}")

    premise = input("Insert premise:")
    hypothesis = input("Insert hypothesis:")
    label = input("Insert gold label (entailment, neutral, contraddiction):")

    while True:
        model_input = tokenizers[chosen_model](
            premise, hypothesis, truncation=True, return_tensors="pt"
        )
        output = models[chosen_model](
            model_input["input_ids"].to(device)
        )  # device = "cuda:0" or "cpu"
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        label_names = ["entailment", "neutral", "contradiction"]
        prediction = {
            name: round(float(pred) * 100, 1)
            for pred, name in zip(prediction, label_names)
        }

        print(f"Gold label: {label}")
        print(f"Predicted label: {max(prediction, key=prediction.get)}")
        print(f"Predicted Percentage: {prediction}")

        c = input("continue? (Y|n)")

        if c == "Y":
            break

        hypothesis = input("Insert hypothesis:")
