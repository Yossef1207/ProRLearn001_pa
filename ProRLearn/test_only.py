import os
import math
import argparse
import torch
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate, ManualVerbalizer
from openprompt import PromptForClassification, PromptDataLoader
import torch.nn.functional as F


def build_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    ds = Dataset.from_pandas(df)
    examples = []
    for idx, data in enumerate(ds):
        text = data.get("processed_func")
        if text is None or (isinstance(text, float) and math.isnan(text)):
            continue
        try:
            orig_index = int(data.get("index"))
        except Exception:
            orig_index = int(idx)
        examples.append(
            InputExample(
                guid=orig_index,
                text_a=str(text),
                label=int(data["target"]),
            )
        )
    return examples


def test(prompt_model, test_dataloader, use_cuda: bool):
    allpreds = []
    alllabels = []
    all_indices = []
    with torch.no_grad():
        for inputs in test_dataloader:
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs["label"]
            preds = torch.argmax(logits, dim=-1)
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(preds.cpu().tolist())
            batch_guids = inputs.get("index", None)
            if batch_guids is not None:
                if isinstance(batch_guids, torch.Tensor):
                    all_indices.extend(batch_guids.cpu().tolist())
                else:
                    all_indices.extend([int(x) for x in batch_guids])
            else:
                start = len(all_indices)
                all_indices.extend(range(start, start + preds.size(0)))

    acc = accuracy_score(alllabels, allpreds)
    f1 = f1_score(alllabels, allpreds)
    precision = precision_score(alllabels, allpreds, zero_division=0)
    recall = recall_score(alllabels, allpreds)

    print(f"acc: {acc}  recall: {recall}  precision: {precision}  f1: {f1}")

    cm = confusion_matrix(alllabels, allpreds, labels=[0, 1])
    print("Confusion matrix (rows: true [0,1], cols: pred [0,1]):")
    print(cm)

    tp_indices, tn_indices, fp_indices, fn_indices = [], [], [], []
    for csv_idx, y_true, y_pred in zip(all_indices, alllabels, allpreds):
        if y_true == 1 and y_pred == 1:
            tp_indices.append(csv_idx)
        elif y_true == 0 and y_pred == 0:
            tn_indices.append(csv_idx)
        elif y_true == 0 and y_pred == 1:
            fp_indices.append(csv_idx)
        elif y_true == 1 and y_pred == 0:
            fn_indices.append(csv_idx)

    print("True Positive indices (CSV index column):", tp_indices)
    print("False Positive indices (CSV index column):", fp_indices)
    print("True Negative indices (CSV index column):", tn_indices)
    print("False Negative indices (CSV index column):", fn_indices)


def main():
    parser = argparse.ArgumentParser(description="Test-only runner for ProRLearn prompt model.")
    parser.add_argument("--dataset", required=True, help="Dataset name under ../data, e.g. reposvul_dataset")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Pfad zum gespeicherten Modell (default: ../models/reposvul_dataset_best_model.pt)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for testing")
    parser.add_argument("--max-seq-l", type=int, default=256, help="Max sequence length")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "..", "data", args.dataset)
    test_csv = os.path.join(dataset_dir, "test.csv")

    test_examples = build_dataset(test_csv)

    plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "../CodeBERT")
    template_text = 'This code {"placeholder":"text_a"} is a vulnerability. {"mask"}.'
    mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text=template_text)
    verbalizer = ManualVerbalizer(tokenizer, num_classes=2, label_words=[["false", "no"], ["true", "yes"]])
    prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=verbalizer, freeze_plm=False)

    if use_cuda:
        prompt_model = prompt_model.cuda()

    model_path = args.model_path
    if model_path is None:
        model_path = os.path.join(script_dir, "..", "models", f"{args.dataset}_best_model.pt")
    state_dict = torch.load(model_path, map_location="cpu")
    prompt_model.load_state_dict(state_dict)

    test_loader = PromptDataLoader(
        dataset=test_examples,
        template=mytemplate,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=args.max_seq_l,
        batch_size=args.batch_size,
        shuffle=False,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="head",
    )

    test(prompt_model, test_loader, use_cuda)


if __name__ == "__main__":
    main()
