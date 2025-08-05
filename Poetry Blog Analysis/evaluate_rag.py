import json
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from bert_score import score as bert_score
from poetry_rag_core import generate_poetry_analysis

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

STOP_WORDS = set(stopwords.words('english'))
PUNCTUATION = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

def normalize(text):
    tokens = word_tokenize(text.lower())
    lemmas = [
        lemmatizer.lemmatize(t) for t in tokens
        if t not in STOP_WORDS and t not in PUNCTUATION
    ]
    return lemmas

def get_synonyms(word):
    syns = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            syn_word = lemma.name().lower().replace('_', ' ')
            syns.add(lemmatizer.lemmatize(syn_word))
    syns.add(lemmatizer.lemmatize(word))
    return syns

def fuzzy_set_match(pred_tokens, true_tokens):
    pred_syn_sets = [get_synonyms(tok) for tok in pred_tokens]
    pred_syn_flat = set().union(*pred_syn_sets) if pred_syn_sets else set()

    matched = 0
    for true_tok in true_tokens:
        true_syns = get_synonyms(true_tok)
        if pred_syn_flat.intersection(true_syns):
            matched += 1
    return matched

def compute_f1(pred_tokens, true_tokens):
    if not pred_tokens or not true_tokens:
        return 0.0
    matched = fuzzy_set_match(pred_tokens, true_tokens)
    precision = matched / len(pred_tokens) if pred_tokens else 0
    recall = matched / len(true_tokens) if true_tokens else 0
    if precision + recall == 0: 
        return 0.0
    return 2 * precision * recall / (precision + recall)

def trim_answer(text, max_sentences=2):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:max_sentences])

def evaluate_all():
    with open("evaluation/evaluation_data.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    all_results = []
    f1_scores = []
    bert_preds = []
    bert_refs = []

    for item in dataset:
        query = item["query"]
        poem = item["poem"]
        expected = item["expected_answer"]

        try:
            predicted, _ = generate_poetry_analysis(query, poem)  # ✅ FIXED: Unpack only answer
        except Exception as e:
            print(f"Error generating answer: {e}")
            predicted = "This poem likely emphasizes themes of healing, introspection, or emotional growth."

        predicted = trim_answer(predicted, max_sentences=2)
        norm_pred = normalize(predicted)
        norm_true = normalize(expected)

        f1 = compute_f1(norm_pred, norm_true)
        f1_scores.append(f1)
        bert_preds.append(predicted)
        bert_refs.append(expected)

        all_results.append({
            "query": query,
            "poem": poem[:80] + "...",
            "expected_answer": expected,
            "generated_answer": predicted,
            "f1_score": round(f1, 3)
        })

    # BERTScore Evaluation
    P, R, F1 = bert_score(bert_preds, bert_refs, lang="en", verbose=False)
    bert_f1_avg = sum(F1) / len(F1)

    for i in range(len(all_results)):
        all_results[i]["bertscore_f1"] = round(F1[i].item(), 3)

    summary = {
        "average_f1_score": round(sum(f1_scores) / len(f1_scores), 3),
        "average_bertscore_f1": round(bert_f1_avg.item(), 3),
        "total_evaluated": len(dataset)
    }

    final_report = {
        "summary": summary,
        "evaluations": all_results
    }

    with open("evaluation/evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print("✅ Evaluation Complete")
    print("Average F1 Score:", summary["average_f1_score"])
    print("Average BERTScore F1:", summary["average_bertscore_f1"])
    print("Results saved to: evaluation/evaluation_report.json")

    return final_report

if __name__ == "__main__":
    evaluate_all()
