#!/usr/bin/env python3
"""Compare LDA and LSA topics in the processed Kaggle complaints collection."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import spacy
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

NEGATIONS = {"no", "not", "never", "neither", "nor", "without"}
TOP_N = 10


def report_status(message: str) -> None:
    print("[status] {}".format(message), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to complaints_processed.csv")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Local output directory")
    parser.add_argument("--sample-size", type=int, default=5000, help="Documents used for topic-count selection")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_and_deduplicate(path: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    raw = pd.read_csv(
        path,
        usecols=["product", "narrative"],
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    total_rows = int(len(raw))
    raw["narrative"] = raw["narrative"].astype(str)
    empty_mask = raw["narrative"].str.strip().eq("")
    empty_count = int(empty_mask.sum())
    nonempty = raw.loc[~empty_mask].copy()

    # Collapse exact source narratives first. Normalize whitespace only afterward.
    exact_duplicate_rows = int(nonempty.duplicated("narrative", keep="first").sum())
    label_sets = nonempty.groupby("narrative", sort=False)["product"].agg(lambda values: sorted(set(values)))
    ambiguous_texts = {text for text, labels in label_sets.items() if len(labels) > 1}
    unique = nonempty.drop_duplicates("narrative", keep="first").copy().reset_index(drop=True)
    unique.insert(0, "narrative_id", np.arange(1, len(unique) + 1))
    unique["product_label_status"] = np.where(unique["narrative"].isin(ambiguous_texts), "ambiguous", "single")
    unique["product_label"] = unique["narrative"].map(
        lambda text: "ambiguous" if text in ambiguous_texts else label_sets[text][0]
    )
    unique["narrative"] = unique["narrative"].map(lambda value: re.sub(r"\s+", " ", value.strip()))

    audit: Dict[str, object] = {
        "source_file": path.name,
        "raw_rows": total_rows,
        "empty_narratives": empty_count,
        "nonempty_rows": int(len(nonempty)),
        "duplicate_rows_removed": exact_duplicate_rows,
        "unique_narratives": int(len(unique)),
        "unique_narratives_with_conflicting_product_labels": int(len(ambiguous_texts)),
        "raw_product_counts": {str(k): int(v) for k, v in raw["product"].value_counts().items()},
    }
    return unique, audit


def preprocess_texts(texts: Sequence[str]) -> Tuple[List[List[str]], int]:
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError as exc:
        raise SystemExit(
            "Missing spaCy model. Run `python -m spacy download en_core_web_sm` in the project environment."
        ) from exc

    processed: List[List[str]] = []
    empty_after_processing = 0
    total = len(texts)
    progress_interval = max(1000, total // 20)
    report_status("spaCy preprocessing: 0/{:,} narratives (0%).".format(total))
    for processed_count, doc in enumerate(nlp.pipe(texts, batch_size=256), start=1):
        tokens: List[str] = []
        for token in doc:
            if token.is_space or token.is_punct or not token.is_alpha:
                continue
            lemma = token.lemma_.lower().strip()
            if not lemma:
                continue
            if token.is_stop and token.lower_ not in NEGATIONS and lemma not in NEGATIONS:
                continue
            tokens.append(lemma)
        if not tokens:
            empty_after_processing += 1
        processed.append(tokens)
        if processed_count % progress_interval == 0 or processed_count == total:
            percent = 100.0 * processed_count / total if total else 100.0
            report_status(
                "spaCy preprocessing: {:,}/{:,} narratives ({:.0f}%).".format(
                    processed_count, total, percent
                )
            )
    return processed, empty_after_processing


def with_bigrams(tokens: Sequence[str]) -> List[str]:
    sequence: List[str] = []
    for index, token in enumerate(tokens):
        sequence.append(token)
        if index + 1 < len(tokens):
            sequence.append(token + " " + tokens[index + 1])
    return sequence


def build_vectorizer(vectorizer_type: str):
    common = {
        "ngram_range": (1, 2),
        "min_df": 5,
        "max_df": 0.95,
        "max_features": 50000,
        "lowercase": False,
        "dtype": np.float64,
    }
    if vectorizer_type == "count":
        return CountVectorizer(**common)
    return TfidfVectorizer(**common, sublinear_tf=True)


def topic_terms(weights: np.ndarray, feature_names: np.ndarray, top_n: int = TOP_N) -> List[List[str]]:
    terms: List[List[str]] = []
    for row in weights:
        indices = np.argsort(row)[::-1][:top_n]
        terms.append([str(feature_names[index]) for index in indices])
    return terms


def coherence_score(topics: List[List[str]], coherence_texts: List[List[str]], dictionary: Dictionary) -> float:
    model = CoherenceModel(
        topics=topics,
        texts=coherence_texts,
        dictionary=dictionary,
        coherence="c_v",
        topn=TOP_N,
        processes=1,
    )
    return float(model.get_coherence())


def matrix_summary(name: str, matrix) -> Dict[str, object]:
    documents, features = matrix.shape
    nonzero = int(matrix.nnz)
    total_cells = int(documents * features)
    return {
        "representation": name,
        "documents": int(documents),
        "features": int(features),
        "nonzero_values": nonzero,
        "density": float(nonzero / total_cells) if total_cells else 0.0,
        "mean_nonzero_features_per_document": float(nonzero / documents) if documents else 0.0,
    }


def choose_k(scores: pd.DataFrame, model_name: str) -> int:
    subset = scores.loc[scores["model"].eq(model_name)].sort_values(
        ["c_v_coherence", "k"], ascending=[False, True]
    )
    best_score = float(subset.iloc[0]["c_v_coherence"])
    # Prefer the smaller, easier-to-explain model when scores are within 0.01.
    tied = subset.loc[subset["c_v_coherence"] >= best_score - 0.01]
    return int(tied["k"].min())


def summarize_topic_rows(model_name: str, terms: List[List[str]], assignments: np.ndarray,
                         confidence: np.ndarray, frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.Series(assignments).value_counts().reindex(range(len(terms)), fill_value=0)
    rows = []
    examples = []
    for topic_index, words in enumerate(terms):
        rows.append({
            "model": model_name,
            "topic": topic_index + 1,
            "unique_narratives": int(counts.iloc[topic_index]),
            "share": float(counts.iloc[topic_index] / len(frame)),
            "top_terms": ", ".join(words),
        })
        candidate_indices = np.flatnonzero(assignments == topic_index)
        best_indices = sorted(candidate_indices, key=lambda idx: float(confidence[idx]), reverse=True)[:3]
        for rank, row_index in enumerate(best_indices, start=1):
            examples.append({
                "model": model_name,
                "topic": topic_index + 1,
                "rank": rank,
                "confidence_or_abs_loading": float(confidence[row_index]),
                "narrative_excerpt_local_only": frame.iloc[row_index]["narrative"][:350],
            })
    return pd.DataFrame(rows), pd.DataFrame(examples)


def print_top_topics(summary: pd.DataFrame, top_n: int = 5) -> None:
    print("\nMost frequent topics by unique narratives (top {} per model):".format(top_n), flush=True)
    for model_name in ("LDA", "LSA"):
        model_rows = summary.loc[summary["model"].eq(model_name)].sort_values(
            ["unique_narratives", "topic"], ascending=[False, True]
        ).head(top_n)
        print("{}:".format(model_name), flush=True)
        for row in model_rows.itertuples(index=False):
            print(
                "  Topic {} | {:,} unique narratives ({:.1%}) | {}".format(
                    row.topic, row.unique_narratives, row.share, row.top_terms
                ),
                flush=True,
            )


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit("Input CSV not found: {}".format(args.input))
    if args.k_min < 2 or args.k_max < args.k_min:
        raise SystemExit("Choose a valid topic range with k_min >= 2 and k_max >= k_min.")

    args.output.mkdir(parents=True, exist_ok=True)
    report_status("Loading source CSV: {}".format(args.input))
    frame, audit = load_and_deduplicate(args.input)
    report_status(
        "Loaded {:,} rows; {:,} unique nonempty narratives remain after exact deduplication.".format(
            audit["raw_rows"], len(frame)
        )
    )
    tokenized, empty_after_processing = preprocess_texts(frame["narrative"].tolist())
    keep_mask = np.array([bool(tokens) for tokens in tokenized])
    frame = frame.loc[keep_mask].reset_index(drop=True)
    tokenized = [tokens for tokens, keep in zip(tokenized, keep_mask) if keep]
    audit["empty_after_spacy_processing"] = int(empty_after_processing)
    audit["documents_analyzed"] = int(len(frame))
    audit["theme_frequency_definition"] = "Counts of unique nonempty narratives after preprocessing; conflicting product labels remain ambiguous."

    model_max_k = min(args.k_max, len(frame) - 1)
    if model_max_k < args.k_min:
        raise SystemExit("Not enough documents for the requested topic range.")
    rng = np.random.RandomState(args.random_state)
    sample_size = min(args.sample_size, len(frame))
    sample_indices = np.sort(rng.choice(len(frame), size=sample_size, replace=False))
    sample_tokens = [tokenized[index] for index in sample_indices]
    coherence_texts = [with_bigrams(tokens) for tokens in sample_tokens]
    dictionary = Dictionary(coherence_texts)

    sample_strings = [" ".join(tokens) for tokens in sample_tokens]
    report_status("Vectorizing {:,} documents for topic-count selection.".format(sample_size))
    count_vectorizer_sample = build_vectorizer("count")
    tfidf_vectorizer_sample = build_vectorizer("tfidf")
    X_count_sample = count_vectorizer_sample.fit_transform(sample_strings)
    X_tfidf_sample = tfidf_vectorizer_sample.fit_transform(sample_strings)
    report_status(
        "Selection matrices ready: Count={} and TF-IDF={}.".format(
            X_count_sample.shape, X_tfidf_sample.shape
        )
    )
    count_features_sample = count_vectorizer_sample.get_feature_names_out()
    tfidf_features_sample = tfidf_vectorizer_sample.get_feature_names_out()
    model_max_k = min(
        model_max_k,
        X_count_sample.shape[0] - 1,
        X_count_sample.shape[1] - 1,
        X_tfidf_sample.shape[0] - 1,
        X_tfidf_sample.shape[1] - 1,
    )
    if model_max_k < args.k_min:
        raise SystemExit("The sample matrices are too small for the requested topic range.")

    score_rows = []
    for k in range(args.k_min, model_max_k + 1):
        report_status("Selecting LDA topic count: fitting K={}.".format(k))
        lda = LatentDirichletAllocation(
            n_components=k,
            learning_method="batch",
            max_iter=8,
            evaluate_every=-1,
            random_state=args.random_state,
            n_jobs=1,
        )
        lda.fit(X_count_sample)
        lda_words = topic_terms(lda.components_, count_features_sample)
        lda_coherence = coherence_score(lda_words, coherence_texts, dictionary)
        lda_perplexity = float(lda.perplexity(X_count_sample))
        report_status("LDA K={} complete; c_v={:.4f}.".format(k, lda_coherence))
        score_rows.append({
            "model": "LDA",
            "k": k,
            "c_v_coherence": lda_coherence,
            "perplexity_diagnostic": lda_perplexity,
            "top_terms_sample": " | ".join("; ".join(words) for words in lda_words),
        })

    max_svd_k = model_max_k
    report_status("Fitting SVD on the TF-IDF selection matrix (maximum K={}).".format(max_svd_k))
    svd_sample = TruncatedSVD(n_components=max_svd_k, n_iter=7, random_state=args.random_state)
    svd_sample.fit(X_tfidf_sample)
    for k in range(args.k_min, model_max_k + 1):
        lsa_words = topic_terms(svd_sample.components_[:k], tfidf_features_sample)
        lsa_coherence = coherence_score(lsa_words, coherence_texts, dictionary)
        report_status("LSA K={} coherence complete; c_v={:.4f}.".format(k, lsa_coherence))
        score_rows.append({
            "model": "LSA",
            "k": k,
            "c_v_coherence": lsa_coherence,
            "perplexity_diagnostic": np.nan,
            "top_terms_sample": " | ".join("; ".join(words) for words in lsa_words),
        })

    scores = pd.DataFrame(score_rows).sort_values(["model", "k"])
    scores.to_csv(args.output / "topic_count_selection.csv", index=False)
    lda_k = choose_k(scores, "LDA")
    lsa_k = choose_k(scores, "LSA")
    audit["topic_count_selection"] = {
        "candidate_k": list(range(args.k_min, model_max_k + 1)),
        "selection_sample_size": int(sample_size),
        "random_state": int(args.random_state),
        "criterion": "Maximum c_v coherence; when scores are within 0.01, prefer the smaller K; inspect top terms and representative narratives for interpretability.",
        "selected_k": {"LDA": lda_k, "LSA": lsa_k},
    }
    with (args.output / "data_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)

    # Fit the selected models to all unique narratives.
    all_strings = [" ".join(tokens) for tokens in tokenized]
    report_status("Vectorizing all {:,} unique narratives.".format(len(all_strings)))
    count_vectorizer = build_vectorizer("count")
    tfidf_vectorizer = build_vectorizer("tfidf")
    X_count = count_vectorizer.fit_transform(all_strings)
    X_tfidf = tfidf_vectorizer.fit_transform(all_strings)
    report_status(
        "Full matrices ready: Count={} and TF-IDF={}.".format(X_count.shape, X_tfidf.shape)
    )
    pd.DataFrame([
        matrix_summary("CountVectorizer", X_count),
        matrix_summary("TfidfVectorizer", X_tfidf),
    ]).to_csv(args.output / "vectorization_comparison.csv", index=False)

    lda_final = LatentDirichletAllocation(
        n_components=lda_k,
        learning_method="online",
        max_iter=10,
        batch_size=2048,
        evaluate_every=-1,
        random_state=args.random_state,
        n_jobs=1,
        verbose=1,
    )
    report_status("Fitting final LDA model with K={}.".format(lda_k))
    lda_distribution = lda_final.fit_transform(X_count)
    report_status("Final LDA model complete.")
    lda_assignments = lda_distribution.argmax(axis=1)
    lda_confidence = lda_distribution.max(axis=1)
    lda_terms = topic_terms(lda_final.components_, count_vectorizer.get_feature_names_out())

    lsa_final = TruncatedSVD(n_components=lsa_k, n_iter=7, random_state=args.random_state)
    report_status("Fitting final LSA model with K={}.".format(lsa_k))
    lsa_scores = lsa_final.fit_transform(X_tfidf)
    report_status("Final LSA model complete.")
    lsa_assignments = np.abs(lsa_scores).argmax(axis=1)
    lsa_confidence = np.abs(lsa_scores).max(axis=1)
    lsa_terms = topic_terms(lsa_final.components_, tfidf_vectorizer.get_feature_names_out())

    lda_summary, lda_examples = summarize_topic_rows("LDA", lda_terms, lda_assignments, lda_confidence, frame)
    lsa_summary, lsa_examples = summarize_topic_rows("LSA", lsa_terms, lsa_assignments, lsa_confidence, frame)
    topic_summary = pd.concat([lda_summary, lsa_summary], ignore_index=True).sort_values(
        ["model", "unique_narratives", "topic"], ascending=[True, False, True]
    ).reset_index(drop=True)
    topic_summary.to_csv(args.output / "selected_topics_and_frequencies.csv", index=False)
    pd.concat([lda_examples, lsa_examples], ignore_index=True).to_csv(args.output / "representative_narratives.csv", index=False)
    pd.DataFrame({
        "narrative_id": frame["narrative_id"],
        "product_label": frame["product_label"],
        "product_label_status": frame["product_label_status"],
        "lda_topic": lda_assignments + 1,
        "lda_probability": lda_confidence,
        "lsa_factor": lsa_assignments + 1,
        "lsa_absolute_score": lsa_confidence,
    }).to_csv(args.output / "topic_assignments.csv", index=False)

    report_status("Writing analysis summaries and representative narratives.")
    print_top_topics(topic_summary, top_n=5)
    print("Raw rows: {}".format(audit["raw_rows"]))
    print("Empty narratives removed: {}".format(audit["empty_narratives"]))
    print("Duplicate rows removed: {}".format(audit["duplicate_rows_removed"]))
    print("Unique narratives analyzed: {}".format(audit["documents_analyzed"]))
    print("Narratives with conflicting product labels: {}".format(audit["unique_narratives_with_conflicting_product_labels"]))
    print("Selected K: LDA={}, LSA={}".format(lda_k, lsa_k))
    if lda_k == model_max_k or lsa_k == model_max_k:
        print("A selected K is at the upper search boundary; inspect the score curve and expand the range if needed.")
    print("LSA cumulative explained variance: {:.4f}".format(float(lsa_final.explained_variance_ratio_.sum())))
    print("Results saved to {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
