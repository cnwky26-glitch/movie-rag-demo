import argparse
import json
import platform
import statistics
from datetime import datetime, timezone
from time import perf_counter

from retrieval import MIN_SCORE, ROOT, corpus_digest, keyword_search, load_movies, load_semantic_index, normalize_title, semantic_search


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在同一片库和人工样例查询上比较检索结果")
    parser.add_argument("--index", action="append", default=[], help="可重复传入不同模型的 .npz 索引")
    parser.add_argument("--top-k", type=int, choices=range(1, 8), default=5)
    parser.add_argument("--output", default="experiments/retrieval_results.json")
    args = parser.parse_args()

    movies = load_movies()
    queries = json.loads((ROOT / "data" / "eval_queries.json").read_text(encoding="utf-8"))
    catalog_ids = {movie["id"] for movie in movies}
    for case in queries:
        if not set(case["relevant_ids"]) <= catalog_ids:
            raise ValueError(f"查询 {case['id']} 的相关电影不在片库中。")

    systems = [("标题／属性匹配", None, None)]
    for filename in args.index:
        model, vectors, name = load_semantic_index(ROOT / filename, movies)
        model.encode(["电影"], normalize_embeddings=True)
        systems.append((name, model, vectors))

    reports = []
    for name, model, vectors in systems:
        cases = []
        for case in queries:
            start = perf_counter()
            if model is None:
                ranking = keyword_search(case["query"], movies, top_k=args.top_k)
                accepted = ranking
            else:
                ranking = semantic_search(
                    case["query"], movies, model, vectors, top_k=args.top_k, min_score=-float("inf")
                )
                accepted = [row for row in ranking if row["score"] >= MIN_SCORE
                            or normalize_title(row["movie"]["title"]) == normalize_title(case["query"])]
            latency_ms = (perf_counter() - start) * 1000
            ranked_ids = [row["movie"]["id"] for row in ranking]
            relevant = set(case["relevant_ids"])
            reciprocal_rank = next((1 / rank for rank, movie_id in enumerate(ranked_ids, 1)
                                    if movie_id in relevant), 0.0)
            cases.append({
                **case,
                "ranked_ids": ranked_ids,
                "scores": [row["score"] for row in ranking],
                "accepted_ids": [row["movie"]["id"] for row in accepted],
                "latency_ms": latency_ms,
                "reciprocal_rank": reciprocal_rank if relevant else None,
                "recall": len(set(ranked_ids) & relevant) / len(relevant) if relevant else None,
            })
        positives = [case for case in cases if case["relevant_ids"]]
        negatives = [case for case in cases if not case["relevant_ids"]]
        reports.append({
            "system": name,
            "positive_queries": len(positives),
            "negative_queries": len(negatives),
            f"mrr@{args.top_k}": statistics.mean(case["reciprocal_rank"] for case in positives),
            f"recall@{args.top_k}": statistics.mean(case["recall"] for case in positives),
            "negative_false_accept_rate": statistics.mean(bool(case["accepted_ids"]) for case in negatives),
            "positive_empty_result_rate": statistics.mean(not case["accepted_ids"] for case in positives),
            "latency_ms_median": statistics.median(case["latency_ms"] for case in cases),
            "cases": cases,
        })
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "movie_count": len(movies),
        "query_count": len(queries),
        "corpus_digest": corpus_digest(movies),
        "top_k": args.top_k,
        "semantic_acceptance_threshold": MIN_SCORE,
        "limitations": [
            "7 部电影、人工构造且未独立标注的查询，仅供小样本功能分析。",
            "MRR 和 Recall 使用未按阈值截断的排序；明确片名优先。拒识率单独统计。",
            "只测单次热加载查询耗时，不含模型和索引加载、数据库、网络及生成。",
            "未用独立验证集校准相似度阈值；结果不能代表真实用户或线上业务收益。",
        ],
        "systems": reports,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": [
        {key: value for key, value in row.items() if key != "cases"} for row in reports
    ]}, ensure_ascii=False, indent=2))
