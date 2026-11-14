import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "movies.json"
INDEX_PATH = ROOT / "data" / "movie_index.npz"
MIN_SCORE = 0.45
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))


def load_movies():
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def movie_texts(movies):
    return [
        f"{m['title']}。导演：{m['director']}。主演：{m['actor']}。"
        f"类型：{m['genre']}。简介：{m['plot']}"
        for m in movies
    ]


def corpus_digest(movies):
    content = json.dumps(movies, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def normalize_title(text):
    return text.strip().strip("《》").casefold()


def keyword_search(query, movies, top_k=3, genre=None):
    query = normalize_title(query)
    if not query:
        return []
    results = []
    for movie in movies:
        if genre is not None and movie["genre"] != genre:
            continue
        fields = [
            field for field in ("title", "director", "actor", "genre", "plot")
            if movie[field].casefold() in query or query in movie[field].casefold()
        ]
        if fields:
            results.append({"movie": movie, "score": None, "matched_fields": fields})
    results.sort(key=lambda row: (
        normalize_title(row["movie"]["title"]) != query,
        "title" not in row["matched_fields"],
        row["movie"]["id"],
    ))
    return results[:top_k]


def load_semantic_index(path, movies):
    from sentence_transformers import SentenceTransformer

    with np.load(path, allow_pickle=False) as index:
        if str(index["corpus_digest"]) != corpus_digest(movies):
            raise ValueError("电影数据已变化，请重新运行 create_embeddings.py。")
        if index["movie_ids"].tolist() != [movie["id"] for movie in movies]:
            raise ValueError("索引与电影 ID 顺序不一致，请重新生成索引。")
        embeddings = index["embeddings"].copy()
        model_name = str(index["model_name"])
    if embeddings.ndim != 2 or embeddings.shape[0] != len(movies):
        raise ValueError("索引矩阵形状不正确，请重新生成索引。")
    return SentenceTransformer(model_name), embeddings, model_name


def semantic_search(query, movies, model, embeddings, top_k=3, min_score=MIN_SCORE, genre=None):
    query = query.strip()
    if not query:
        return []
    query_vector = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ query_vector
    eligible = [
        i for i, movie in enumerate(movies)
        if (genre is None or movie["genre"] == genre)
        and (scores[i] >= min_score or normalize_title(movie["title"]) == normalize_title(query))
    ]
    eligible.sort(key=lambda i: (
        normalize_title(movies[i]["title"]) != normalize_title(query),
        -float(scores[i]), movies[i]["id"],
    ))
    return [
        {"movie": movies[i], "score": float(scores[i]), "matched_fields": []}
        for i in eligible[:top_k]
    ]
