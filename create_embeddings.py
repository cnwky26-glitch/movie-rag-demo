import argparse

import numpy as np

from retrieval import INDEX_PATH, ROOT, corpus_digest, load_movies, movie_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为电影语料生成带数据指纹的归一化向量索引")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output", type=str, default=str(INDEX_PATH))
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    movies = load_movies()
    model = SentenceTransformer(args.model)
    vectors = model.encode(movie_texts(movies), normalize_embeddings=True, show_progress_bar=True)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embeddings=vectors,
        movie_ids=np.array([movie["id"] for movie in movies]),
        model_name=np.array(args.model),
        corpus_digest=np.array(corpus_digest(movies)),
    )
    print(f"已生成 {output}，形状 {vectors.shape}，模型 {args.model}")
