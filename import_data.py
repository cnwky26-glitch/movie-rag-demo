import os

import truststore

# 与网页入口保持一致，使用系统原生证书验证。
truststore.inject_into_ssl()

from graph import IMPORT_QUERY, connect_graph
from retrieval import load_movies


if __name__ == "__main__":
    movies = load_movies()
    with connect_graph() as driver:
        driver.verify_connectivity()
        database = os.environ["NEO4J_DATABASE"]
        for label, key in (("RagMovie", "id"), ("RagPerson", "name"), ("RagGenre", "name")):
            driver.execute_query(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE",
                database_=database,
            )
        records, _, _ = driver.execute_query(IMPORT_QUERY, movies=movies, database_=database)
    print(f"已导入 {records[0]['imported']} 部电影；仅更新 Rag* 标签下的样例，不清空数据库。")
