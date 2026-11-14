import os

from dotenv import load_dotenv

from retrieval import ROOT

RELATED_QUERY = """
MATCH (source:RagMovie {id: $movie_id})-[sr]-(shared)-[tr]-(other:RagMovie)
WHERE source <> other AND other.id IN $catalog_ids
  AND (shared:RagPerson OR shared:RagGenre)
  AND type(sr) IN ['DIRECTED', 'ACTED_IN', 'IN_GENRE']
  AND type(tr) IN ['DIRECTED', 'ACTED_IN', 'IN_GENRE']
WITH other, shared, collect(DISTINCT type(sr)) AS source_roles,
     collect(DISTINCT type(tr)) AS target_roles
WITH other, collect({
    kind: CASE WHEN shared:RagPerson THEN '人物' ELSE '类型' END,
    name: shared.name, source_roles: source_roles, target_roles: target_roles
}) AS evidence
RETURN other.id AS id, other.title AS title, evidence, size(evidence) AS score
ORDER BY score DESC, id ASC
LIMIT $limit
"""

IMPORT_QUERY = """
UNWIND $movies AS movie
MERGE (m:RagMovie {id: movie.id})
SET m.title = movie.title, m.plot = movie.plot
WITH m, movie
OPTIONAL MATCH (m)-[old:DIRECTED|ACTED_IN|IN_GENRE]-()
DELETE old
WITH DISTINCT m, movie
MERGE (d:RagPerson {name: movie.director})
MERGE (d)-[:DIRECTED]->(m)
MERGE (a:RagPerson {name: movie.actor})
MERGE (a)-[:ACTED_IN]->(m)
MERGE (g:RagGenre {name: movie.genre})
MERGE (m)-[:IN_GENRE]->(g)
RETURN count(m) AS imported
"""


def connect_graph():
    from neo4j import GraphDatabase

    load_dotenv(ROOT / ".env")
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )


def related_movies(movie_id, catalog_ids, limit=3):
    with connect_graph() as driver:
        records, _, _ = driver.execute_query(
            RELATED_QUERY,
            movie_id=movie_id,
            catalog_ids=catalog_ids,
            limit=limit,
            database_=os.environ["NEO4J_DATABASE"],
            routing_="r",
        )
    return [record.data() for record in records]
