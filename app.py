import os

import truststore

# 在应用入口使用系统原生证书验证，早于网络客户端导入。
truststore.inject_into_ssl()

import streamlit as st
from dotenv import load_dotenv

from generation import generate_explanation
from retrieval import INDEX_PATH, ROOT, corpus_digest, keyword_search, load_movies, load_semantic_index, semantic_search

load_dotenv(ROOT / ".env")


@st.cache_resource
def semantic_resources(index_version, corpus_version):
    return load_semantic_index(INDEX_PATH, load_movies())


st.set_page_config(page_title="电影发现助手", page_icon="🎬")
st.title("🎬 电影发现助手")
st.caption("检索电影，查看人物与类型关联，用片库中的事实解释结果。当前为 7 部电影的演示。")
movies = load_movies()

with st.sidebar:
    st.header("检索选项")
    method = st.radio("检索方式", ["标题／属性匹配", "语义检索"])
    genre = st.selectbox("电影类型", ["全部"] + sorted({movie["genre"] for movie in movies}))
    st.caption("标题／属性匹配不需要下载模型。语义检索需先生成索引。")

with st.form("search"):
    query = st.text_input("想看什么电影？", placeholder="例如：流浪地球、张国荣主演的电影、带着地球逃离太阳系")
    submitted = st.form_submit_button("查找电影", type="primary")

if submitted:
    st.session_state.pop("search_result", None)
    st.session_state.pop("explanation", None)
    st.session_state.pop("related", None)
    if not query.strip():
        st.warning("请输入片名、人物、类型或剧情描述。")
    else:
        genre_filter = None if genre == "全部" else genre
        if method == "标题／属性匹配":
            results = keyword_search(query, movies, genre=genre_filter)
        else:
            if not INDEX_PATH.exists():
                st.error("尚未生成语义索引，请先按 README 运行 create_embeddings.py。")
                st.stop()
            with st.spinner("正在加载语义模型并检索…"):
                model, embeddings, model_name = semantic_resources(
                    INDEX_PATH.stat().st_mtime_ns, corpus_digest(movies)
                )
                results = semantic_search(query, movies, model, embeddings, genre=genre_filter)
        st.session_state["search_result"] = {"query": query, "results": results, "method": method, "genre": genre}

if "search_result" in st.session_state:
    search = st.session_state["search_result"]
    results = search["results"]
    st.subheader(f"“{search['query']}”的检索结果")
    st.caption(f"方式：{search['method']} · 类型：{search['genre']}")
    if not results:
        st.info("当前片库没有找到匹配结果。可以换一个描述，或检查类型筛选。")
    else:
        for rank, row in enumerate(results, 1):
            movie = row["movie"]
            st.markdown(f"**{rank}.《{movie['title']}》**")
            st.write(f"导演：{movie['director']} · 主演：{movie['actor']} · 类型：{movie['genre']}")
            st.write(movie["plot"])
            if row["score"] is not None:
                st.caption(f"向量相似度：{row['score']:.3f}，仅用于排序，不代表置信度。")
        if search["method"] == "语义检索":
            st.caption("示例最低相似度为 0.45，尚未校准；仍可能误匹配。明确片名会优先保留。")

        with st.expander("补充查看：首部电影的图谱关联"):
            st.caption("关联作品可能与本次需求不同，不替代上面的直接检索结果。需配置并导入 Neo4j。")
            if st.button("查询关联作品"):
                required = ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE")
                if not all(os.getenv(key) for key in required):
                    st.error("请先在 .env 中填写 Neo4j 配置，配置后重启应用。")
                else:
                    from graph import related_movies
                    st.session_state["related"] = related_movies(
                        results[0]["movie"]["id"], [movie["id"] for movie in movies]
                    )
            if "related" in st.session_state:
                related = st.session_state["related"]
                if not related:
                    st.info("未找到关联作品，请确认已导入样例数据。")
                for row in related:
                    st.markdown(f"**《{row['title']}》** · 共同人物／类型数：{row['score']}")
                    for evidence in row["evidence"]:
                        if evidence["kind"] == "类型":
                            st.write(f"共同类型：{evidence['name']}")
                        else:
                            labels = {"DIRECTED": "导演", "ACTED_IN": "演员"}
                            left = "、".join(sorted(labels[role] for role in evidence["source_roles"]))
                            right = "、".join(sorted(labels[role] for role in evidence["target_roles"]))
                            st.write(f"共同人物：{evidence['name']}（原片：{left}；关联片：{right}）")

        st.subheader("按需求选择推荐依据")
        st.caption("点击后调用 DeepSeek，可能产生接口费用。模型只选择上面候选的事实编号，页面展示片库原文。")
        if st.button("生成有据说明"):
            st.session_state.pop("explanation", None)
            required = ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL")
            if not all(os.getenv(key) for key in required):
                st.error("请先在 .env 中填写 DeepSeek 配置，配置后重启应用。")
            else:
                from openai import APIError
                try:
                    with st.spinner("正在选择相关证据…"):
                        st.session_state["explanation"] = generate_explanation(search["query"], results)
                except ValueError as error:
                    st.error(f"生成结果未通过校验：{error}")
                except APIError as error:
                    st.error(f"接口请求失败（{type(error).__name__}），请检查服务和配置。")
        for item in st.session_state.get("explanation", []):
            st.markdown(f"**《{item['title']}》**")
            for fact in item["facts"]:
                st.write(fact)
