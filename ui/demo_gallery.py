"""Screenshots of actual archived experiments, with verifiable image hashes."""
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "static" / "competition_gallery"


def gallery_records():
    path = GALLERY / "manifest.json"
    if not path.is_file():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))["screenshots"]
    for row in records:
        image = (GALLERY / row["file"]).resolve()
        if not image.is_relative_to(GALLERY.resolve()) or sha256(image.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Screenshot checksum mismatch")
    return records


def gallery_page(archive, comparisons, result):
    st.markdown("# 运行截图")
    st.write("按一次实验的执行顺序，回看分工、审查和演进。")
    try:
        rows = gallery_records()
    except (OSError, ValueError, KeyError):
        st.warning("截图文件不完整，可在演示总览中查看实验。")
        return
    if not rows:
        st.info("截图正在整理；可从演示总览查看已完成实验。")
        return
    titles = [row["title"] for row in rows]
    selected = st.selectbox("选择展示内容", range(len(rows)), format_func=lambda i: f"{i+1}. {titles[i]}", key="gallery_item")
    row = rows[selected]
    st.markdown("### " + row["title"])
    st.write(row["description"])
    st.image(str(GALLERY / row["file"]), width="stretch")
    columns = st.columns([1,1,1])
    def move(delta):
        st.session_state.gallery_item = (selected + delta) % len(rows)
    columns[0].button("上一张", on_click=move, args=(-1,), width="stretch")
    columns[1].button("下一张", on_click=move, args=(1,), type="primary", width="stretch")
    params={"page": row["page"], "artifact_run": row["baseline_run_id"]}
    if row.get("view"): params["view"] = row["view"]
    columns[2].link_button("打开对应实验", "?" + urlencode(params), width="stretch")
    with st.expander("截图来源与下载"):
        st.write("已完成实验截图 · " + row["captured_at"])
        st.write("实验编号：" + row["experiment_id"])
        st.write("执行后端：" + row["backend"] + "；提示词优化模型：" + row["model"])
        st.download_button("下载原图", (GALLERY / row["file"]).read_bytes(), row["file"], "image/png")
