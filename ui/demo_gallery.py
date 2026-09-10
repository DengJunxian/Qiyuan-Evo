"""Image-first homepage backed by local, source-verified screenshots."""
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from urllib.parse import quote, urlencode

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "static" / "competition_gallery"
DOMAIN_GALLERY = ROOT / "static" / "showcase_gallery"


def _verified_records(directory):
    path = directory / "manifest.json"
    if not path.is_file():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))["screenshots"]
    for row in records:
        image = (directory / row["file"]).resolve()
        if not image.is_relative_to(directory.resolve()) or sha256(image.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Screenshot checksum mismatch")
    return records


def gallery_records():
    """Keep experiment provenance separate from the original Civitas images."""
    return _verified_records(GALLERY)


def domain_gallery_records():
    return _verified_records(DOMAIN_GALLERY)


def _experiment_url(row):
    params = {"page": row["page"]}
    if row.get("baseline_run_id"):
        params["artifact_run"] = row["baseline_run_id"]
    if row.get("view"):
        params["view"] = row["view"]
    return "?" + urlencode(params)


def _image_grid(rows, directory, *, archived):
    figures = []
    for index, row in enumerate(rows, 1):
        src = "app/static/" + directory.name + "/" + quote(row["file"])
        title = escape(row["title"])
        link_label = "查看实验记录" if archived else "打开" + row["page"]
        figures.append(
            f'<figure class="qg-image">'
            f'<figcaption><h3><span>{index:02d}</span>{title}</h3></figcaption>'
            f'<a class="qg-original" href="{src}" target="_blank" rel="noopener" '
            f'aria-label="查看大图：{title}">'
            f'<img src="{src}" alt="{title}，界面截图" loading="{"eager" if index < 3 else "lazy"}">'
            '</a><div class="qg-image-links">'
            f'<a href="{src}" target="_blank" rel="noopener">查看大图 ↗</a>'
            f'<a href="{escape(_experiment_url(row), quote=True)}" target="_self">{escape(link_label)} →</a>'
            '</div></figure>'
        )
    st.html('<div class="qg-grid">' + "".join(figures) + '</div>')


def gallery_page(archive=None, comparisons=None, result=None):
    # Browsing images does not depend on the model runtime or experiment store.
    st.html('<header class="qg-heading"><h1>项目演示</h1>'
            '<a class="qg-enter" href="?' + urlencode({"page": "自演进驾驶舱"}) +
            '" target="_self">进入实验 →</a></header>')
    collections = (
        ("team-gallery", "团队自演进", "已完成实验截图", GALLERY, gallery_records, True),
        ("domain-gallery", "金融政策风洞", "原 Civitas 项目截图", DOMAIN_GALLERY, domain_gallery_records, False),
    )
    st.html('<nav class="qg-jump" aria-label="图集目录">'
            '<a href="#team-gallery" target="_self">团队自演进</a>'
            '<a href="#domain-gallery" target="_self">金融政策风洞</a></nav>')
    for anchor, title, source, directory, loader, archived in collections:
        st.html(f'<div id="{anchor}" class="qg-section"><h2>{title}</h2><span>{source}</span></div>')
        try:
            rows = loader()
        except (OSError, ValueError, KeyError, TypeError):
            st.warning(f"{title}图集暂时无法读取，可从顶部导航打开对应页面。")
            continue
        if not rows:
            st.info(f"{title}暂无截图，可从顶部导航查看系统。")
            continue
        _image_grid(rows, directory, archived=archived)

    with st.expander("图片来源"):
        st.write("团队自演进截图来自项目中保存的已完成实验。每张图下方可打开对应实验记录。")
        st.markdown("金融政策风洞截图来自 [Civitas-Economica-Demo 原仓库]"
                    "(https://github.com/DengJunxian/Civitas-Economica-Demo/tree/"
                    "585ffcbaaeb292f0addc10331ae7dae0c57ebd84/static/showcase_gallery)。")
        for directory, label in ((GALLERY, "下载实验截图清单"), (DOMAIN_GALLERY, "下载原项目截图清单")):
            path = directory / "manifest.json"
            if path.is_file():
                st.download_button(label, path.read_bytes(), directory.name + "_manifest.json", "application/json")
