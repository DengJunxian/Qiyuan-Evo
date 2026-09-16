"""Product presentation with clearly labeled demo and original project screenshots."""
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from urllib.parse import quote, urlencode

import streamlit as st
from ui.showcase_visuals import presentation_cases, visual_story

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "static" / "evolution_story_gallery"
ARCHIVE_GALLERY = ROOT / "static" / "competition_gallery"
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
    return _verified_records(GALLERY)


def archived_gallery_records():
    return _verified_records(ARCHIVE_GALLERY)


def domain_gallery_records():
    return _verified_records(DOMAIN_GALLERY)


def _experiment_url(row):
    params = {"page": row["page"], "appearance": st.session_state.get("ui_theme", "dark")}
    if row.get("baseline_run_id"):
        params["artifact_run"] = row["baseline_run_id"]
    if row.get("view"):
        params["view"] = row["view"]
    if row.get("chapter"):
        params["chapter"] = row["chapter"]
    return "?" + urlencode(params)


def _src(directory, row):
    return "app/static/" + directory.name + "/" + quote(row["file"])


def _gallery_markup(rows, directory, *, anchor, title, heading, note, illustrative):
    source = "演示案例截图 · 示例数据" if illustrative else "原 Civitas 项目截图"
    destination = {"page": "演进案例", "chapter": 1} if illustrative else {"page": "金融政策风洞"}
    figures = []
    for row in rows:
        src = _src(directory, row)
        label = escape(row["title"])
        description = escape(row.get("description", ""))
        figures.append(
            '<figure class="qg-image">'
            f'<a class="qg-original" href="{src}" target="_blank" rel="noopener" '
            f'data-fullscreen data-title="{label}" aria-label="全屏查看：{label}">'
            f'<img src="{src}" alt="{label}，界面截图" loading="lazy" width="1600" height="900"></a>'
            f'<figcaption><h3>{label}</h3><p>{description}</p><div class="qg-image-links">'
            f'<a href="{escape(_experiment_url(row), quote=True)}" target="_self">'
            f'{"打开此阶段" if illustrative else "打开" + escape(row["page"])} <span aria-hidden="true">↗</span></a>'
            f'<a href="{src}" target="_blank" rel="noopener" data-fullscreen data-title="{label}">全屏查看 <span aria-hidden="true">＋</span></a>'
            '</div></figcaption></figure>'
        )
    return (
        f'<section class="qy-gallery" id="{anchor}" data-source="{source}" aria-labelledby="{anchor}-title">'
        '<header class="qy-section-head"><div>'
        f'<div class="qy-eyebrow">{title}</div><h2 id="{anchor}-title">{heading}</h2><p>{note}</p></div>'
        f'<a class="qy-text-link" href="{_experiment_url(destination)}" target="_self">'
        f'{"打开完整案例" if illustrative else "进入金融风洞"} <span aria-hidden="true">↗</span></a></header>'
        f'<div class="qy-gallery-track" tabindex="0" aria-label="{title}图集，使用左右方向键切换">' + ''.join(figures) + '</div>'
        f'<div class="qy-gallery-controls"><span data-counter aria-live="polite">01 / {len(rows):02d}   {source}</span>'
        '<div class="qy-control-group"><button type="button" data-action="play" aria-pressed="false">自动播放</button>'
        f'<button type="button" data-action="previous" aria-label="{title}：上一张" disabled>‹</button>'
        f'<button type="button" data-action="next" aria-label="{title}：下一张">›</button></div></div></section>'
    )


def gallery_page(archive=None, comparisons=None, result=None):
    # Archived visualizations are optional; local screenshots remain available offline.
    collections = []
    warnings = []
    for directory, loader, title in ((GALLERY, gallery_records, "团队自演进"), (DOMAIN_GALLERY, domain_gallery_records, "金融政策风洞")):
        try:
            rows = loader()
        except (OSError, ValueError, KeyError, TypeError):
            rows = []
            warnings.append(f"{title}图集暂时无法读取，可从顶部导航打开对应页面。")
        collections.append((directory, rows))
    team_rows, domain_rows = collections[0][1], collections[1][1]
    try:
        scene, proof = visual_story(presentation_cases(), _experiment_url)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError):
        scene, proof = "", ""
    hero = ""
    hero_row = team_rows[0] if team_rows else domain_rows[0] if domain_rows else None
    if hero_row and not scene:
        directory = GALLERY if team_rows else DOMAIN_GALLERY
        src = _src(directory, hero_row)
        label = escape(hero_row["title"])
        hero = (f'<figure class="qy-product"><a class="qy-device" href="{src}" target="_blank" rel="noopener" '
                f'data-fullscreen data-title="{label}" aria-label="全屏查看：{label}">'
                f'<img src="{src}" alt="{label}，界面截图" width="1600" height="900" fetchpriority="high"></a>'
                f'<figcaption>启元 Qiyuan Evo · {"演示案例截图 · 示例数据" if team_rows else "原项目界面实录"}</figcaption></figure>')
    entry_url = _experiment_url({"page": "自演进驾驶舱"})
    first_anchor = "collaboration" if scene else "team-gallery" if team_rows else "domain-gallery" if domain_rows else "experiment-entry"
    content = (
        '<main class="qy-home"><section class="qy-hero" aria-labelledby="qy-title">'
        '<div class="qy-eyebrow">启元 Qiyuan Evo</div>'
        '<h1 id="qy-title">让智能协作，<span>在每一次任务中<em>进化。</em></span></h1>'
        '<p><span>自演进多智能体系统。</span><span>让分工、执行与改进，清晰可见。</span></p>'
        f'<div class="qy-actions"><a class="qy-primary" href="{entry_url}" target="_self">进入实验 <span aria-hidden="true">↗</span></a>'
        f'<a class="qy-text-link" href="#{first_anchor}" target="_self">浏览演示 <span aria-hidden="true">›</span></a></div>'
        + hero + '</section>' + scene + proof
    )
    if team_rows:
        content += _gallery_markup(team_rows, GALLERY, anchor="team-gallery", title="团队自演进",
                                   heading="从初轮协作，到下一次进化。", note="一个完整设计案例，八个演进阶段。发现问题、修改策略、重组团队，让每一次变化都有迹可循。", illustrative=True)
    if domain_rows:
        content += _gallery_markup(domain_rows, DOMAIN_GALLERY, anchor="domain-gallery", title="金融政策风洞",
                                   heading="让政策推演，<br>发生在数字世界。", note="政策输入、市场推演、历史验证。把复杂的研究过程，展开来看。", illustrative=False)
    content += (
        '<section class="qy-finale" id="experiment-entry"><h2>下一次进化，从这里开始。</h2>'
        '<p>打开已完成案例，探索团队如何协作、审查与改进。</p>'
        f'<a class="qy-primary" href="{entry_url}" target="_self">探索实验总览 <span aria-hidden="true">↗</span></a></section>'
        '<footer class="qy-footer"><span>启元 Qiyuan Evo</span><span>基于 openJiuwen 的自演进多智能体系统</span>'
        f'<a href="{_experiment_url({"page": "技术与复现"})}" target="_self">技术与复现 ↗</a></footer>'
        '<dialog class="qy-lightbox" aria-labelledby="qy-lightbox-title"><div class="qy-lightbox-top">'
        '<h2 id="qy-lightbox-title">项目界面实录</h2><div><button type="button" data-zoom aria-pressed="false">查看细节</button>'
        '<button type="button" data-close autofocus>关闭</button></div></div>'
        '<img alt=""></dialog></main>'
    )
    script = (ROOT / "static" / "presentation.js").read_text(encoding="utf-8")
    st.html(content + '<script>' + script + '</script>', unsafe_allow_javascript=True)
    for warning in warnings:
        st.warning(warning)
    if not team_rows and not domain_rows and not warnings:
        st.info("暂无项目截图，可从顶部导航查看实验。")
    with st.expander("图片来源"):
        st.write("团队自演进图集来自独立设计案例 DEMO-LIQ-01，全部在新版前端截图。案例数据为示例；每张图可跳转至对应阶段，查看完整流程。")
        guide = ROOT / "docs" / "自演进演示案例.md"
        if guide.is_file():
            st.download_button("下载完整案例讲解稿", guide.read_bytes(), "自演进演示案例.md", "text/markdown")
        st.markdown("金融政策风洞截图来自 [Civitas-Economica-Demo 原仓库]"
                    "(https://github.com/DengJunxian/Civitas-Economica-Demo/tree/"
                    "585ffcbaaeb292f0addc10331ae7dae0c57ebd84/static/showcase_gallery)。")
        for directory, label in ((GALLERY, "下载演示案例截图清单"), (DOMAIN_GALLERY, "下载原项目截图清单")):
            path = directory / "manifest.json"
            if path.is_file():
                st.download_button(label, path.read_bytes(), directory.name + "_manifest.json", "application/json")
