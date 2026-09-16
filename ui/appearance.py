"""Shared appearance for the native interface, SVG diagrams and Plotly charts."""
import streamlit as st


def is_dark():
    return st.session_state.get("ui_theme", "dark") == "dark"


def diagram_colors(svg):
    if is_dark():
        replacements = {"#101c2d": "#182439", "#2a4058": "#435570", "#103039": "#23364e", "#64c8c5": "#a8caff"}
    else:
        replacements = {
            "#101c2d": "#f5f5f7", "#2a4058": "#d2d2d7", "#103039": "#eaf3ff", "#64c8c5": "#0071e3",
            "#eef4ff": "#1d1d1f", "#a7b8cc": "#6e6e73", "#a3b3c7": "#6e6e73", "#96e3d8": "#0071e3",
            "#506984": "#939399", "#829cb5": "#6e6e73", "#67c7b3": "#277553", "#43c5d0": "#0071e3",
            "#87aef5": "#3468ad", "#ed8b82": "#b33336", "#dcb473": "#8c5a12",
        }
    for old, new in replacements.items():
        svg = svg.replace(old, new)
    return svg
