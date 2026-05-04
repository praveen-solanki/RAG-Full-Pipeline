# from __future__ import annotations

# import base64
# import sys
# from pathlib import Path
# from typing import Any, Dict, Optional, List

# import pandas as pd
# import streamlit as st
# import altair as alt

# # ✅ ensure repo root is on PYTHONPATH
# ROOT = Path(__file__).resolve().parents[1]  # .../glm-ocr
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

# from eval_backend import evaluate_overlap, safe_load_json_bytes  # noqa: E402


# # --------------------------- Theme / Palette ---------------------------

# PRIMARY = "#96C2DB"   # blue-grey
# BG = "#E5EDF1"        # light blue-grey
# WHITE = "#FFFFFF"
# TEXT = "#0B1220"
# SIDEBAR_BG = "#0F172A"

# GLM_COLOR = "#2563EB"  # blue
# GPT_COLOR = "#14B8A6"  # teal

# st.set_page_config(page_title="OCR Quality Check", layout="wide")

# st.markdown(
#     f"""
# <style>
# html, body, [data-testid="stAppViewContainer"] {{
#     background: {BG} !important;
#     color: {TEXT} !important;
# }}

# .block-container {{
#     padding-top: 3.25rem;
#     padding-bottom: 2rem;
# }}

# [data-testid="stSidebar"] {{
#     background: {SIDEBAR_BG} !important;
# }}
# [data-testid="stSidebar"] * {{
#     color: #E5E7EB !important;
# }}

# div.stButton > button {{
#     background: {PRIMARY} !important;
#     color: {TEXT} !important;
#     border: 1px solid rgba(0,0,0,0.12) !important;
#     border-radius: 12px !important;
#     padding: 0.72rem 1.0rem !important;
#     font-weight: 700 !important;
# }}
# div.stButton > button:hover {{
#     filter: brightness(0.98);
# }}

# .card {{
#     background: {WHITE};
#     border: 1px solid rgba(15,23,42,0.10);
#     border-radius: 16px;
#     padding: 16px 16px 10px 16px;
#     box-shadow: 0 8px 18px rgba(15,23,42,0.06);
# }}

# .card-title {{
#     font-size: 1.15rem;
#     font-weight: 800;
#     margin: 0 0 10px 0;
# }}

# .small-muted {{
#     color: rgba(11,18,32,0.70);
#     font-size: 0.95rem;
#     margin-top: -6px;
#     margin-bottom: 10px;
# }}

# table.tbl {{
#     width: 100%;
#     border-collapse: collapse;
#     overflow: hidden;
#     border-radius: 12px;
# }}
# table.tbl thead th {{
#     background: {PRIMARY};
#     color: {TEXT};
#     text-align: left;
#     padding: 10px 12px;
#     font-weight: 800;
#     border-bottom: 1px solid rgba(15,23,42,0.12);
# }}
# table.tbl tbody td {{
#     padding: 10px 12px;
#     border-bottom: 1px solid rgba(15,23,42,0.08);
# }}
# table.tbl tbody tr:nth-child(even) td {{
#     background: #F7FAFC;
# }}
# .badge {{
#     display:inline-block;
#     padding: 4px 10px;
#     border-radius: 999px;
#     font-weight: 700;
#     font-size: 0.85rem;
#     background: rgba(150,194,219,0.30);
#     border: 1px solid rgba(15,23,42,0.10);
# }}
# </style>
# """,
#     unsafe_allow_html=True,
# )


# # --------------------------- Logo (no cut) ---------------------------

# def _load_logo_bytes() -> Optional[bytes]:
#     candidates = [
#         ROOT / "frontend" / "assets" / "bosch.png",
#         Path("/mnt/data/bosch.png"),
#     ]
#     for p in candidates:
#         try:
#             if p.exists():
#                 return p.read_bytes()
#         except Exception:
#             pass
#     return None


# logo = _load_logo_bytes()
# if logo:
#     b64 = base64.b64encode(logo).decode("ascii")
#     st.markdown(
#         f"""
# <div style="margin: 0 0 10px 0;">
#   <img src="data:image/png;base64,{b64}" style="height:46px;width:auto;display:block;">
# </div>
# """,
#         unsafe_allow_html=True,
#     )

# st.title("OCR Quality Check (Overlap Recall)")
# st.markdown(
#     '<div class="small-muted">Upload GT + outputs → Evaluate individually → Compare at the bottom.</div>',
#     unsafe_allow_html=True
# )


# # --------------------------- Helpers ---------------------------

# def _put_in_state(key: str, uploaded_file) -> None:
#     if uploaded_file is None:
#         return
#     data = safe_load_json_bytes(uploaded_file.getvalue())
#     st.session_state[key] = data
#     st.session_state[f"{key}_name"] = uploaded_file.name


# def _get_display_scopes(report: Dict[str, Any]) -> List[str]:
#     """
#     If report contains numeric pages -> show those.
#     Else if only DOC -> show DOC row.
#     """
#     bd = report.get("breakdown_by_page", {}) or {}
#     numeric = []
#     has_doc = False
#     for k in bd.keys():
#         if k == "DOC":
#             has_doc = True
#         elif isinstance(k, str) and k.isdigit():
#             numeric.append(int(k))

#     if numeric:
#         return [str(x) for x in sorted(numeric)]
#     if has_doc or bd:
#         return ["DOC"]
#     return []


# def _scope_title(scopes: List[str]) -> str:
#     if not scopes:
#         return ""
#     if scopes == ["DOC"]:
#         return "(DOC)"
#     # numeric pages
#     try:
#         nums = [int(x) for x in scopes if x.isdigit()]
#         if nums and nums == list(range(min(nums), max(nums) + 1)):
#             return f"(Pages {min(nums)}–{max(nums)})"
#         return f"(Pages {len(nums)})"
#     except Exception:
#         return ""


# def _scope_table(report: Dict[str, Any]) -> pd.DataFrame:
#     bd = report.get("breakdown_by_page", {}) or {}
#     scopes = _get_display_scopes(report)

#     rows = []
#     for s in scopes:
#         v = bd.get(s, {"total": 0, "hit": 0, "recall": 0.0})
#         rows.append(
#             {
#                 "Page": s,  # "1".."N" or "DOC"
#                 "Fields": int(v.get("total", 0)),
#                 "Matched": int(v.get("hit", 0)),
#                 "Recall": float(v.get("recall", 0.0)),
#             }
#         )
#     df = pd.DataFrame(rows)
#     if not df.empty:
#         df["Recall"] = df["Recall"].map(lambda x: f"{x:.4f}")
#     return df


# def _render_table_card(title: str, report: Optional[Dict[str, Any]]) -> None:
#     if report is None:
#         html = f"""
# <div class="card">
#   <div class="card-title">{title}</div>
#   <div class="small-muted">No results yet. Click the evaluate button above.</div>
# </div>
# """
#         st.markdown(html, unsafe_allow_html=True)
#         return

#     df = _scope_table(report)
#     scopes = _get_display_scopes(report)
#     title2 = f"{title} {_scope_title(scopes)}".strip()

#     table_html = df.to_html(index=False, escape=True, classes="tbl")
#     html = f"""
# <div class="card">
#   <div class="card-title">{title2}</div>
#   {table_html}
# </div>
# """
#     st.markdown(html, unsafe_allow_html=True)


# def _compare_chart(glm_report: Dict[str, Any], gpt_report: Dict[str, Any]):
#     def scopes_union(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
#         ka = set((a.get("breakdown_by_page", {}) or {}).keys())
#         kb = set((b.get("breakdown_by_page", {}) or {}).keys())
#         keys = list(ka.union(kb))

#         nums = sorted([int(x) for x in keys if isinstance(x, str) and x.isdigit()])
#         out = [str(x) for x in nums]
#         if "DOC" in keys and not out:
#             out = ["DOC"]
#         elif "DOC" in keys and out:
#             # keep DOC last if present alongside pages
#             out.append("DOC")
#         return out

#     scopes = scopes_union(glm_report, gpt_report)

#     def df_for(model: str, report: Dict[str, Any]) -> pd.DataFrame:
#         bd = report.get("breakdown_by_page", {}) or {}
#         rows = []
#         for s in scopes:
#             v = bd.get(s, {"recall": 0.0})
#             rows.append({"Page": s, "Recall": float(v.get("recall", 0.0)), "Model": model})
#         return pd.DataFrame(rows)

#     long_df = pd.concat(
#         [df_for("GLM-OCR", glm_report), df_for("GPT", gpt_report)],
#         ignore_index=True
#     )

#     chart = (
#         alt.Chart(long_df)
#         .mark_bar(size=22)
#         .encode(
#             x=alt.X("Page:N", title="Page / Scope"),
#             y=alt.Y("Recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
#             xOffset="Model:N",
#             color=alt.Color(
#                 "Model:N",
#                 scale=alt.Scale(domain=["GLM-OCR", "GPT"], range=[GLM_COLOR, GPT_COLOR]),
#                 legend=alt.Legend(title=None, orient="top"),
#             ),
#             tooltip=["Model", "Page", alt.Tooltip("Recall:Q", format=".4f")],
#         )
#         .properties(height=320)
#         .configure_view(strokeOpacity=0)
#         .configure_axis(labelColor=TEXT, titleColor=TEXT)
#     )
#     return chart


# # --------------------------- Sidebar ---------------------------

# st.sidebar.title("Inputs")

# gt_up = st.sidebar.file_uploader("Upload Ground Truth JSON", type=["json"], key="u_gt")
# glm_up = st.sidebar.file_uploader("Upload GLM-OCR merged JSON", type=["json"], key="u_glm")
# gpt_up = st.sidebar.file_uploader("Upload GPT output (JSON or TXT)", type=["json", "txt"], key="u_gpt")

# _put_in_state("gt_json", gt_up)
# _put_in_state("glm_json", glm_up)
# _put_in_state("gpt_json", gpt_up)

# st.sidebar.divider()
# st.sidebar.write("**Loaded**")
# st.sidebar.write(f"- GT: `{st.session_state.get('gt_json_name', 'not loaded')}`")
# st.sidebar.write(f"- GLM: `{st.session_state.get('glm_json_name', 'not loaded')}`")
# st.sidebar.write(f"- GPT: `{st.session_state.get('gpt_json_name', 'not loaded')}`")

# if st.sidebar.button("Clear"):
#     for k in [
#         "gt_json", "glm_json", "gpt_json",
#         "gt_json_name", "glm_json_name", "gpt_json_name",
#         "glm_report", "gpt_report", "show_compare"
#     ]:
#         st.session_state.pop(k, None)
#     st.rerun()


# # --------------------------- Actions ---------------------------

# gt = st.session_state.get("gt_json")
# glm = st.session_state.get("glm_json")
# gpt = st.session_state.get("gpt_json")

# b1, b2, b3 = st.columns([1, 1, 1])
# with b1:
#     eval_glm = st.button("Evaluate GLM-OCR", use_container_width=True)
# with b2:
#     eval_gpt = st.button("Evaluate GPT", use_container_width=True)
# with b3:
#     st.markdown("")

# if eval_glm:
#     if gt is None or glm is None:
#         st.error("Upload Ground Truth + GLM merged JSON first.")
#     else:
#         with st.spinner("Evaluating GLM-OCR..."):
#             st.session_state["glm_report"] = evaluate_overlap(gt, glm)

# if eval_gpt:
#     if gt is None or gpt is None:
#         st.error("Upload Ground Truth + GPT output first.")
#     else:
#         with st.spinner("Evaluating GPT..."):
#             st.session_state["gpt_report"] = evaluate_overlap(gt, gpt)


# # --------------------------- Results tables ---------------------------

# colL, colR = st.columns(2)
# with colL:
#     _render_table_card("GLM-OCR", st.session_state.get("glm_report"))
# with colR:
#     _render_table_card("GPT", st.session_state.get("gpt_report"))


# # --------------------------- Compare (only after both tables exist) ---------------------------

# glm_report = st.session_state.get("glm_report")
# gpt_report = st.session_state.get("gpt_report")

# st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

# if glm_report and gpt_report:
#     c1, c2, c3 = st.columns([2, 1, 2])
#     with c2:
#         do_compare = st.button("Compare", use_container_width=True)
#     if do_compare:
#         st.session_state["show_compare"] = True

#     if st.session_state.get("show_compare"):
#         st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
#         st.markdown('<div class="card"><div class="card-title">Comparison</div>', unsafe_allow_html=True)
#         st.altair_chart(_compare_chart(glm_report, gpt_report), use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
# else:
#     st.markdown(
#         "<div class='small-muted'><span class='badge'>Compare</span> will appear after you have evaluated both GLM-OCR and GPT.</div>",
#         unsafe_allow_html=True,
#     )

#########################################################################################################################################################################

# from __future__ import annotations

# import base64
# import sys
# from pathlib import Path
# from typing import Any, Dict, Optional, List

# import pandas as pd
# import streamlit as st

# # ✅ ensure repo root is on PYTHONPATH
# ROOT = Path(__file__).resolve().parents[1]  # .../glm-ocr
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

# from eval_backend import (  # noqa: E402
#     evaluate_overlap_from_leaves,
#     safe_load_excel_bytes,
#     safe_load_json_bytes,
#     safe_load_text_bytes,
#     parse_gt,
#     parse_pred,
#     build_alignment_xlsx_bytes,
# )

# # --------------------------- Theme / Palette ---------------------------

# PRIMARY = "#96C2DB"   # blue-grey
# BG = "#E5EDF1"        # light blue-grey
# WHITE = "#FFFFFF"
# TEXT = "#0B1220"
# SIDEBAR_BG = "#0F172A"

# st.set_page_config(page_title="OCR Quality Check", layout="wide")

# st.markdown(
#     f"""
# <style>
# html, body, [data-testid="stAppViewContainer"] {{
#     background: {BG} !important;
#     color: {TEXT} !important;
# }}

# .block-container {{
#     padding-top: 3.25rem;
#     padding-bottom: 2rem;
# }}

# [data-testid="stSidebar"] {{
#     background: {SIDEBAR_BG} !important;
# }}
# [data-testid="stSidebar"] * {{
#     color: #E5E7EB !important;
# }}

# div.stButton > button {{
#     background: {PRIMARY} !important;
#     color: {TEXT} !important;
#     border: 1px solid rgba(0,0,0,0.12) !important;
#     border-radius: 12px !important;
#     padding: 0.72rem 1.0rem !important;
#     font-weight: 700 !important;
# }}
# div.stButton > button:hover {{
#     filter: brightness(0.98);
# }}

# .card {{
#     background: {WHITE};
#     border: 1px solid rgba(15,23,42,0.10);
#     border-radius: 16px;
#     padding: 16px 16px 10px 16px;
#     box-shadow: 0 8px 18px rgba(15,23,42,0.06);
# }}

# .card-title {{
#     font-size: 1.15rem;
#     font-weight: 800;
#     margin: 0 0 10px 0;
# }}

# .small-muted {{
#     color: rgba(11,18,32,0.70);
#     font-size: 0.95rem;
#     margin-top: -6px;
#     margin-bottom: 10px;
# }}

# table.tbl {{
#     width: 100%;
#     border-collapse: collapse;
#     overflow: hidden;
#     border-radius: 12px;
# }}
# table.tbl thead th {{
#     background: {PRIMARY};
#     color: {TEXT};
#     text-align: left;
#     padding: 10px 12px;
#     font-weight: 800;
#     border-bottom: 1px solid rgba(15,23,42,0.12);
# }}
# table.tbl tbody td {{
#     padding: 10px 12px;
#     border-bottom: 1px solid rgba(15,23,42,0.08);
# }}
# table.tbl tbody tr:nth-child(even) td {{
#     background: #F7FAFC;
# }}
# .badge {{
#     display:inline-block;
#     padding: 4px 10px;
#     border-radius: 999px;
#     font-weight: 700;
#     font-size: 0.85rem;
#     background: rgba(150,194,219,0.30);
#     border: 1px solid rgba(15,23,42,0.10);
# }}
# </style>
# """,
#     unsafe_allow_html=True,
# )

# # --------------------------- Logo (optional) ---------------------------

# def _load_logo_bytes() -> Optional[bytes]:
#     candidates = [
#         ROOT / "frontend" / "assets" / "bosch.png",
#         Path("/mnt/data/bosch.png"),
#     ]
#     for p in candidates:
#         try:
#             if p.exists():
#                 return p.read_bytes()
#         except Exception:
#             pass
#     return None

# logo = _load_logo_bytes()
# if logo:
#     b64 = base64.b64encode(logo).decode("ascii")
#     st.markdown(
#         f"""
# <div style="margin: 0 0 10px 0;">
#   <img src="data:image/png;base64,{b64}" style="height:46px;width:auto;display:block;">
# </div>
# """,
#         unsafe_allow_html=True,
#     )

# st.title("OCR Quality Check (Overlap Recall + GT↔Pred Alignment)")
# st.markdown(
#     '<div class="small-muted">Select doc type + upload GT + prediction output → Evaluate overlap → Download color-coded alignment Excel.</div>',
#     unsafe_allow_html=True
# )

# # --------------------------- Helpers ---------------------------

# def _put_in_state_any(key: str, uploaded_file, fmt: str) -> None:
#     if uploaded_file is None:
#         return
#     b = uploaded_file.getvalue()

#     if fmt == "json":
#         data = safe_load_json_bytes(b)
#     elif fmt == "excel":
#         data = safe_load_excel_bytes(b)
#     elif fmt == "txt":
#         data = safe_load_text_bytes(b)
#     else:
#         data = safe_load_json_bytes(b)

#     st.session_state[key] = data
#     st.session_state[f"{key}_name"] = uploaded_file.name


# def _scope_table(report: Dict[str, Any]) -> pd.DataFrame:
#     bd = report.get("breakdown_by_page", {}) or {}
#     rows = []
#     for k, v in bd.items():
#         rows.append(
#             {
#                 "Scope": k,  # "1".."N" or "DOC"
#                 "Fields": int(v.get("total", 0)),
#                 "Matched": int(v.get("hit", 0)),
#                 "Recall": float(v.get("recall", 0.0)),
#             }
#         )
#     df = pd.DataFrame(rows)
#     if not df.empty:
#         # stable ordering: numeric pages then DOC
#         def _k(x):
#             s = str(x)
#             if s.isdigit():
#                 return (0, int(s))
#             if s == "DOC":
#                 return (2, 10**9)
#             return (1, s)
#         df = df.sort_values(by="Scope", key=lambda col: col.map(_k))
#         df["Recall"] = df["Recall"].map(lambda x: f"{x:.4f}")
#     return df


# def _render_report_card(title: str, report: Optional[Dict[str, Any]]) -> None:
#     if report is None:
#         st.markdown(
#             f"""
# <div class="card">
#   <div class="card-title">{title}</div>
#   <div class="small-muted">No results yet. Click <b>Evaluate Overlap Recall</b>.</div>
# </div>
# """,
#             unsafe_allow_html=True,
#         )
#         return

#     df = _scope_table(report)
#     summary = f"Total: {report.get('total_fields_checked', 0)} | Matched: {report.get('matched_fields', 0)} | Recall: {report.get('recall', 0.0)}"
#     table_html = df.to_html(index=False, escape=True, classes="tbl")
#     st.markdown(
#         f"""
# <div class="card">
#   <div class="card-title">{title}</div>
#   <div class="small-muted">{summary}</div>
#   {table_html}
# </div>
# """,
#         unsafe_allow_html=True,
#     )

# # --------------------------- Sidebar ---------------------------

# st.sidebar.title("Inputs")

# doc_type = st.sidebar.selectbox("Document Type", ["shipping_bill", "purchase_order"])

# gt_fmt = st.sidebar.selectbox("GT Format", ["json", "excel"])
# pred_fmt = st.sidebar.selectbox("Pred Format", ["json", "excel", "txt"])

# st.sidebar.divider()

# gt_up = st.sidebar.file_uploader(
#     "Upload Ground Truth (JSON/Excel)",
#     type=["json", "xlsx"],
#     key="u_gt",
# )
# pred_up = st.sidebar.file_uploader(
#     "Upload Prediction Output (JSON/Excel/TXT)",
#     type=["json", "xlsx", "txt"],
#     key="u_pred",
# )

# _put_in_state_any("gt_obj", gt_up, gt_fmt)
# _put_in_state_any("pred_obj", pred_up, pred_fmt)

# st.sidebar.divider()
# st.sidebar.write("**Loaded**")
# st.sidebar.write(f"- Doc Type: `{doc_type}`")
# st.sidebar.write(f"- GT: `{st.session_state.get('gt_obj_name', 'not loaded')}` ({gt_fmt})")
# st.sidebar.write(f"- Pred: `{st.session_state.get('pred_obj_name', 'not loaded')}` ({pred_fmt})")

# if st.sidebar.button("Clear"):
#     for k in [
#         "gt_obj", "pred_obj",
#         "gt_obj_name", "pred_obj_name",
#         "overlap_report",
#         "align_xlsx",
#         "align_filename",
#     ]:
#         st.session_state.pop(k, None)
#     st.rerun()

# # --------------------------- Actions ---------------------------

# gt_obj = st.session_state.get("gt_obj")
# pred_obj = st.session_state.get("pred_obj")

# b1, b2 = st.columns([1, 1])
# with b1:
#     eval_overlap_btn = st.button("Evaluate Overlap Recall", use_container_width=True)
# with b2:
#     build_align_btn = st.button("Build GT↔Pred Alignment Excel", use_container_width=True)

# if eval_overlap_btn:
#     if gt_obj is None or pred_obj is None:
#         st.error("Upload GT + Pred first.")
#     else:
#         try:
#             gt_leaves = parse_gt(doc_type, gt_fmt, gt_obj)

#             # overlap candidate can be JSON or raw_text or any dict/list; if pred is excel -> convert to big raw text
#             if isinstance(pred_obj, pd.DataFrame):
#                 txt = "\n".join(pred_obj.astype(str).fillna("").values.flatten().tolist())
#                 candidate_for_overlap = {"raw_text": txt}
#             else:
#                 candidate_for_overlap = pred_obj

#             with st.spinner("Evaluating overlap recall..."):
#                 st.session_state["overlap_report"] = evaluate_overlap_from_leaves(gt_leaves, candidate_for_overlap)

#         except Exception as e:
#             st.exception(e)

# if build_align_btn:
#     if gt_obj is None or pred_obj is None:
#         st.error("Upload GT + Pred first.")
#     else:
#         if pred_fmt == "txt":
#             st.error("Alignment Excel requires Pred in JSON or Excel (TXT doesn't have key paths).")
#         else:
#             try:
#                 gt_leaves = parse_gt(doc_type, gt_fmt, gt_obj)
#                 pred_leaves = parse_pred(pred_fmt, pred_obj)

#                 with st.spinner("Building alignment excel..."):
#                     xlsx_bytes = build_alignment_xlsx_bytes(gt_leaves, pred_leaves)

#                 st.session_state["align_xlsx"] = xlsx_bytes
#                 st.session_state["align_filename"] = f"{doc_type}_gt_pred_alignment.xlsx"

#             except Exception as e:
#                 st.exception(e)

# # --------------------------- Results ---------------------------

# st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
# _render_report_card("Overlap Recall", st.session_state.get("overlap_report"))

# st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

# if st.session_state.get("align_xlsx"):
#     st.markdown(
#         """
# <div class="card">
#   <div class="card-title">GT↔Pred Alignment (Color-coded Excel)</div>
#   <div class="small-muted">Download the exact matching visualization: GT key, Pred key, GT value, Pred value + row-level color coding.</div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )

#     st.download_button(
#         "Download Alignment Excel (Color-coded)",
#         data=st.session_state["align_xlsx"],
#         file_name=st.session_state.get("align_filename", "gt_pred_alignment.xlsx"),
#         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#         use_container_width=True,
#     )
# else:
#     st.markdown(
#         "<div class='small-muted'><span class='badge'>Alignment Excel</span> will appear after you click <b>Build GT↔Pred Alignment Excel</b>.</div>",
#         unsafe_allow_html=True,
#     )


# /home/mtq3kor/aman/GLM/glm-ocr/frontend/app.py
from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

# ✅ ensure repo root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]  # .../glm-ocr
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_backend import (  # noqa: E402
    build_alignment_xlsx_bytes,
    build_alignment_3way_xlsx_bytes,
    evaluate_overlap_from_leaves,
    parse_gt,
    parse_pred,
    safe_load_excel_bytes,
    safe_load_json_bytes,
    safe_load_text_bytes,
)

# --------------------------- Theme / Palette ---------------------------

PRIMARY = "#96C2DB"   # blue-grey
BG = "#E5EDF1"        # light blue-grey
WHITE = "#FFFFFF"
TEXT = "#0B1220"
SIDEBAR_BG = "#0F172A"

st.set_page_config(page_title="OCR Quality Check", layout="wide")

st.markdown(
    f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background: {BG} !important;
    color: {TEXT} !important;
}}

.block-container {{
    padding-top: 3.25rem;
    padding-bottom: 2rem;
}}

[data-testid="stSidebar"] {{
    background: {SIDEBAR_BG} !important;
}}
[data-testid="stSidebar"] * {{
    color: #E5E7EB !important;
}}

div.stButton > button {{
    background: {PRIMARY} !important;
    color: {TEXT} !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    border-radius: 12px !important;
    padding: 0.72rem 1.0rem !important;
    font-weight: 700 !important;
}}
div.stButton > button:hover {{
    filter: brightness(0.98);
}}

/* ✅ Download button styling (light red -> dark red on hover) */
div[data-testid="stDownloadButton"] > button {{
    background: #FCA5A5 !important;   /* light red */
    color: {TEXT} !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    border-radius: 12px !important;
    padding: 0.72rem 1.0rem !important;
    font-weight: 800 !important;
}}
div[data-testid="stDownloadButton"] > button:hover {{
    background: #DC2626 !important;   /* dark red */
    color: #FFFFFF !important;
    border-color: rgba(0,0,0,0.18) !important;
    filter: none !important;          /* override global hover brightness */
}}

.card {{
    background: {WHITE};
    border: 1px solid rgba(15,23,42,0.10);
    border-radius: 16px;
    padding: 16px 16px 10px 16px;
    box-shadow: 0 8px 18px rgba(15,23,42,0.06);
}}

.card-title {{
    font-size: 1.15rem;
    font-weight: 800;
    margin: 0 0 10px 0;
}}

.small-muted {{
    color: rgba(11,18,32,0.70);
    font-size: 0.95rem;
    margin-top: -6px;
    margin-bottom: 10px;
}}

table.tbl {{
    width: 100%;
    border-collapse: collapse;
    overflow: hidden;
    border-radius: 12px;
}}
table.tbl thead th {{
    background: {PRIMARY};
    color: {TEXT};
    text-align: left;
    padding: 10px 12px;
    font-weight: 800;
    border-bottom: 1px solid rgba(15,23,42,0.12);
}}
table.tbl tbody td {{
    padding: 10px 12px;
    border-bottom: 1px solid rgba(15,23,42,0.08);
}}
table.tbl tbody tr:nth-child(even) td {{
    background: #F7FAFC;
}}
.badge {{
    display:inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
    background: rgba(150,194,219,0.30);
    border: 1px solid rgba(15,23,42,0.10);
}}
</style>
""",
    unsafe_allow_html=True,
)
# --------------------------- Logo (optional) ---------------------------

def _load_logo_bytes() -> Optional[bytes]:
    candidates = [
        ROOT / "frontend" / "assets" / "bosch.png",
        Path("/mnt/data/bosch.png"),
    ]
    for p in candidates:
        try:
            if p.exists():
                return p.read_bytes()
        except Exception:
            pass
    return None


logo = _load_logo_bytes()
if logo:
    b64 = base64.b64encode(logo).decode("ascii")
    st.markdown(
        f"""
<div style="margin: 0 0 10px 0;">
  <img src="data:image/png;base64,{b64}" style="height:46px;width:auto;display:block;">
</div>
""",
        unsafe_allow_html=True,
    )

st.title("OCR Quality Check (Overlap Recall + GT↔Pred Alignment)")
st.markdown(
    '<div class="small-muted">Select doc type + upload GT + prediction output → Evaluate overlap → Download color-coded alignment Excel.</div>',
    unsafe_allow_html=True
)

# --------------------------- Helpers ---------------------------

def _put_in_state_any(key: str, uploaded_file, fmt: str) -> None:
    if uploaded_file is None:
        return
    b = uploaded_file.getvalue()

    if fmt == "json":
        data = safe_load_json_bytes(b)
    elif fmt == "excel":
        data = safe_load_excel_bytes(b)
    elif fmt == "txt":
        data = safe_load_text_bytes(b)
    else:
        data = safe_load_json_bytes(b)

    st.session_state[key] = data
    st.session_state[f"{key}_name"] = uploaded_file.name


def _scope_table(report: Dict[str, Any]) -> pd.DataFrame:
    bd = report.get("breakdown_by_page", {}) or {}
    rows = []
    for k, v in bd.items():
        rows.append(
            {
                "Scope": k,  # "1".."N" or "DOC"
                "Fields": int(v.get("total", 0)),
                "Matched": int(v.get("hit", 0)),
                "Recall": float(v.get("recall", 0.0)),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        # stable ordering: numeric pages then DOC
        def _k(x):
            s = str(x)
            if s.isdigit():
                return (0, int(s))
            if s == "DOC":
                return (2, 10**9)
            return (1, s)

        df = df.sort_values(by="Scope", key=lambda col: col.map(_k))
        df["Recall"] = df["Recall"].map(lambda x: f"{x:.4f}")
    return df


def _render_report_card(title: str, report: Optional[Dict[str, Any]]) -> None:
    if report is None:
        st.markdown(
            f"""
<div class="card">
  <div class="card-title">{title}</div>
  <div class="small-muted">No results yet. Click <b>Evaluate Overlap Recall</b>.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    df = _scope_table(report)
    summary = f"Total: {report.get('total_fields_checked', 0)} | Matched: {report.get('matched_fields', 0)} | Recall: {report.get('recall', 0.0)}"
    table_html = df.to_html(index=False, escape=True, classes="tbl")
    st.markdown(
        f"""
<div class="card">
  <div class="card-title">{title}</div>
  <div class="small-muted">{summary}</div>
  {table_html}
</div>
""",
        unsafe_allow_html=True,
    )


def _validate_payload(fmt: str, obj: Any) -> Optional[str]:
    if fmt == "excel" and not isinstance(obj, pd.DataFrame):
        return "You selected Excel format but the uploaded file did not parse as Excel."
    if fmt == "txt" and not (isinstance(obj, dict) and isinstance(obj.get("raw_text"), str)):
        return "You selected TXT format but the uploaded file did not parse as TXT."
    if fmt == "json" and not isinstance(obj, (dict, list)):
        # safe_load_json_bytes can return {"raw_text":..., "_parse_error":...} on invalid json
        if isinstance(obj, dict) and "_parse_error" in obj:
            return f"Invalid JSON: {obj.get('_parse_error')}"
    return None


# --------------------------- Sidebar ---------------------------

st.sidebar.title("Inputs")

doc_type = st.sidebar.selectbox("Document Type", ["shipping_bill", "purchase_order"])
gt_fmt = st.sidebar.selectbox("GT Format", ["json", "excel"])
pred_fmt = st.sidebar.selectbox("Pred Format", ["json", "excel", "txt"])

st.sidebar.divider()

gt_up = st.sidebar.file_uploader(
    "Upload Ground Truth (JSON/Excel)",
    type=["json", "xlsx"],
    key="u_gt",
)
pred_up = st.sidebar.file_uploader(
    "Upload Prediction Output (JSON/Excel/TXT)",
    type=["json", "xlsx", "txt"],
    key="u_pred",
)
gpt_up = st.sidebar.file_uploader(
    "Upload GPT Prediction (JSON)",
    type=["json"],
    key="u_gpt",
)
_put_in_state_any("gpt_obj", gpt_up, "json")

_put_in_state_any("gt_obj", gt_up, gt_fmt)
_put_in_state_any("pred_obj", pred_up, pred_fmt)

st.sidebar.divider()
st.sidebar.write("**Loaded**")
st.sidebar.write(f"- Doc Type: `{doc_type}`")
st.sidebar.write(f"- GT: `{st.session_state.get('gt_obj_name', 'not loaded')}` ({gt_fmt})")
st.sidebar.write(f"- Pred: `{st.session_state.get('pred_obj_name', 'not loaded')}` ({pred_fmt})")
st.sidebar.write(f"- GPT: `{st.session_state.get('gpt_obj_name', 'not loaded')}` (json)")

if st.sidebar.button("Clear"):
    for k in [
        "gt_obj", "pred_obj", "gpt_obj",           # ✅ add gpt_obj
        "gt_obj_name", "pred_obj_name", "gpt_obj_name",
        "overlap_report",
        "overlap_report_gpt",                      # ✅ add
        "align_xlsx",
        "align_xlsx_gpt",                          # ✅ add
        "align_xlsx_3way",                         # ✅ add
        "align_filename",
        "align_filename_gpt",
        "align_filename_3way",
    ]:
        st.session_state.pop(k, None)
    st.rerun()

# --------------------------- Actions ---------------------------

gt_obj = st.session_state.get("gt_obj")
pred_obj = st.session_state.get("pred_obj")
gpt_obj = st.session_state.get("gpt_obj")

b1, b2, b3 = st.columns([1, 1, 1])
with b1:
    eval_overlap_btn = st.button("Evaluate Overlap Recall (MyModel)", use_container_width=True)
with b2:
    eval_overlap_gpt_btn = st.button("Evaluate Overlap Recall (GPT)", use_container_width=True)
with b3:
    build_align_btn = st.button("Build GT↔Pred Alignment Excel (MyModel)", use_container_width=True)

b4, b5 = st.columns([1, 1])
with b4:
    build_align_gpt_btn = st.button("Build GT↔Pred Alignment Excel (GPT)", use_container_width=True)
with b5:
    build_align_3way_btn = st.button("Build 3-way Comparison Excel", use_container_width=True)

if eval_overlap_btn:
    if gt_obj is None or pred_obj is None:
        st.error("Upload GT + Pred first.")
    else:
        err = _validate_payload(gt_fmt, gt_obj) or _validate_payload(pred_fmt, pred_obj)
        if err:
            st.error(err)
        else:
            try:
                gt_leaves = parse_gt(doc_type, gt_fmt, gt_obj)

                # overlap candidate can be JSON or raw_text or any dict/list;
                # if pred is excel -> convert to big raw text
                if isinstance(pred_obj, pd.DataFrame):
                    txt = "\n".join(pred_obj.astype(str).fillna("").values.flatten().tolist())
                    candidate_for_overlap = {"raw_text": txt}
                else:
                    candidate_for_overlap = pred_obj

                with st.spinner("Evaluating overlap recall..."):
                    st.session_state["overlap_report"] = evaluate_overlap_from_leaves(gt_leaves, candidate_for_overlap)

            except Exception as e:
                st.exception(e)

if eval_overlap_gpt_btn:
    if gt_obj is None or gpt_obj is None:
        st.error("Upload GT + GPT JSON first.")
    else:
        err = _validate_payload(gt_fmt, gt_obj) or _validate_payload("json", gpt_obj)
        if err:
            st.error(err)
        else:
            try:
                gt_leaves = parse_gt(doc_type, gt_fmt, gt_obj)

                candidate_for_overlap = gpt_obj  # GPT JSON
                with st.spinner("Evaluating GPT overlap recall..."):
                    st.session_state["overlap_report_gpt"] = evaluate_overlap_from_leaves(
                        gt_leaves, candidate_for_overlap
                    )
            except Exception as e:
                st.exception(e)

if build_align_btn:
    if gt_obj is None or pred_obj is None:
        st.error("Upload GT + Pred first.")
    else:
        if pred_fmt == "txt":
            st.error("Alignment Excel requires Pred in JSON or Excel (TXT doesn't have key paths).")
        else:
            err = _validate_payload(gt_fmt, gt_obj) or _validate_payload(pred_fmt, pred_obj)
            if err:
                st.error(err)
            else:
                try:
                    gt_leaves = parse_gt(doc_type, gt_fmt, gt_obj)
                    pred_leaves = parse_pred(pred_fmt, pred_obj)

                    with st.spinner("Building alignment excel..."):
                        xlsx_bytes = build_alignment_xlsx_bytes(gt_leaves, pred_leaves)

                    st.session_state["align_xlsx"] = xlsx_bytes
                    st.session_state["align_filename"] = f"{doc_type}_gt_pred_alignment_mymodel.xlsx"

                except Exception as e:
                    st.exception(e)

if build_align_gpt_btn:
    if gt_obj is None or gpt_obj is None:
        st.error("Upload GT + GPT JSON first.")
    else:
        err = _validate_payload(gt_fmt, gt_obj) or _validate_payload("json", gpt_obj)
        if err:
            st.error(err)
        else:
            try:
                gt_leaves = parse_gt(doc_type, gt_fmt, gt_obj)
                gpt_leaves = parse_pred("json", gpt_obj)

                with st.spinner("Building GPT alignment excel..."):
                    xlsx_bytes = build_alignment_xlsx_bytes(gt_leaves, gpt_leaves)

                st.session_state["align_xlsx_gpt"] = xlsx_bytes
                st.session_state["align_filename_gpt"] = f"{doc_type}_gt_pred_alignment_gpt.xlsx"

            except Exception as e:
                st.exception(e)

if build_align_3way_btn:
    if gt_obj is None or pred_obj is None or gpt_obj is None:
        st.error("Upload GT + MyModel Pred + GPT JSON first.")
    else:
        if pred_fmt == "txt":
            st.error("3-way comparison requires MyModel Pred in JSON or Excel (not TXT).")
        else:
            err = _validate_payload(gt_fmt, gt_obj) or _validate_payload(pred_fmt, pred_obj) or _validate_payload("json", gpt_obj)
            if err:
                st.error(err)
            else:
                try:
                    gt_leaves = parse_gt(doc_type, gt_fmt, gt_obj)
                    my_leaves = parse_pred(pred_fmt, pred_obj)
                    gpt_leaves = parse_pred("json", gpt_obj)

                    with st.spinner("Building 3-way comparison excel..."):
                        xlsx_bytes = build_alignment_3way_xlsx_bytes(gt_leaves, my_leaves, gpt_leaves)

                    st.session_state["align_xlsx_3way"] = xlsx_bytes
                    st.session_state["align_filename_3way"] = f"{doc_type}_gt_mymodel_gpt_comparison.xlsx"

                except Exception as e:
                    st.exception(e)

# --------------------------- Results ---------------------------

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
c1, c2 = st.columns([1, 1])
with c1:
    _render_report_card("Overlap Recall (MyModel)", st.session_state.get("overlap_report"))
with c2:
    _render_report_card("Overlap Recall (GPT)", st.session_state.get("overlap_report_gpt"))

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

# MyModel alignment (this is the original color-coded download)
if st.session_state.get("align_xlsx"):
    st.markdown(
        """
<div class="card">
  <div class="card-title">GT↔Pred Alignment (Color-coded Excel)</div>
  <div class="small-muted">Download the exact matching visualization: GT key, Pred key, GT value, Pred value + row-level color coding.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.download_button(
        "Download Alignment Excel (MyModel - Color-coded)",
        data=st.session_state["align_xlsx"],
        file_name=st.session_state.get("align_filename", "gt_pred_alignment_mymodel.xlsx"),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
else:
    st.markdown(
        "<div class='small-muted'><span class='badge'>Alignment Excel</span> will appear after you click <b>Build GT↔Pred Alignment Excel (MyModel)</b>.</div>",
        unsafe_allow_html=True,
    )

# GPT alignment
if st.session_state.get("align_xlsx_gpt"):
    st.download_button(
        "Download Alignment Excel (GPT - Color-coded)",
        data=st.session_state["align_xlsx_gpt"],
        file_name=st.session_state.get("align_filename_gpt", "gt_pred_alignment_gpt.xlsx"),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# 3-way report
if st.session_state.get("align_xlsx_3way"):
    st.download_button(
        "Download 3-way Comparison Excel (GT vs MyModel vs GPT)",
        data=st.session_state["align_xlsx_3way"],
        file_name=st.session_state.get("align_filename_3way", "gt_mymodel_gpt_comparison.xlsx"),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )