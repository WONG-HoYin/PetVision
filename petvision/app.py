import streamlit as st
import pandas as pd
import numpy as np
import json, os, asyncio, base64
from io import BytesIO
from PIL import Image
from language import LANG_DICT
import logic
from llm_utils import service as llm_service
from llm_utils.models import StructuredHumanizedOutput 


st.set_page_config(page_title="PetVision", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = os.path.join(BASE_DIR, "20260328_best.h5")
MAP_PATH = os.path.join(BASE_DIR, "category_map.json")
TRANS_PATH = os.path.join(BASE_DIR, "label_translations.json")
ATTR_PATH = os.path.join(BASE_DIR, "attributes.csv")
CSS_PATH = os.path.join(BASE_DIR, "style.css")
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


for k in ['lang', 'history', 'analysis_results', 'active_mode', 'last_file_id', 'deep_report_data', 'chat_history']:
    if k not in st.session_state: 
        if k == 'lang': st.session_state[k] = "ZH"
        elif k in ['history', 'chat_history']: st.session_state[k] = []
        else: st.session_state[k] = None

L = LANG_DICT[st.session_state.lang]


if os.path.exists(CSS_PATH):
    with open(CSS_PATH, encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def apply_logic_guardrails(raw_res, standards):

    corrected = raw_res.copy()
    breed_en_raw = raw_res['breed_label'][0]
    lookup_key = breed_en_raw.lower().replace('_', ' ').replace('-', ' ').strip()
    
    species_en = raw_res['species_label'][0]
    breed_conf = raw_res['breed_label'][1]
    

    std_info = standards.get(lookup_key)
    warning_msg = None

    if std_info:
        if breed_conf > 80.0 and species_en.lower() != std_info['Species'].lower():
            old_species = species_en
            new_species = std_info['Species']
            corrected['species_label'] = (new_species, breed_conf)
            warning_msg = f"Logic Guardrail Active: Adjusted Species from {old_species} to {new_species} based on Breed {breed_en_raw}"
    
    return corrected, std_info, warning_msg


@st.dialog(L["dialog_title"])
def chat_with_expert_dialog():
    if not st.session_state.get("deep_report_data"):
        st.error("找不到分析數據")
        return

    rep = st.session_state.deep_report_data
    

    if prompt := st.chat_input(placeholder=""): 
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(L["dialog_thinking"]):

                response = llm_service.get_llm_chat_response(
                    prompt, 
                    rep["cnn_input"], 
                    st.session_state.chat_history, 
                    rep["img_b64"], 
                    st.session_state.lang,
                    rep["report_text"]
                )
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
        

@st.cache_resource
def load_data():
    model = logic.load_pet_model(MODEL_PATH)
    with open(MAP_PATH, 'r', encoding='utf-8') as f: 
        m = json.load(f)
    
    reverse_m = {}
    for task in ['species_label', 'breed_label', 'size_label', 'coat_label']:
        reverse_m[task] = {v: k for k, v in m[task].items()}
    
    with open(TRANS_PATH, 'r', encoding='utf-8') as f: 
        t = json.load(f)
    attr_df = pd.read_csv(ATTR_PATH)
    attr_df['Lookup_Key'] = attr_df['Breed'].str.lower().str.replace('_',' ').str.replace('-',' ').str.strip()
    return model, reverse_m, t, attr_df.set_index('Lookup_Key').to_dict('index')

model, reverse_mappings, trans, attr_standards = load_data()

with st.sidebar:
    st.header(" Language / 語言")
    lang_choice = st.selectbox("Select Language", ["繁體中文", "English"], index=0 if st.session_state.lang == "ZH" else 1)
    st.session_state.lang = "ZH" if lang_choice == "繁體中文" else "EN"
    L = LANG_DICT[st.session_state.lang]

    st.divider()
    st.header(L["upload_header"])
    uploaded_file = st.file_uploader(L["upload_label"], type=["jpg", "png", "jpeg"])
    
    st.subheader(L["history_header"])
    if not st.session_state.history: st.caption(L["history_empty"])
    else:
        for h in reversed(st.session_state.history[-10:]):
            st.markdown(f'<div class="history-item">{h}</div>', unsafe_allow_html=True)

    if st.button(L["reset_btn"], use_container_width=True):
        st.session_state.analysis_results = None; st.session_state.last_file_id = None; st.rerun()




current_upload = uploaded_file or st.session_state.get("main_page_uploader")
if current_upload:
    st.session_state["locked_file"] = current_upload

final_file = st.session_state.get("locked_file")


if not final_file:
    _, center_col, _ = st.columns([1, 4, 1])
    with center_col:
        st.markdown(f"""
            <div class="main-container">
                <div class="main-icon"></div>
                <h1 class="main-title">{L['title']}</h1>
                <p class="main-subtitle">{L['info']}</p>
                <div class="feature-grid">
                    <div class="feature-item">{L['feat_cnn']}</div>
                    <div class="feature-item">{L['feat_grad']}</div>
                    <div class="feature-item">{L['feat_llm']}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.file_uploader(
            L["upload_label"], 
            type=["jpg", "png", "jpeg"], 
            key="main_page_uploader",
            label_visibility="collapsed"
        )
else:

    st.title(L["title"])
    curr_id = f"{final_file.name}_{final_file.size}"
    if st.session_state.last_file_id != curr_id:
        st.session_state.analysis_results = None
        st.session_state.last_file_id = curr_id
        st.session_state.chat_history = []
        st.session_state.deep_report_data = None

    img = Image.open(final_file).convert("RGB")
    col_viz, col_ctrl = st.columns([1.6, 1.4])
    

    with col_viz:
        st.subheader(L["viz_header"])
        if st.session_state.analysis_results is None:
            st.image(img, use_container_width=True)
        else:
            t1, t2 = st.tabs([f" {L['tab_orig']}", f" {L['tab_grad']}"])
            with t1:
                st.image(img, use_container_width=True)
            with t2:
                st.image(st.session_state.analysis_results["heatmap"], use_container_width=True)

    with col_ctrl:
        st.subheader(L["ctrl_header"])
        if st.button(L["analyze_btn"], type="primary", use_container_width=True):
            with st.spinner(L["llm_status"]):
                img_input = logic.process_image(img)
                preds = model.predict(img_input, verbose=0)
                
                res = {}
                keys = ['species_label', 'breed_label', 'size_label', 'coat_label']
                for i, k in enumerate(keys):
                    idx = np.argmax(preds[i])
                    res[k] = (reverse_mappings[k][idx], np.max(preds[i]) * 100)
                
                res, std_info, warning_msg = apply_logic_guardrails(res, attr_standards)
                if warning_msg: st.toast(warning_msg, icon="")

                h_map = logic.get_gradcam_heatmap(model, img_input, "out_relu", np.argmax(preds[1]))
                final_h = logic.overlay_heatmap(np.array(img), h_map)
                
                st.session_state.analysis_results = {
                    "res": res, "breed": res['breed_label'][0], "heatmap": final_h
                }
                
                h_name = trans['breed'].get(res['breed_label'][0], res['breed_label'][0]) if st.session_state.lang == "ZH" else res['breed_label'][0]
                if h_name not in st.session_state.history:
                    st.session_state.history.append(h_name)
                
                st.rerun()

        if st.session_state.analysis_results:
            d = st.session_state.analysis_results
            res = d["res"]
            is_zh = (st.session_state.lang == "ZH")

            st.success(L["res_success"])
            c1, c2 = st.columns(2)

            sp_disp = trans['species'].get(res['species_label'][0], res['species_label'][0]) if is_zh else res['species_label'][0]
            with c1:
                st.markdown(f'<div class="info-card"><div class="card-label">{L["metric_species"]}</div><div class="card-value" style="color: #3B82F6;">{sp_disp}</div><div class="card-conf">↑ {res["species_label"][1]:.1f}%</div></div>', unsafe_allow_html=True)

            br_disp = trans['breed'].get(d['breed'], d['breed']) if is_zh else d['breed']
            with c2:
                st.markdown(f'<div class="info-card"><div class="card-label">{L["metric_breed"]}</div><div class="card-value" style="color: #3B82F6;">{br_disp}</div><div class="card-conf">↑ {res["breed_label"][1]:.1f}%</div></div>', unsafe_allow_html=True)
            
            st.divider()
            st.markdown(f"### {L['attr_header']}")
            lk = d["breed"].lower().replace('_',' ').strip(); std = attr_standards.get(lk)
            if std:
                p_sz = trans['size'].get(res['size_label'][0], res['size_label'][0]) if is_zh else res['size_label'][0]
                s_sz = trans['size'].get(std['Size'], std['Size']) if is_zh else std['Size']
                p_ct = trans['coat'].get(res['coat_label'][0], res['coat_label'][0]) if is_zh else res['coat_label'][0]
                s_ct = trans['coat'].get(std['Coat'], std['Coat']) if is_zh else std['Coat']
                comp_df = pd.DataFrame({L["attr_col_feat"]: [L["attr_size"], L["attr_coat"]], L["attr_col_pred"]: [p_sz, p_ct], L["attr_col_std"]: [s_sz, s_ct]})
                def highlight(row):
                    color = 'background-color: rgba(34, 197, 94, 0.2);' if row[L["attr_col_pred"]] == row[L["attr_col_std"]] else 'background-color: rgba(239, 68, 68, 0.2);'
                    return [color] * len(row)
                st.dataframe(comp_df.style.apply(highlight, axis=1), hide_index=True, use_container_width=True)

            st.divider()
            
            if "deep_report_data" not in st.session_state:
                st.session_state.deep_report_data = None

            
            if st.button(L["report_btn"], use_container_width=True):
                with st.spinner("AI 正在觀察影像細節..."):
                    process_img = img.copy().resize((320, 320))
                    buf = BytesIO(); process_img.save(buf, format="JPEG", quality=75)
                    img_b64 = base64.b64encode(buf.getvalue()).decode()

                    cnn_input_data = {
                        "cnn_guess": d["breed"],
                        "cnn_guess_zh": trans['breed'].get(d["breed"], d["breed"]),
                        "species": res['species_label'][0],
                        "breed_conf": res['breed_label'][1],
                        "species_conf": res['species_label'][1]
                    }

                    analysis_result = llm_service.get_llm_structured_analysis(
                    cnn_input_data, 
                    image_b64=img_b64, 
                    language=st.session_state.lang,
                    confidence=res['breed_label'][1],
                    species_match=(res['species_label'][0] != "Other")
                    )
                    
                    if analysis_result:
                        st.session_state.deep_report_data = {
                            "report_text": llm_service.format_analysis_for_display(analysis_result, language=st.session_state.lang),
                            "case_type": analysis_result.get("case_type"),
                            "img_b64": img_b64,
                            "cnn_input": cnn_input_data
                        }
                    else:
                        st.error("AI 分析暫時失效")

            if st.session_state.deep_report_data:
                rep = st.session_state.deep_report_data
                
                if rep["case_type"] == "special_case":
                    st.warning(L["report_status_warn"])
                else:
                    st.success(L["report_status_success"])
                
                with st.container(border=True):
                    st.markdown(rep["report_text"])
                
                st.divider()
                
                if st.button(L["ask_expert_btn"], use_container_width=True):
                    chat_with_expert_dialog()