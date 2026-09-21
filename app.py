# ============================================================
# SCIENTIFIC ENGINEERING ENGINE + AGENT IA DYNAMIQUE & SÉMANTIQUE
# Meta-moteur générique d'ingénierie scientifique industrielle
#
# VERSION 4.0 (Agent intelligent, flexible et adaptatif)
# ============================================================

import io
import json
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)

ENGINE_NAME = "Scientific Engineering Engine"
ENGINE_VERSION = "4.0"

st.set_page_config(
    page_title=ENGINE_NAME,
    page_icon="⚙️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title { font-size: 2.5rem; font-weight: 750; margin-bottom: 0.2rem; }
    .subtitle { color: #777; font-size: 1.05rem; margin-bottom: 2rem; }
    .block-ok { padding: 0.7rem; border-radius: 0.5rem; background: #eaf7ea; border: 1px solid #9bd39b; color: black; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_STATE = {"analysis": None, "df": None, "confirmations": {}}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def normalize_name(value):
    value = str(value).strip().lower()
    replacements = {"é": "e", "è": "e", "ê": "e", "à": "a", "ç": "c"}
    for old, new in replacements.items():
        value = value.replace(old, new)
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


SYNONYMS_DICTIONARY = {
    "equipment": ["equipement", "machine", "materiel", "actif", "asset", "appareil", "id", "identifiant"],
    "failure": ["defaillance", "panne", "casse", "defaut", "incident", "anomalie", "rupture", "event", "target"],
    "exposure": ["exposition", "kilometrage", "km", "heure", "heures", "compteur", "cycles", "distance", "temps", "duree"],
    "date": ["date", "horodatage", "timestamp", "moment", "periode"],
    "segmentation": ["organe", "cause", "conditions", "intervention", "type", "atelier", "ligne"]
}


def matches_synonyms(text, category):
    norm = normalize_name(text)
    return any(kw in norm for kw in SYNONYMS_DICTIONARY.get(category, []))


def smart_clean_dataframe(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            cleaned_series = df[col].astype(str).str.replace(' ', '').str.replace(',', '.')
            try:
                numeric_val = pd.to_numeric(cleaned_series, errors='raise')
                if numeric_val.notna().sum() > len(df) * 0.5:
                    df[col] = numeric_val
            except:
                pass
    return df


def load_uploaded_data(uploaded_file):
    if uploaded_file is None: return None
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        uploaded_file.seek(0)
        try: return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=";")
    if filename.endswith(".xlsx"):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)
    raise ValueError("Format non supporté.")


def standardize_dataframe(df):
    df = df.copy()
    df = df.drop(columns=[col for col in df.columns if df[col].notna().sum() == 0])
    df.columns = [str(col).strip() if str(col).strip() else "variable" for col in df.columns]
    return df


# ============================================================
# AGENT IA : ANALYSE SÉMANTIQUE DYNAMIQUE ET DÉCOUVERTE DE COLONNES
# ============================================================

def intelligent_agent_analysis(df, user_prompt=""):
    """
    L'agent lit la structure complète, détecte les rôles de base et analyse 
    les colonnes secondaires (ex: Organe, Cause) selon la demande de l'utilisateur.
    """
    semantics = []
    mapping = {"unit": None, "target": None, "exposure": None, "time": None, "segment": None}
    detected_segments = []

    for col in df.columns:
        series = df[col]
        role = "Variable générale"
        
        if matches_synonyms(col, "equipment"):
            role = "Identifiant d'équipement"
            if not mapping["unit"]: mapping["unit"] = col
        elif matches_synonyms(col, "failure"):
            role = "Événement / Panne"
            if not mapping["target"]: mapping["target"] = col
        elif matches_synonyms(col, "exposure"):
            role = "Exposition / Compteur"
            if not mapping["exposure"]: mapping["exposure"] = col
        elif matches_synonyms(col, "date"):
            role = "Horodatage / Date"
            if not mapping["time"]: mapping["time"] = col
        elif matches_synonyms(col, "segmentation") or (series.dtype == 'object' and series.nunique() < len(df) * 0.4):
            role = "Variable de segmentation dynamique"
            detected_segments.append(col)

        semantics.append({
            "Variable": col,
            "Type": str(series.dtype),
            "Rôle détecté par l'Agent": role,
            "Valeurs uniques": int(series.nunique(dropna=True))
        })

    # Intent analysis from user prompt
    prompt_lower = normalize_name(user_prompt)
    chosen_segment = None
    for seg in detected_segments:
        if normalize_name(seg) in prompt_lower:
            chosen_segment = seg
            break
    if not chosen_segment and detected_segments:
        chosen_segment = detected_segments[0] # Premier segment pertinent par défaut

    mapping["segment"] = chosen_segment

    # Fallbacks de sécurité
    if not mapping["unit"]: mapping["unit"] = df.columns[0]
    if not mapping["target"]: mapping["target"] = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    if not mapping["exposure"]: mapping["exposure"] = df.columns[2] if len(df.columns) > 2 else df.columns[0]

    return pd.DataFrame(semantics), mapping, detected_segments


# ============================================================
# MOTEUR SCIENTIFIQUE & CALCULS DE FIABILITÉ
# ============================================================

def kaplan_meier(duration, event):
    data = pd.DataFrame({"duration": pd.to_numeric(duration, errors="coerce"), "event": pd.to_numeric(event, errors="coerce")}).dropna()
    data = data[data["duration"] >= 0]
    data["event"] = (data["event"] > 0).astype(int)
    if len(data) == 0: return {"status": "NON EXECUTABLE"}
    times = np.sort(data["duration"].unique())
    survival = 1.0
    rows = []
    for t in times:
        at_risk = int((data["duration"] >= t).sum())
        events = int(((data["duration"] == t) & (data["event"] == 1)).sum())
        if at_risk > 0: survival *= (1 - events / at_risk)
        rows.append({"Temps": float(t), "À risque": at_risk, "Événements": events, "Survie": float(survival)})
    return {"status": "OK", "curve": pd.DataFrame(rows), "n": len(data), "events": int(data["event"].sum()), "censored": int((data["event"] == 0).sum())}


def weibull_fit(duration, event):
    if not SCIPY_AVAILABLE: return {"status": "NON EXECUTABLE"}
    data = pd.DataFrame({"duration": pd.to_numeric(duration, errors="coerce"), "event": pd.to_numeric(event, errors="coerce")}).dropna()
    data = data[data["duration"] > 0]
    data["event"] = (data["event"] > 0).astype(int)
    if len(data) < 5 or data["event"].sum() < 2: return {"status": "NON EXECUTABLE"}
    t, d = data["duration"].values, data["event"].values
    def neg_log_likelihood(params):
        log_beta, log_eta = params
        beta, eta = np.exp(log_beta), np.exp(log_eta)
        z = (t / eta) ** beta
        log_hazard = np.log(beta) - np.log(eta) + (beta - 1) * (np.log(t) - np.log(eta))
        return -np.sum(d * log_hazard - z)
    res = minimize(neg_log_likelihood, np.log([1.0, np.median(t)]), method="Nelder-Mead")
    if not res.success: return {"status": "NON EXECUTABLE"}
    beta, eta = np.exp(res.x[0]), np.exp(res.x[1])
    grid = np.linspace(max(t.min(), 1e-12), t.max(), 200)
    reliability = np.exp(-(grid / eta) ** beta)
    interp = "β < 1 : taux décroissant." if beta < 1 else ("β ≈ 1 : taux constant." if np.isclose(beta, 1, atol=0.05) else "β > 1 : taux croissant (vieillissement de la flotte).")
    return {"status": "OK", "beta": float(beta), "eta": float(eta), "aic": float(2 * 2 - 2 * (-res.fun)), "curve": pd.DataFrame({"Temps": grid, "Fiabilité": reliability}), "interpretation": interp}


def run_agent_engine(df, mapping, user_prompt):
    time_var = mapping["exposure"] or mapping["time"]
    target_var = mapping["target"]
    segment_var = mapping["segment"]

    execution = {}
    if time_var and target_var:
        execution["kaplan_meier"] = kaplan_meier(df[time_var], df[target_var])
        execution["weibull"] = weibull_fit(df[time_var], df[target_var])

    # Analyse dynamique par segment si détecté (ex: Organe)
    segment_stats = None
    if segment_var and segment_var in df.columns and target_var in df.columns:
        segment_stats = df.groupby(segment_var)[target_var].agg(['count', 'sum', 'mean']).reset_index()
        segment_stats.columns = [segment_var, 'Total Observations', 'Total Pannes', 'Taux de Panne Moyen']

    return {
        "execution": execution,
        "segment_analysis": segment_stats,
        "mapping_used": mapping,
        "dataset_profile": {"n_observations": len(df), "n_variables": len(df.columns)},
        "study_unit": mapping["unit"],
        "target": target_var,
        "segment": segment_var
    }


def generate_pdf(analysis):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1*cm, leftMargin=1*cm, topMargin=1*cm, bottomMargin=1*cm)
    styles = getSampleStyleSheet()
    story = [Paragraph("RAPPORT D'ANALYSE INTELLIGENTE", styles["Title"]), Spacer(1, 0.5*cm)]
    story.append(Paragraph(f"Unité suivie : {analysis['study_unit']}", styles["BodyText"]))
    story.append(Paragraph(f"Indicateur cible : {analysis['target']}", styles["BodyText"]))
    if analysis['segment']:
        story.append(Paragraph(f"Segmentation dynamique : {analysis['segment']}", styles["BodyText"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# INTERFACE STREAMLIT DE L'AGENT FLEXIBLE
# ============================================================

st.markdown('<div class="main-title">⚙️ Scientific Engineering Engine + Agent IA Flexible</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">L\'agent lit, comprend et adapte l\'analyse à la volée selon vos demandes.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("1 — Instruction à l'Agent IA")
    user_prompt = st.text_area(
        "💬 Que souhaitez-vous analyser ?",
        value="Je veux analyser l'historique des pannes, prédire la fiabilité et étudier l'impact par organe.",
        height=125,
        help="Exprimez votre besoin en français. L'agent détectera tout seul les colonnes utiles et s'adaptera."
    )
    
    st.header("2 — Données (CSV ou Excel)")
    uploaded_file = st.file_uploader("Importer votre fichier", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("👋 Veuillez importer votre fichier dans la barre latérale pour activer l'agent intelligent.")
    st.stop()

try:
    df_raw = load_uploaded_data(uploaded_file)
    df = smart_clean_dataframe(standardize_dataframe(df_raw))
    st.session_state.df = df
except Exception as exc:
    st.error(f"Erreur de lecture : {exc}")
    st.stop()

# L'Agent analyse le fichier et détecte les colonnes dynamiquement
semantics_df, auto_mapping, detected_segments = intelligent_agent_analysis(df, user_prompt)

st.header("3 — Analyse Sémantique & Découverte Dynamique")
col1, col2, col3 = st.columns(3)
col1.metric("Lignes analysées", f"{len(df):,}")
col2.metric("Colonnes détectées", f"{len(df.columns):,}")
col3.metric("Segments dynamiques", f"{len(detected_segments)}")

with st.expander("🔍 Voir comment l'Agent a compris vos colonnes", expanded=False):
    st.dataframe(semantics_df, use_container_width=True)

st.markdown("---")
st.header("4 — Validation de la Formalisation Agent")
st.success(f"✨ **L'Agent a configuré l'analyse automatiquement :** Machine = `{auto_mapping['unit']}` | Panne = `{auto_mapping['target']}` | Temps = `{auto_mapping['exposure']}` | Segmentation = `{auto_mapping['segment'] or 'Aucune'}`")

if st.button("🚀 Lancer l'Analyse Intelligente", type="primary", use_container_width=True):
    with st.spinner("L'Agent exécute les modèles et croise les colonnes dynamiques..."):
        analysis_result = run_agent_engine(df, auto_mapping, user_prompt)
        st.session_state.analysis = analysis_result

# Affichage des résultats
analysis = st.session_state.get("analysis")
if analysis:
    st.markdown("---")
    st.header("💡 Synthèse & Bilan de l'Agent")
    exec_res = analysis["execution"]
    
    summary_html = f"""
    <div class="block-ok">
        <h4 style="color: black;">📌 Résumé de l'analyse :</h4>
        <ul style="color: black;">
            <li><b>Équipements suivis :</b> {analysis['study_unit']}</li>
            <li><b>Indicateur de panne :</b> {analysis['target']}</li>
    """
    if analysis['segment']:
        summary_html += f"<li><b>Segmentation dynamique activée :</b> Croisement réalisé sur la colonne <b>{analysis['segment']}</b>.</li>"
        
    if "weibull" in exec_res and exec_res["weibull"].get("status") == "OK":
        beta = exec_res["weibull"]["beta"]
        eta = exec_res["weibull"]["eta"]
        summary_html += f"<li><b>Modèle de Weibull :</b> β = <b>{beta:.2f}</b>, η = <b>{eta:.1f}</b>. ({exec_res['weibull']['interpretation']})</li>"
    summary_html += "</ul></div>"
    st.markdown(summary_html, unsafe_allow_html=True)

    # Si une colonne de segmentation comme 'Organe' a été détectée et ciblée
    if analysis["segment_analysis"] is not None:
        st.markdown("---")
        st.header(f"📊 Analyse détaillée par {analysis['segment']}")
        st.dataframe(analysis["segment_analysis"], use_container_width=True)

    st.markdown("---")
    st.header("📊 Courbes de Fiabilité")
    if "weibull" in exec_res and exec_res["weibull"].get("status") == "OK":
        wb_curve = exec_res["weibull"]["curve"]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(wb_curve["Temps"], wb_curve["Fiabilité"], color="#2ca02c", lw=2)
        ax.set_xlabel("Temps / Exposition")
        ax.set_ylabel("Fiabilité R(t)")
        ax.set_title("Fonction de Fiabilité R(t)")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)

    st.markdown("---")
    st.header("📥 Export des Rapports")
    pdf_data = generate_pdf(analysis)
    st.download_button("📥 Télécharger le rapport PDF", data=pdf_data, file_name="Rapport_Agent.pdf", mime="application/pdf", use_container_width=True)
