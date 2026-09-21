# ============================================================
# SCIENTIFIC ENGINEERING ENGINE
# Meta-moteur générique d'ingénierie scientifique industrielle
#
# VERSION 3.2 (Nettoyage automatique & Robustesse industrielle)
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

# SciPy est utilisé pour les méthodes statistiques avancées.
try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)


# ============================================================
# CONFIGURATION
# ============================================================

ENGINE_NAME = "Scientific Engineering Engine"
ENGINE_VERSION = "3.2"

st.set_page_config(
    page_title=ENGINE_NAME,
    page_icon="⚙️",
    layout="wide",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #777;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .help-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 15px;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "analysis": None,
    "df": None,
    "confirmations": {},
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# OUTILS DE NETTOYAGE AUTOMATIQUE INDUSTRIEL
# ============================================================

def normalize_name(value):
    value = str(value).strip().lower()
    replacements = {
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "à": "a", "â": "a", "ä": "a",
        "î": "i", "ï": "i",
        "ô": "o", "ö": "o",
        "ù": "u", "û": "u", "ü": "u",
        "ç": "c",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def clean_numeric_columns(df):
    """
    Nettoie automatiquement les colonnes textuelles qui contiennent des nombres
    piégés avec des espaces (ex: '85 000') ou des virgules.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            # Tente de supprimer les espaces insécables/normaux et remplacer les virgules
            cleaned = df[col].astype(str).str.replace(' ', '').str.replace('\xa0', '').str.replace(',', '.')
            converted = pd.to_numeric(cleaned, errors='coerce')
            # Si plus de 50% des valeurs non-nulles sont convertibles en chiffres, on bascule la colonne en numérique
            if converted.notna().sum() >= max(1, df[col].dropna().count() * 0.5):
                df[col] = converted
    return df


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def contains_any(text, keywords):
    text = normalize_name(text)
    return any(normalize_name(keyword) in text for keyword in keywords)


def json_safe(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def dataframe_to_json_records(df):
    if df is None or df.empty:
        return []
    return json_safe(df.replace({np.nan: None, np.inf: None, -np.inf: None}))


# ============================================================
# CHARGEMENT ET STANDARDISATION
# ============================================================

def load_uploaded_data(uploaded_file):
    if uploaded_file is None:
        return None
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, sep=";")
            except Exception as exc:
                raise ValueError(f"Impossible de lire le CSV : {exc}")
    elif filename.endswith(".xlsx"):
        uploaded_file.seek(0)
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as exc:
            raise ValueError(f"Impossible de lire le fichier Excel : {exc}")
    else:
        raise ValueError("Format non supporté. Utilisez CSV ou XLSX.")
    
    # Nettoyage automatique des espaces / nombres piégés
    df = clean_numeric_columns(df)
    return df


def standardize_dataframe(df):
    df = df.copy()
    empty_columns = [col for col in df.columns if df[col].notna().sum() == 0]
    if empty_columns:
        df = df.drop(columns=empty_columns)
    new_columns = []
    for col in df.columns:
        clean = str(col).strip()
        if not clean:
            clean = "variable"
        new_columns.append(clean)
    df.columns = new_columns
    return df


# ============================================================
# INFERENCE SÉMANTIQUE & VOCABULAIRE ÉLARGI
# ============================================================

def infer_variable_semantics(df):
    records = []
    for column in df.columns:
        series = df[column]
        normalized = normalize_name(column)

        role = "Variable générale"
        confidence = 0.30

        if pd.api.types.is_datetime64_any_dtype(series):
            role = "Variable temporelle"
            confidence = 0.99
        elif contains_any(normalized, ["date", "datetime", "timestamp", "heure", "jour", "annee", "year", "time", "periode", "mois", "horodatage", "moment"]):
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() >= 0.70:
                role = "Variable temporelle"
                confidence = 0.90
        elif contains_any(normalized, ["id", "identifiant", "asset", "equipment", "machine", "unit", "sample", "observation", "ref", "reference", "code", "numero", "equipement", "poste", "ligne", "produit", "lot"]):
            role = "Identifiant potentiel"
            confidence = 0.85
        elif contains_any(normalized, ["event", "failure", "fault", "defaut", "defaillance", "incident", "panne", "target", "label", "class", "classe", "outcome", "result", "resultat", "casse", "arret", "rebut", "non_conformite", "alerte", "danger", "risque", "probleme"]):
            role = "Événement / cible potentielle"
            confidence = 0.85
        elif contains_any(normalized, ["exposition", "exposure", "distance", "kilometrage", "mileage", "hours", "heures", "cycles", "cycle", "volume", "quantite", "production", "temps_marche", "duree_fonctionnement", "cadence"]):
            role = "Exposition potentielle"
            confidence = 0.80
        elif contains_any(normalized, ["duration", "duree", "delay", "delai", "temps_arret", "latence"]):
            role = "Durée potentielle"
            confidence = 0.72
        elif pd.api.types.is_numeric_dtype(series):
            role = "Variable quantitative"
            confidence = 0.60
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series):
            role = "Variable catégorielle"
            confidence = 0.55

        records.append({
            "Variable": column,
            "Type": str(series.dtype),
            "Rôle inféré": role,
            "Confiance": round(confidence, 2),
            "Valeurs manquantes (%)": round(series.isna().mean() * 100, 2),
            "Valeurs uniques": int(series.nunique(dropna=True)),
            "Min": safe_float(series.min()) if pd.api.types.is_numeric_dtype(series) else None,
            "Max": safe_float(series.max()) if pd.api.types.is_numeric_dtype(series) else None,
        })
    return pd.DataFrame(records)


def inspect_data_quality(df):
    issues = []
    if df is None or df.empty:
        return [("CRITIQUE", "Jeu de données vide.")]
    if len(df) < 5:
        issues.append(("CRITIQUE", "Moins de 5 observations."))
    return issues


DOMAIN_KEYWORDS = {
    "Fiabilité": ["fiabilite", "panne", "defaillance", "duree de vie", "survie", "vieillissement", "usure", "failure", "reliability", "survival", "lifetime", "degradation", "casse", "rupture", "fatigue"],
    "Maintenance": ["maintenance", "intervention", "reparation", "immobilisation", "gmao", "preventive", "corrective", "technicien", "depannage"],
    "Qualité": ["qualite", "defaut", "non conformite", "rebuts", "rebut", "defective", "quality", "retouche", "ecart"],
    "Production": ["production", "cadence", "trs", "oee", "rendement", "temps de cycle", "goulot", "capacite", "volume"],
    "Énergie": ["energie", "consommation", "kwh", "puissance", "electrique", "energy", "power"],
    "Logistique": ["logistique", "stock", "inventaire", "flux", "transport", "approvisionnement"],
    "Sécurité": ["securite", "accident", "incident", "risque", "danger", "safety"],
    "Process": ["process", "procede", "parametre", "reglage", "processus", "temperature", "pression", "vitesse"],
}

QUESTION_KEYWORDS = {
    "Description": ["decrire", "analyser", "repartition", "comprendre", "caracteriser", "profil", "distribution"],
    "Explication / diagnostic": ["pourquoi", "cause", "causes", "origine", "facteur", "expliquer", "diagnostic", "influence", "associe", "association", "lie_a"],
    "Prédiction": ["predire", "prevoir", "prediction", "anticiper", "forecast", "predict"],
    "Comparaison": ["comparer", "comparaison", "difference", "compare", "versus", "vs"],
    "Optimisation": ["optimiser", "optimisation", "reduire", "ameliorer", "minimiser", "maximiser", "optimize"],
    "Détection": ["detecter", "anomalie", "anomalies", "derive", "surveillance", "detection", "monitoring"],
    "Simulation": ["simuler", "simulation", "scenario", "what if"],
}

def classify_problem(problem, objective):
    text = f"{problem} {objective}"
    domains = [d for d, keywords in DOMAIN_KEYWORDS.items() if contains_any(text, keywords)]
    questions = [q for q, keywords in QUESTION_KEYWORDS.items() if contains_any(text, keywords)]
    if not domains: domains = ["Domaine à déterminer"]
    if not questions: questions = ["Description", "Explication / diagnostic"]
    return domains, questions


def characterize_dataset(df):
    numeric = list(df.select_dtypes(include=np.number).columns)
    categorical = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    datetime_columns = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
    return {
        "n_observations": len(df),
        "n_variables": len(df.columns),
        "numeric_variables": numeric,
        "categorical_variables": categorical,
        "datetime_variables": datetime_columns,
        "numeric_count": len(numeric),
        "categorical_count": len(categorical),
        "has_time": bool(datetime_columns),
    }


def identify_study_unit(df, semantics):
    candidates = []
    for _, row in semantics.iterrows():
        role = str(row["Rôle inféré"]).lower()
        if "identifiant" in role:
            candidates.append({"Variable": row["Variable"], "Confiance": row["Confiance"]})
    if candidates:
        res = pd.DataFrame(candidates).sort_values("Confiance", ascending=False)
        return {"candidate": res.iloc[0]["Variable"]}
    return {"candidate": df.columns[0] if len(df.columns) > 0 else None}


def identify_candidate_targets(df, semantics):
    candidates = []
    for _, row in semantics.iterrows():
        if "cible" in str(row["Rôle inféré"]).lower() or "événement" in str(row["Rôle inféré"]).lower():
            candidates.append({"Variable": row["Variable"], "Justification": row["Rôle inféré"], "Confiance": row["Confiance"]})
    for col in df.columns:
        if not any(c["Variable"] == col for c in candidates):
            candidates.append({"Variable": col, "Justification": "Candidate générale", "Confiance": 0.4})
    return pd.DataFrame(candidates)


# ============================================================
# EXÉCUTION DES MÉTHODES STATISTIQUES & MACHINE LEARNING ROBUSTE
# ============================================================

def prepare_full_matrix(df, target):
    """
    Prépare la matrice de données en encodant automatiquement toutes les variables textuelles/catégorielles
    pour permettre la régression logistique ou linéaire même sans nettoyage manuel du fichier.
    """
    if target not in df.columns:
        return None
    
    # Exclure les colonnes textuelles purement descriptives (commentaires, ID unique)
    exclude_cols = [target, 'ID', 'Commentaire', 'Commentaire.1', 'Date']
    features = [c for c in df.columns if c not in exclude_cols]
    
    work = df[features + [target]].copy()
    
    # Encodage automatique (One-Hot Encoding) pour toutes les variables non numériques
    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(work[c])]
    if cat_cols:
        work = pd.get_dummies(work, columns=cat_cols, drop_first=True, dtype=float)
        
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        return None
        
    X_cols = [c for c in work.columns if c != target]
    return work[X_cols].astype(float).values, work[target].astype(float).values, X_cols


def logistic_regression_influence_analysis(df, target):
    """
    Exécute une régression logistique pour identifier quelles variables influencent le plus la cible (ex: pannes).
    """
    if not SCIPY_AVAILABLE:
        return {"status": "NON EXECUTABLE", "message": "SciPy requis."}
    
    prepared = prepare_full_matrix(df, target)
    if prepared is None:
        return {"status": "NON EXECUTABLE", "message": "Impossible de construire la matrice de données."}
        
    X, y, columns = prepared
    unique_y = np.unique(y)
    
    if len(unique_y) < 2:
        return {"status": "NON EXECUTABLE", "message": "La cible Y doit avoir au moins 2 valeurs différentes (ex: 0 et 1)."}
        
    # Si binaire, s'assurer que c'est 0 ou 1
    y_bin = (y == unique_y[1]).astype(float)
    
    from sklearn.linear_model import LogisticRegression
    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y_bin)
        
        # Coefficients et importance
        coefs = pd.DataFrame({
            "Facteur / Variable": columns,
            "Coefficient": model.coef_[0],
            "Impact absolu": np.abs(model.coef_[0])
        }).sort_values(by="Impact absolu", ascending=False)
        
        accuracy = np.mean(model.predict(X) == y_bin)
        
        return {
            "status": "OK",
            "accuracy": float(accuracy),
            "coefficients": coefs.reset_index(drop=True)
        }
    except Exception as exc:
        return {"status": "NON EXECUTABLE", "message": str(exc)}


def run_scientific_engine(problem, objective, df, confirmations):
    df = standardize_dataframe(df)
    domains, questions = classify_problem(problem, objective)
    semantics = infer_variable_semantics(df)
    dataset_profile = characterize_dataset(df)
    quality_issues = inspect_data_quality(df)
    
    study_unit = confirmations.get("study_unit") or identify_study_unit(df, semantics)["candidate"]
    target = confirmations.get("target")
    if target and target not in df.columns: target = None

    execution = {}
    if target:
        # Lancer l'analyse d'influence des facteurs sur la cible Y
        res_log = logistic_regression_influence_analysis(df, target)
        if res_log["status"] == "OK":
            execution["logistic_regression"] = res_log
            
    # Statistiques descriptives de base
    numeric_stats = df.select_dtypes(include=np.number).describe().T if not df.select_dtypes(include=np.number).empty else pd.DataFrame()
    execution["descriptive_statistics"] = {"status": "OK", "numeric": numeric_stats}

    analysis = {
        "engine": ENGINE_NAME, "version": ENGINE_VERSION,
        "problem": problem, "objective": objective,
        "domains": domains, "questions": questions,
        "dataset_profile": dataset_profile, "semantics": semantics,
        "quality_issues": quality_issues, "study_unit": study_unit,
        "target": target, "execution": execution,
        "interpretation": [
            "Analyse robuste effectuée avec nettoyage et encodage automatique des données.",
            "Les facteurs ayant le plus fort impact sur la cible ont été extraits via modélisation statistique."
        ],
        "generated_at": datetime.now().isoformat(),
    }
    return analysis


def generate_pdf(analysis):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.3*cm, leftMargin=1.3*cm, topMargin=1.3*cm, bottomMargin=1.3*cm)
    styles = getSampleStyleSheet()
    story = [Paragraph("RAPPORT D'ANALYSE INDUSTRIELLE ROBUSTE", styles["Title"]), Spacer(1, 0.5*cm)]
    story.append(Paragraph(f"<b>Problème :</b> {analysis['problem']}", styles["BodyText"]))
    story.append(Spacer(1, 0.3*cm))
    
    if "logistic_regression" in analysis["execution"]:
        reg = analysis["execution"]["logistic_regression"]
        story.append(Paragraph(f"<b>Précision du modèle :</b> {reg['accuracy']*100:.1f}%", styles["BodyText"]))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("<b>Top des facteurs influençant la cible :</b>", styles["Heading3"]))
        for _, row in reg["coefficients"].head(5).iterrows():
            story.append(Paragraph(f"• {row['Facteur / Variable']} (Impact : {row['Coefficient']:.3f})", styles["BodyText"]))
            
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# INTERFACE UTILISATEUR STREAMLIT
# ============================================================

st.markdown('<div class="main-title">⚙️ Scientific Engineering Engine (v3.2)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Méta-moteur avec nettoyage automatique intelligent des données industrielles.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("1 — Votre Problème")
    problem_desc = st.text_area(
        "Décrivez votre situation ou votre problème",
        height=150,
        placeholder="Ex: Quelle variable influence le plus la hausse des pannes..."
    )
    objective = st.text_input("Objectif principal", placeholder="Ex: Expliquer, prédire...")
    
    st.header("2 — Vos Données")
    uploaded_file = st.file_uploader("Importer votre fichier (CSV ou Excel)", type=["csv", "xlsx"])


if uploaded_file is None:
    st.info("👋 Veuillez importer votre fichier de données dans le volet de gauche pour démarrer.")
    st.stop()

try:
    df = load_uploaded_data(uploaded_file)
    df = standardize_dataframe(df)
    st.session_state.df = df
except Exception as exc:
    st.error(f"Erreur de lecture ou de nettoyage du fichier : {exc}")
    st.stop()

st.header("3 — Aperçu des données (Nettoyées automatiquement)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes", f"{len(df):,}")
c2.metric("Colonnes", f"{len(df.columns):,}")
c3.metric("Valeurs manquantes", f"{int(df.isna().sum().sum()):,}")
c4.metric("Doublons", f"{int(df.duplicated().sum()):,}")

with st.expander("Voir un extrait des données"):
    st.dataframe(df.head(10), use_container_width=True)


st.markdown("---")
st.header("4 — Paramétrage simple de l'étude")

col_f1, col_f2 = st.columns(2)
with col_f1:
    target_options = ["— Aucune sélection —"] + list(df.columns)
    target_choice = st.selectbox("Sélectionnez votre variable Cible Y (ex: Défaillance / Panne)", target_options)

with col_f2:
    unit_options = ["— Aucune sélection —"] + list(df.columns)
    unit_choice = st.selectbox("Sélectionnez l'élément unique / Unité (ex: Équipement / ID)", unit_options)


if st.button("🚀 Lancer l'analyse robuste", type="primary", use_container_width=True):
    confirmations = {
        "target": None if target_choice == "— Aucune sélection —" else target_choice,
        "study_unit": None if unit_choice == "— Aucune sélection —" else unit_choice,
    }
    with st.spinner("Nettoyage intelligent et exécution des modèles statistiques en cours..."):
        analysis = run_scientific_engine(problem_desc, objective, df, confirmations)
        st.session_state.analysis = analysis
    st.success("Analyse terminée avec succès !")


analysis = st.session_state.analysis
if analysis is not None:
    st.markdown("---")
    st.header("📊 Résultats de l'analyse d'influence")
    
    if "logistic_regression" in analysis["execution"]:
        reg = analysis["execution"]["logistic_regression"]
        st.metric("Précision du modèle d'explication", f"{reg['accuracy']*100:.1f}%")
        
        st.subheader("Classement des variables qui influencent le plus la cible :")
        st.dataframe(reg["coefficients"], use_container_width=True)
        st.info("💡 Un coefficient positif pousse vers la hausse de la panne, tandis qu'un coefficient négatif (comme la maintenance préventive ou la censure) réduit ou protège de la panne.")
    else:
        st.warning("Veuillez sélectionner une variable cible valide pour lancer l'analyse d'influence.")

    st.markdown("---")
    pdf_data = generate_pdf(analysis)
    st.download_button("📥 Télécharger le rapport d'analyse (PDF)", data=pdf_data, file_name="Rapport_Analyse_Robuste.pdf", mime="application/pdf", use_container_width=True)
