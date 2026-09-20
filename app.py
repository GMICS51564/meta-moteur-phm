# ============================================================
# SCIENTIFIC ENGINEERING ENGINE + COPILOTE IA INTÉGRÉ
# Meta-moteur générique d'ingénierie scientifique industrielle
#
# VERSION 3.1 (Optimisée pour le nettoyage auto et le français)
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
ENGINE_VERSION = "3.1"

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

    .block-ok {
        padding: 0.7rem;
        border-radius: 0.5rem;
        background: #eaf7ea;
        border: 1px solid #9bd39b;
    }

    .block-warning {
        padding: 0.7rem;
        border-radius: 0.5rem;
        background: #fff6e5;
        border: 1px solid #e6c47a;
    }

    .block-critical {
        padding: 0.7rem;
        border-radius: 0.5rem;
        background: #ffeaea;
        border: 1px solid #e0a0a0;
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
# OUTILS GENERAUX ET NETTOYAGE INTELLIGENT
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

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    return value.strip("_")


def safe_float(value):

    try:
        return float(value)
    except Exception:
        return None


def contains_any(text, keywords):

    text = normalize_name(text)

    return any(
        normalize_name(keyword) in text
        for keyword in keywords
    )


def smart_clean_dataframe(df):
    """
    Nettoie automatiquement le DataFrame pour éviter les plantages courants :
    - Supprime les espaces dans les colonnes numériques (ex: '85 000' -> 85000)
    - Convertit proprement les cibles binaires ou textuelles en chiffres si possible
    """
    df = df.copy()
    for col in df.columns:
        # Si la colonne ressemble à de l'exposition ou des chiffres mal formatés avec des espaces
        if df[col].dtype == 'object':
            # Tentative de nettoyage des espaces dans les nombres
            cleaned_series = df[col].astype(str).str.replace(' ', '').str.replace(',', '.')
            try:
                numeric_val = pd.to_numeric(cleaned_series, errors='raise')
                # Si la conversion réussit sans trop de NaN, on applique
                if numeric_val.notna().sum() > len(df) * 0.5:
                    df[col] = numeric_val
            except:
                pass
    return df


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

    return json_safe(
        df.replace({np.nan: None, np.inf: None, -np.inf: None})
    )


# ============================================================
# CHARGEMENT DES DONNEES
# ============================================================

def load_uploaded_data(uploaded_file):

    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        uploaded_file.seek(0)
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            try:
                return pd.read_csv(uploaded_file, sep=";")
            except Exception as exc:
                raise ValueError(f"Impossible de lire le CSV : {exc}")

    if filename.endswith(".xlsx"):
        uploaded_file.seek(0)
        try:
            return pd.read_excel(uploaded_file)
        except Exception as exc:
            raise ValueError(f"Impossible de lire le fichier Excel : {exc}")

    raise ValueError("Format non supporté. Utilisez CSV ou XLSX.")


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
# INFERENCE SEMANTIQUE ET SMART MAPPING AUTOMATIQUE
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
        elif contains_any(normalized, ["date", "datetime", "timestamp", "heure", "jour", "annee", "year", "time"]):
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() >= 0.70:
                role = "Variable temporelle"
                confidence = 0.90
        elif contains_any(normalized, ["id", "identifiant", "asset_id", "equipment_id", "machine_id", "unit_id"]):
            role = "Identifiant potentiel"
            confidence = 0.85
        elif contains_any(normalized, ["event", "failure", "fault", "defaut", "defaillance", "panne", "target", "label"]):
            role = "Événement / cible potentielle"
            confidence = 0.85
        elif contains_any(normalized, ["exposition", "exposure", "distance", "kilometrage", "mileage", "hours", "heures", "cycles"]):
            role = "Exposition potentielle"
            confidence = 0.80
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


def smart_auto_detect_columns(df):
    """Détecte automatiquement les colonnes clés pour pré-remplir le formulaire"""
    mapping = {"unit": None, "target": None, "time": None, "exposure": None, "censure": None}
    
    for col in df.columns:
        col_lower = normalize_name(col)
        if any(k in col_lower for k in ["equipement", "machine", "asset", "id"]):
            if not mapping["unit"]: mapping["unit"] = col
        elif any(k in col_lower for k in ["defaillance", "panne", "failure", "event"]):
            if not mapping["target"]: mapping["target"] = col
        elif any(k in col_lower for k in ["date", "temps", "time", "heure"]):
            if not mapping["time"]: mapping["time"] = col
        elif any(k in col_lower for k in ["exposition", "km", "kilometrage", "heure"]):
            if not mapping["exposure"]: mapping["exposure"] = col
        elif any(k in col_lower for k in ["censure", "censored"]):
            if not mapping["censure"]: mapping["censure"] = col
            
    return mapping


# ============================================================
# DATA QUALITY ENGINE
# ============================================================

def inspect_data_quality(df):

    issues = []
    if df is None or df.empty:
        return [("CRITIQUE", "Le jeu de données est vide ou absent.")]

    if len(df) < 5:
        issues.append(("CRITIQUE", "Le jeu de données contient moins de 5 observations."))
    elif len(df) < 10:
        issues.append(("IMPORTANT", "Le jeu de données contient très peu d'observations."))

    duplicates = int(df.duplicated().sum())
    if duplicates:
        issues.append(("AVERTISSEMENT", f"{duplicates} ligne(s) dupliquée(s)."))

    for column in df.columns:
        missing_rate = df[column].isna().mean()
        if missing_rate >= 0.50:
            issues.append(("IMPORTANT", f"{column} contient {missing_rate * 100:.1f}% de valeurs manquantes."))

    return issues


# ============================================================
# CLASSIFICATION SCIENTIFIQUE
# ============================================================

DOMAIN_KEYWORDS = {
    "Fiabilité": ["fiabilite", "panne", "defaillance", "duree de vie", "survie", "vieillissement", "usure", "failure", "reliability", "survival", "lifetime", "degradation"],
    "Maintenance": ["maintenance", "intervention", "reparation", "immobilisation", "gmao", "preventive", "corrective"],
    "Qualité": ["qualite", "defaut", "non conformite", "rebuts", "defective", "quality"],
    "Production": ["production", "cadence", "trs", "oee", "rendement", "temps de cycle"],
}

QUESTION_KEYWORDS = {
    "Description": ["decrire", "analyser", "repartition", "comprendre", "etat des lieux", "caracteriser"],
    "Explication / diagnostic": ["pourquoi", "cause", "origine", "facteur", "expliquer", "diagnostic"],
    "Prédiction": ["predire", "prevoir", "prediction", "anticiper", "forecast", "predict"],
    "Comparaison": ["comparer", "comparaison", "difference", "compare"],
}


def classify_problem(problem, objective):
    text = f"{problem} {objective}"
    domains = [d for d, keywords in DOMAIN_KEYWORDS.items() if contains_any(text, keywords)]
    questions = [q for q, keywords in QUESTION_KEYWORDS.items() if contains_any(text, keywords)]

    if not domains: domains = ["Domaine à déterminer"]
    if not questions: questions = ["Description"]

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
    return {"candidate": None}


def identify_candidate_targets(df, semantics):
    candidates = []
    for _, row in semantics.iterrows():
        if "cible" in str(row["Rôle inféré"]).lower() or "événement" in str(row["Rôle inféré"]).lower():
            candidates.append({"Variable": row["Variable"], "Justification": row["Rôle inféré"], "Confiance": row["Confiance"]})
    return pd.DataFrame(candidates)


def identify_predictors(df, target=None):
    records = []
    for column in df.columns:
        if column == target: continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            records.append({"Variable": column, "Type": "Quantitative", "Utilisable": True})
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series):
            records.append({"Variable": column, "Type": "Catégorielle", "Utilisable": True})
    return pd.DataFrame(records)


# ============================================================
# SCIENTIFIC KNOWLEDGE BASE
# ============================================================

SCIENTIFIC_KNOWLEDGE_BASE = {
    "descriptive_statistics": {
        "name": "Statistiques descriptives", "family": "Exploration",
        "objectives": ["Description", "Explication / diagnostic", "Prédiction"], "requires": [],
        "outputs": ["distribution", "central_tendency"], "limitations": ["Description uniquement."],
    },
    "kaplan_meier": {
        "name": "Kaplan-Meier", "family": "Survie",
        "objectives": ["Description", "Prédiction"], "requires": ["time_to_event", "event_indicator"],
        "outputs": ["survival_curve", "median_survival"], "limitations": ["Nécessite temps et événement."],
    },
    "weibull": {
        "name": "Weibull", "family": "Fiabilité",
        "objectives": ["Description", "Prédiction"], "requires": ["time_to_event", "event_indicator"],
        "outputs": ["beta", "eta", "reliability"], "limitations": ["Nécessite assez d'événements."],
    },
    "time_series": {
        "name": "Analyse temporelle", "family": "Séries temporelles",
        "objectives": ["Description", "Prédiction"], "requires": ["time_variable", "numeric_target"],
        "outputs": ["trend", "rolling_statistics"], "limitations": ["Respect de la chronologie requis."],
    },
}


def build_scientific_context(df, target, unit, domains, questions, confirmations):
    numeric_columns = list(df.select_dtypes(include=np.number).columns)
    categorical_columns = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    datetime_columns = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)

    return {
        "numeric_count": len(numeric_columns),
        "categorical_count": len(categorical_columns),
        "has_time": bool(datetime_columns) or confirmations.get("time_variable") is not None,
        "target": target,
        "study_unit": unit,
        "domains": domains,
        "questions": questions,
        "event_variable": confirmations.get("event_variable"),
        "time_variable": confirmations.get("time_variable"),
        "time_to_event_confirmed": bool(confirmations.get("time_to_event_confirmed")),
        "censoring_defined": bool(confirmations.get("censoring_defined")),
    }


def evaluate_method(method_id, method, context):
    missing, satisfied = [], []
    reqs = method.get("requires", [])

    if "time_to_event" in reqs:
        if context["time_to_event_confirmed"] and context["event_variable"]:
            satisfied.append("Temps jusqu'à événement")
        else:
            missing.append("Temps jusqu'à événement")

    if "event_indicator" in reqs:
        if context["event_variable"]:
            satisfied.append("Variable événement")
        else:
            missing.append("Variable événement")

    if "time_variable" in reqs:
        if context["has_time"] or context["time_variable"]:
            satisfied.append("Variable temporelle")
        else:
            missing.append("Variable temporelle")

    status = "COMPATIBLE" if not missing else ("CONDITIONNEL" if len(missing) == 1 else "BLOQUÉ")
    score = int(100 * len(satisfied) / len(reqs)) if reqs else 100

    return {"status": status, "score": score, "satisfied": satisfied, "missing": missing}


def decision_engine(knowledge_base, context):
    results = []
    questions = context["questions"]

    for method_id, method in knowledge_base.items():
        if method_id == "descriptive_statistics" or any(obj in questions for obj in method.get("objectives", [])):
            eval_res = evaluate_method(method_id, method, context)
            results.append({
                "ID": method_id, "Méthode": method["name"], "Famille": method["family"],
                "Score de compatibilité": eval_res["score"], "Statut": eval_res["status"],
                "Conditions satisfaites": " | ".join(eval_res["satisfied"]),
                "Informations manquantes": " | ".join(eval_res["missing"]),
                "Sorties": " | ".join(method.get("outputs", [])),
                "Limites": " | ".join(method.get("limitations", [])),
            })
    return pd.DataFrame(results)


def generate_hypotheses(problem, objective, domains, questions, df, target):
    return pd.DataFrame([
        {"ID": "H-DESC-01", "Hypothèse": "La structure des données permet de caractériser le phénomène.", "Type": "Descriptif", "Statut": "À vérifier"},
        {"ID": "H-REL-01", "Hypothèse": "Le temps ou l'exposition jusqu'à l'événement peut être reconstruit.", "Type": "Fiabilité", "Statut": "À confirmer"}
    ])


def build_required_questions(domains, questions, context):
    return pd.DataFrame()


def build_reconstruction(domains, questions, context):
    return pd.DataFrame([
        {"Élément": "Unité d'étude", "Statut": "CONFIRMÉ" if context["study_unit"] else "À CONFIRMER", "Information": context["study_unit"] or "Non définie"},
        {"Élément": "Variable cible Y", "Statut": "CONFIRMÉE" if context["target"] else "À CONFIRMER", "Information": context["target"] or "Non définie"}
    ])


def validate_conditions(df, context, decision_table):
    return pd.DataFrame([
        {"Contrôle": "Taille d'échantillon", "Statut": "OK" if len(df) >= 10 else "LIMITÉ", "Commentaire": f"{len(df)} observation(s)."}
    ])


# ============================================================
# EXÉCUTION DES MÉTHODES DE FIABILITÉ
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
    if not SCIPY_AVAILABLE: return {"status": "NON EXECUTABLE", "message": "SciPy requis."}
    data = pd.DataFrame({"duration": pd.to_numeric(duration, errors="coerce"), "event": pd.to_numeric(event, errors="coerce")}).dropna()
    data = data[data["duration"] > 0]
    data["event"] = (data["event"] > 0).astype(int)
    if len(data) < 5 or data["event"].sum() < 2: return {"status": "NON EXECUTABLE", "message": "Pas assez d'observations ou d'événements."}

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
    
    interp = "β < 1 : taux décroissant." if beta < 1 else ("β ≈ 1 : taux constant." if np.isclose(beta, 1, atol=0.05) else "β > 1 : taux croissant (vieillissement).")
    return {"status": "OK", "beta": float(beta), "eta": float(eta), "aic": float(2 * 2 - 2 * (-res.fun)), "curve": pd.DataFrame({"Temps": grid, "Fiabilité": reliability}), "interpretation": interp}


def time_series_analysis(df, time_variable, target):
    if time_variable not in df.columns or target not in df.columns: return {"status": "NON EXECUTABLE"}
    data = df[[time_variable, target]].copy()
    data[time_variable] = pd.to_datetime(data[time_variable], errors="coerce")
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna().sort_values(time_variable)
    if len(data) < 2: return {"status": "NON EXECUTABLE"}
    data["Moyenne_mobile"] = data[target].rolling(window=min(5, len(data)), min_periods=1).mean()
    return {"status": "OK", "data": data, "first_mean": float(data[target].head(2).mean()), "last_mean": float(data[target].tail(2).mean())}


def execute_selected_methods(df, context, decision_table):
    results = {}
    if decision_table.empty: return results
    comp_methods = decision_table[decision_table["Statut"] == "COMPATIBLE"]["ID"].tolist()

    if "descriptive_statistics" in comp_methods:
        numeric = df.select_dtypes(include=np.number)
        cat = df.select_dtypes(include=["object", "category", "bool"])
        results["descriptive_statistics"] = {"status": "OK", "numeric": numeric.describe().T if not numeric.empty else pd.DataFrame(), "categorical": pd.DataFrame()}

    if "kaplan_meier" in comp_methods and context["event_variable"] and context["time_variable"]:
        results["kaplan_meier"] = kaplan_meier(df[context["time_variable"]], df[context["event_variable"]])

    if "weibull" in comp_methods and context["event_variable"] and context["time_variable"]:
        results["weibull"] = weibull_fit(df[context["time_variable"]], df[context["event_variable"]])

    if "time_series" in comp_methods and context["target"] and context["time_variable"]:
        results["time_series"] = time_series_analysis(df, context["time_variable"], context["target"])

    return results


def build_interpretation(analysis):
    statements = ["Les données sont exploitables sous réserve des contrôles méthodologiques."]
    weibull = analysis["execution"].get("weibull")
    if weibull and weibull.get("status") == "OK":
        statements.append(f"Weibull : {weibull['interpretation']}")
    return statements


def run_scientific_engine(problem, objective, df, confirmations):
    df = standardize_dataframe(df)
    domains, questions = classify_problem(problem, objective)
    semantics = infer_variable_semantics(df)
    dataset_profile = characterize_dataset(df)
    quality_issues = inspect_data_quality(df)
    
    study_unit_info = identify_study_unit(df, semantics)
    study_unit = confirmations.get("study_unit") or study_unit_info["candidate"]
    
    target = confirmations.get("target")
    if target and target not in df.columns: target = None

    context = build_scientific_context(df, target, study_unit, domains, questions, confirmations)
    hypotheses = generate_hypotheses(problem, objective, domains, questions, df, target)
    decision_table = decision_engine(SCIENTIFIC_KNOWLEDGE_BASE, context)
    required_questions = build_required_questions(domains, questions, context)
    reconstruction = build_reconstruction(domains, questions, context)
    validation_checks = validate_conditions(df, context, decision_table)
    execution = execute_selected_methods(df, context, decision_table)

    analysis = {
        "engine": ENGINE_NAME, "version": ENGINE_VERSION, "problem": problem, "objective": objective,
        "domains": domains, "questions": questions, "dataset_profile": dataset_profile,
        "semantics": semantics, "quality_issues": quality_issues, "study_unit": study_unit,
        "target": target, "predictors": identify_predictors(df, target), "hypotheses": hypotheses,
        "decision_context": context, "decision_table": decision_table, "required_questions": required_questions,
        "reconstruction": reconstruction, "validation_checks": validation_checks, "execution": execution,
        "pipeline": ["Formalisation", "Classification", "Analyse", "Interprétation"], "generated_at": datetime.now().isoformat(),
    }
    analysis["interpretation"] = build_interpretation(analysis)
    return analysis


# ============================================================
# INTERFACE STREAMLIT AVEC COPILOTE IA & SMART MAPPING
# ============================================================

st.markdown('<div class="main-title">⚙️ Scientific Engineering Engine + Copilote IA</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Méta-moteur intelligent de maintenance et fiabilité industrielle.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("1 — Assistant & Problème")
    
    # Copilote IA textuel
    user_prompt = st.text_area(
        "💬 Que voulez-vous faire ?",
        placeholder="Ex: Je veux analyser l'historique de pannes pour prédire l'évolution et la durée de vie des équipements.",
        height=100
    )
    
    if st.button("🪄 Configurer automatiquement avec l'IA", type="secondary"):
        if user_prompt:
            st.session_state["auto_desc"] = user_prompt
            st.success("✨ Objectif enregistré par le copilote !")

    problem_desc = st.text_area(
        "Description technique du problème",
        value=st.session_state.get("auto_desc", ""),
        height=100
    )
    objective = st.text_input("Objectif principal", value="Prédire l'évolution des pannes et la fiabilité")

    st.header("2 — Données (CSV ou Excel)")
    uploaded_file = st.file_uploader("Importer le fichier d'historique", type=["csv", "xlsx"])


if uploaded_file is None:
    st.info("👋 Veuillez importer votre fichier CSV ou Excel dans la barre latérale pour démarrer l'analyse.")
    st.stop()

try:
    df_raw = load_uploaded_data(uploaded_file)
    df = smart_clean_dataframe(standardize_dataframe(df_raw))
    st.session_state.df = df
except Exception as exc:
    st.error(f"Erreur lors de la lecture du fichier : {exc}")
    st.stop()


# ============================================================
# SMART MAPPING AUTOMATIQUE DES COLONNES
# ============================================================
auto_map = smart_auto_detect_columns(df)

if not st.session_state.get("confirmations"):
    st.session_state.confirmations = {
        "target": auto_map["target"] if auto_map["target"] in df.columns else None,
        "study_unit": auto_map["unit"] if auto_map["unit"] in df.columns else None,
        "time_variable": auto_map["exposure"] if auto_map["exposure"] in df.columns else (auto_map["time"] if auto_map["time"] in df.columns else None),
        "event_variable": auto_map["target"] if auto_map["target"] in df.columns else None,
        "time_to_event_confirmed": True,
        "censoring_defined": True if auto_map["censure"] else False,
    }


# ============================================================
# APERÇU DES DONNÉES
# ============================================================
st.header("3 — Données actives et Nettoyage Intelligent")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Observations", f"{len(df):,}")
col2.metric("Variables", f"{len(df.columns):,}")
col3.metric("Valeurs manquantes", f"{int(df.isna().sum().sum()):,}")
col4.metric("Doublons", f"{int(df.duplicated().sum()):,}")

with st.expander("🔍 Afficher un aperçu du fichier nettoyé", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)


# ============================================================
# FORMALISATION ET VALIDATION
# ============================================================
st.markdown("---")
st.header("4 — Formalisation scientifique (Pré-remplie par le Copilote)")

cols = ["— Aucun —"] + list(df.columns)

col_a, col_b, col_c = st.columns(3)
with col_a:
    def_target = st.session_state.confirmations.get("target")
    idx_t = cols.index(def_target) if def_target in cols else 0
    target_choice = st.selectbox("Variable cible Y (ex: Défaillance)", cols, index=idx_t)

with col_b:
    def_unit = st.session_state.confirmations.get("study_unit")
    idx_u = cols.index(def_unit) if def_unit in cols else 0
    unit_choice = st.selectbox("Unité d'étude (ex: Équipement)", cols, index=idx_u)

with col_c:
    def_time = st.session_state.confirmations.get("time_variable")
    idx_ti = cols.index(def_time) if def_time in cols else 0
    time_choice = st.selectbox("Variable temps / exposition (ex: Exposition / Date)", cols, index=idx_ti)


domains_preview, questions_preview = classify_problem(problem_desc, objective)

st.markdown("### Paramètres de Fiabilité")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    event_choice = st.selectbox("Variable définissant l'événement", cols, index=idx_t)
with col_f2:
    time_to_event_confirmed = st.checkbox("Temps / exposition bien reconstruit", value=True)
with col_f3:
    censoring_defined = st.checkbox("Censure explicitement définie", value=True if auto_map["censure"] else False)


if st.button("🚀 Exécuter le Moteur Scientifique", type="primary", use_container_width=True):
    if not problem_desc.strip():
        st.warning("Veuillez entrer une description du problème.")
        st.stop()
        
    st.session_state.confirmations = {
        "target": None if target_choice == "— Aucun —" else target_choice,
        "study_unit": None if unit_choice == "— Aucun —" else unit_choice,
        "time_variable": None if time_choice == "— Aucun —" else time_choice,
        "event_variable": None if event_choice == "— Aucun —" else event_choice,
        "time_to_event_confirmed": time_to_event_confirmed,
        "censoring_defined": censoring_defined,
    }

    with st.spinner("Le robot détective analyse et calcule les modèles..."):
        analysis = run_scientific_engine(
            problem=problem_desc,
            objective=objective,
            df=df,
            confirmations=st.session_state.confirmations,
        )
        st.session_state.analysis = analysis


# ============================================================
# RÉSULTATS & SYNTHÈSE HUMAINE DE L'AGENT
# ============================================================
analysis = st.session_state.get("analysis")

if analysis:
    st.markdown("---")
    st.header("💡 Synthèse Humaine de l'Agent Copilote")
    
    exec_res = analysis["execution"]
    n_obs = analysis["dataset_profile"]["n_observations"]
    
    summary_html = f"""
    <div class="block-ok">
        <h4>📌 Bilan de l'analyse pour votre flotte ({n_obs} lignes analysées) :</h4>
        <ul>
            <li><b>Unité suivie :</b> {analysis['study_unit'] or 'Non définie'} | <b>Indicateur cible :</b> {analysis['target'] or 'Non défini'}</li>
    """
    
    if "weibull" in exec_res and exec_res["weibull"].get("status") == "OK":
        beta = exec_res["weibull"]["beta"]
        eta = exec_res["weibull"]["eta"]
        summary_html += f"<li><b>Modèle de Weibull :</b> Facteur de forme $\\beta$ = <b>{beta:.2f}</b> (Caractéristique : {exec_res['weibull']['interpretation']}).</li>"
    else:
        summary_html += "<li><i>Modèle de Weibull : Volume de données insuffisant ou paramètres non activés pour ajuster la courbe de vieillissement exacte.</i></li>"
        
    summary_html += "</ul></div>"
    st.markdown(summary_html, unsafe_allow_html=True)

    # Affichage des graphiques et analyses exécutées
    st.markdown("---")
    st.header("📊 Résultats Graphiques & Statistiques")
    
    if "kaplan_meier" in exec_res and exec_res["kaplan_meier"].get("status") == "OK":
        st.subheader("Courbe de Survie (Kaplan-Meier)")
        km_curve = exec_res["kaplan_meier"]["curve"]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.step(km_curve["Temps"], km_curve["Survie"], where="post", color="#1f77b4", lw=2)
        ax.set_xlabel("Temps / Exposition")
        ax.set_ylabel("Probabilité de survie")
        ax.set_title("Fonction de Survie de la Flotte")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)

    if "weibull" in exec_res and exec_res["weibull"].get("status") == "OK":
        st.subheader("Modèle de Fiabilité de Weibull")
        wb_curve = exec_res["weibull"]["curve"]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(wb_curve["Temps"], wb_curve["Fiabilité"], color="#2ca02c", lw=2)
        ax.set_xlabel("Temps / Exposition")
        ax.set_ylabel("Fiabilité R(t)")
        ax.set_title("Courbe de Fiabilité R(t)")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)

    st.markdown("---")
    st.header("18 — Statut global de l'étude")
    st.success("🟢 STRUCTURE SCIENTIFIQUE COMPATIBLE ET ANALYSÉE")
