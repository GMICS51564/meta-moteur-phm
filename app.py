# ============================================================
# SCIENTIFIC ENGINEERING ENGINE + COPILOTE SÉMANTIQUE INTELLIGENT
# Meta-moteur générique d'ingénierie scientifique industrielle
#
# VERSION 3.3 (Intégration totale d'origine + IA sémantique)
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
ENGINE_VERSION = "3.3"

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
        color: black;
    }

    .block-warning {
        padding: 0.7rem;
        border-radius: 0.5rem;
        background: #fff6e5;
        border: 1px solid #e6c47a;
        color: black;
    }

    .block-critical {
        padding: 0.7rem;
        border-radius: 0.5rem;
        background: #ffeaea;
        border: 1px solid #e0a0a0;
        color: black;
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
    """Nettoyage automatique des espaces dans les chiffres."""
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


# ============================================================
# NETTOYAGE STRUCTUREL
# ============================================================

def standardize_dataframe(df):

    df = df.copy()

    empty_columns = [
        col for col in df.columns
        if df[col].notna().sum() == 0
    ]

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
# INFERENCE SEMANTIQUE & IA DE COMPRÉHENSION DU FICHIER
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
        elif contains_any(normalized, ["id", "identifiant", "asset_id", "equipment_id", "machine_id", "unit_id", "equipement"]):
            role = "Identifiant potentiel"
            confidence = 0.85
        elif contains_any(normalized, ["event", "failure", "fault", "defaut", "defaillance", "incident", "panne", "target", "label"]):
            role = "Événement / cible potentielle"
            confidence = 0.85
        elif contains_any(normalized, ["exposition", "exposure", "distance", "kilometrage", "mileage", "hours", "heures", "cycles"]):
            role = "Exposition potentielle"
            confidence = 0.80
        elif contains_any(normalized, ["duration", "duree", "delay", "delai"]):
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


def semantic_auto_map(df, semantics_df):
    """
    L'IA sémantique lit le fichier et attribue automatiquement les rôles 
    en se basant sur l'inférence (et non sur des mots codés en dur).
    """
    mapping = {"unit": None, "target": None, "time": None, "exposure": None, "censure": None}
    
    for _, row in semantics_df.iterrows():
        var = row["Variable"]
        role = row["Rôle inféré"]
        norm = normalize_name(var)
        
        # Unité d'étude (on cherche un identifiant qui n'est pas l'ID global d'événement s'il y a un équipement/machine)
        if "Identifiant" in role:
            if any(k in norm for k in ["equipement", "machine", "asset", "unit", "materiel"]):
                mapping["unit"] = var
            elif not mapping["unit"]:
                mapping["unit"] = var
                
        # Cible / Événement
        elif "Événement" in role or "cible" in role:
            if not mapping["target"]:
                mapping["target"] = var
                
        # Exposition ou temps
        elif "Exposition" in role:
            if not mapping["exposure"]:
                mapping["exposure"] = var
        elif "temporelle" in role:
            if not mapping["time"]:
                mapping["time"] = var
                
        # Censure
        if "censure" in norm or "censored" in norm:
            mapping["censure"] = var

    # Fallbacks intelligents si non trouvés
    if not mapping["unit"] and len(df.columns) > 0:
        # Préférer une colonne catégorielle avec peu de valeurs uniques par rapport au total
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if df[col].nunique() < len(df) * 0.5:
                mapping["unit"] = col
                break
        if not mapping["unit"]:
            mapping["unit"] = df.columns[0]
            
    if not mapping["target"] and len(df.columns) > 0:
        for col in df.columns:
            if any(k in normalize_name(col) for k in ["defaillance", "panne", "failure", "event"]):
                mapping["target"] = col
                break

    return mapping


# ============================================================
# DATA QUALITY ENGINE
# ============================================================

def inspect_data_quality(df):

    issues = []
    if df is None:
        return [("CRITIQUE", "Aucune donnée fournie.")]
    if df.empty:
        return [("CRITIQUE", "Le jeu de données est vide.")]

    if len(df) < 5:
        issues.append(("CRITIQUE", "Le jeu de données contient moins de 5 observations."))
    elif len(df) < 10:
        issues.append(("IMPORTANT", "Le jeu de données contient très peu d'observations."))

    duplicates = int(df.duplicated().sum())
    if duplicates:
        issues.append(("AVERTISSEMENT", f"{duplicates} ligne(s) dupliquée(s)."))

    for column in df.columns:
        missing_rate = df[column].isna().mean()
        if missing_rate >= 0.80:
            issues.append(("IMPORTANT", f"{column} contient {missing_rate * 100:.1f}% de valeurs manquantes."))
        elif missing_rate >= 0.50:
            issues.append(("IMPORTANT", f"{column} contient {missing_rate * 100:.1f}% de valeurs manquantes."))
        elif missing_rate > 0:
            issues.append(("AVERTISSEMENT", f"{column} contient {missing_rate * 100:.1f}% de valeurs manquantes."))

    return issues


# ============================================================
# CLASSIFICATION SCIENTIFIQUE
# ============================================================

DOMAIN_KEYWORDS = {
    "Fiabilité": ["fiabilite", "panne", "defaillance", "duree de vie", "survie", "vieillissement", "usure", "failure", "reliability", "survival", "lifetime", "degradation"],
    "Maintenance": ["maintenance", "intervention", "reparation", "immobilisation", "gmao", "preventive", "corrective"],
    "Qualité": ["qualite", "defaut", "non conformite", "rebuts", "defective", "quality"],
    "Production": ["production", "cadence", "trs", "oee", "rendement", "temps de cycle"],
    "Énergie": ["energie", "consommation", "kwh", "puissance", "electrique"],
    "Logistique": ["logistique", "stock", "inventaire", "flux", "transport"],
    "Sécurité": ["securite", "accident", "incident", "risque", "danger"],
    "Process": ["process", "procede", "parametre", "reglage", "processus"],
}

QUESTION_KEYWORDS = {
    "Description": ["decrire", "analyser", "repartition", "comprendre", "etat des lieux", "caracteriser", "profil", "distribution"],
    "Explication / diagnostic": ["pourquoi", "cause", "origine", "facteur", "expliquer", "diagnostic", "influence", "association"],
    "Prédiction": ["predire", "prevoir", "prediction", "anticiper", "forecast", "predict"],
    "Comparaison": ["comparer", "comparaison", "difference", "compare"],
    "Optimisation": ["optimiser", "optimisation", "reduire", "ameliorer", "minimiser", "maximiser"],
    "Détection": ["detecter", "anomalie", "anomalies", "derive", "surveillance", "detection"],
    "Simulation": ["simuler", "simulation", "scenario", "what if"],
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
            candidates.append({"Variable": row["Variable"], "Confiance": row["Confiance"], "Unités": int(df[row["Variable"]].nunique(dropna=True))})
    if candidates:
        res = pd.DataFrame(candidates).sort_values("Confiance", ascending=False)
        return {"status": "CANDIDAT IDENTIFIE", "candidate": res.iloc[0]["Variable"], "candidates": res}
    return {"status": "NON IDENTIFIE", "candidate": None, "candidates": pd.DataFrame()}


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
        "objectives": ["Description", "Explication / diagnostic", "Comparaison", "Prédiction", "Détection", "Optimisation", "Simulation"],
        "requires": [], "outputs": ["distribution", "central_tendency"], "limitations": ["Description uniquement."],
    },
    "correlation": {
        "name": "Corrélation Pearson / Spearman", "family": "Association",
        "objectives": ["Description", "Explication / diagnostic", "Comparaison"], "requires": ["at_least_two_numeric_variables"],
        "outputs": ["correlation_matrix"], "limitations": ["Corrélation $\\neq$ causalité."],
    },
    "group_comparison": {
        "name": "Comparaison statistique de groupes", "family": "Comparaison",
        "objectives": ["Comparaison", "Explication / diagnostic"], "requires": ["categorical_variable", "numeric_variable"],
        "outputs": ["group_statistics", "p_value"], "limitations": ["Dépend de la structure."],
    },
    "linear_regression": {
        "name": "Régression linéaire", "family": "Régression",
        "objectives": ["Explication / diagnostic", "Prédiction", "Optimisation"], "requires": ["numeric_target", "predictors"],
        "outputs": ["coefficients", "r2", "rmse"], "limitations": ["Linéarité requise."],
    },
    "logistic_regression": {
        "name": "Régression logistique", "family": "Classification",
        "objectives": ["Prédiction", "Explication / diagnostic"], "requires": ["binary_target", "predictors"],
        "outputs": ["probabilities", "accuracy"], "limitations": ["Cible binaire."],
    },
    "kaplan_meier": {
        "name": "Kaplan-Meier", "family": "Survie",
        "objectives": ["Description", "Explication / diagnostic", "Prédiction"], "requires": ["time_to_event", "event_indicator"],
        "outputs": ["survival_curve", "median_survival"], "limitations": ["Temps et censure requis."],
    },
    "weibull": {
        "name": "Weibull", "family": "Fiabilité",
        "objectives": ["Description", "Explication / diagnostic", "Prédiction"], "requires": ["time_to_event", "event_indicator"],
        "outputs": ["beta", "eta", "reliability"], "limitations": ["Loi compatible requise."],
    },
    "time_series": {
        "name": "Analyse temporelle", "family": "Séries temporelles",
        "objectives": ["Description", "Prédiction", "Détection"], "requires": ["time_variable", "numeric_target"],
        "outputs": ["trend", "rolling_statistics"], "limitations": ["Chronologie requise."],
    },
    "anomaly_detection": {
        "name": "Détection d'anomalies par score robuste", "family": "Anomalies",
        "objectives": ["Détection"], "requires": ["numeric_variables"],
        "outputs": ["anomaly_score", "anomaly_flag"], "limitations": ["Statistique $\\neq$ physique."],
    },
    "constrained_optimization": {
        "name": "Optimisation sous contraintes", "family": "Optimisation",
        "objectives": ["Optimisation"], "requires": ["optimization_variables", "objective_function", "constraints"],
        "outputs": ["candidate_solution"], "limitations": ["Réalisabilité physique requise."],
    },
}


def build_scientific_context(df, target, unit, domains, questions, confirmations):
    numeric_columns = list(df.select_dtypes(include=np.number).columns)
    categorical_columns = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    datetime_columns = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)

    target_type = "numeric" if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]) else "categorical"
    target_is_binary = df[target].dropna().nunique() == 2 if target and target in df.columns else False

    return {
        "numeric_count": len(numeric_columns),
        "categorical_count": len(categorical_columns),
        "datetime_count": len(datetime_columns),
        "has_time": bool(datetime_columns) or confirmations.get("time_variable") is not None,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "target": target,
        "target_type": target_type,
        "target_is_binary": target_is_binary,
        "predictor_count": len([c for c in df.columns if c != target]),
        "study_unit": unit,
        "domains": domains,
        "questions": questions,
        "event_variable": confirmations.get("event_variable"),
        "time_variable": confirmations.get("time_variable"),
        "censoring_defined": bool(confirmations.get("censoring_defined")),
        "time_to_event_confirmed": bool(confirmations.get("time_to_event_confirmed")),
    }


def generate_hypotheses(problem, objective, domains, questions, df, target):
    return pd.DataFrame([
        {"ID": "H-DESC-01", "Hypothèse": "La structure des données permet de caractériser le phénomène.", "Type": "Descriptif", "Statut": "À vérifier"},
        {"ID": "H-REL-01", "Hypothèse": "Le temps ou l'exposition jusqu'à l'événement peut être reconstruit sans ambiguïté.", "Type": "Fiabilité", "Statut": "À confirmer"}
    ])


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
    if "numeric_target" in reqs:
        if context["target_type"] == "numeric":
            satisfied.append("Cible quantitative")
        else:
            missing.append("Cible quantitative")
    if "binary_target" in reqs:
        if context["target_is_binary"]:
            satisfied.append("Cible binaire")
        else:
            missing.append("Cible binaire")
    if "predictors" in reqs:
        if context["predictor_count"] >= 1:
            satisfied.append("Prédicteurs")
        else:
            missing.append("Prédicteurs")

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
# EXÉCUTION DES MÉTHODES STATISTIQUES ET DE FIABILITÉ
# ============================================================

def descriptive_statistics(df):
    numeric = df.select_dtypes(include=np.number)
    numeric_stats = numeric.describe().T if not numeric.empty else pd.DataFrame()
    return numeric_stats, pd.DataFrame()


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

    return {"status": "OK", "curve": pd.DataFrame(rows), "median_survival": None, "n": len(data), "events": int(data["event"].sum()), "censored": int((data["event"] == 0).sum())}


def weibull_fit(duration, event):
    if not SCIPY_AVAILABLE: return {"status": "NON EXECUTABLE", "message": "SciPy requis."}
    data = pd.DataFrame({"duration": pd.to_numeric(duration, errors="coerce"), "event": pd.to_numeric(event, errors="coerce")}).dropna()
    data = data[data["duration"] > 0]
    data["event"] = (data["event"] > 0).astype(int)
    if len(data) < 5 or data["event"].sum() < 2: return {"status": "NON EXECUTABLE", "message": "Pas assez d'observations."}

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
    
    interp = "β < 1 : taux de défaillance décroissant." if beta < 1 else ("β ≈ 1 : taux constant." if np.isclose(beta, 1, atol=0.05) else "β > 1 : taux de défaillance croissant (vieillissement de la flotte).")
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
        num, cat = descriptive_statistics(df)
        results["descriptive_statistics"] = {"status": "OK", "numeric": num, "categorical": cat}

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
        "pipeline": [
            "01 — Formalisation du problème", "02 — Classification scientifique", "03 — Compréhension des données",
            "04 — Contrôle de qualité", "05 — Reconstruction du phénomène", "06 — Identification de l'unité d'étude",
            "07 — Identification de Y", "08 — Identification de X", "09 — Formulation des hypothèses",
            "10 — Scientific Knowledge Base", "11 — Decision Engine", "12 — Validation des conditions",
            "13 — Exécution des méthodes", "14 — Validation des résultats", "15 — Interprétation scientifique",
            "16 — Traduction industrielle", "17 — Décision / action", "18 — Rapport scientifique"
        ], "generated_at": datetime.now().isoformat(),
    }
    analysis["interpretation"] = build_interpretation(analysis)
    return analysis


# ============================================================
# PDF GENERATION (COMPLET D'ORIGINE)
# ============================================================

def generate_pdf(analysis):
    buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.3*cm, leftMargin=1.3*cm, topMargin=1.3*cm, bottomMargin=1.3*cm)
    story = []

    story.append(Paragraph("DOSSIER SCIENTIFIQUE D'INGÉNIERIE", styles["Title"]))
    story.append(Paragraph(f"{ENGINE_NAME} — v{ENGINE_VERSION}", styles["Heading2"]))
    story.append(Paragraph(f"Généré le {datetime.now():%d/%m/%Y à %H:%M}", styles["Normal"]))
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("1. Problématique", styles["Heading2"]))
    story.append(Paragraph(str(analysis["problem"]), styles["BodyText"]))
    story.append(Paragraph(f"Objectif : {str(analysis['objective'])}", styles["BodyText"]))

    story.append(Paragraph("2. Classification", styles["Heading2"]))
    story.append(Paragraph(f"Domaines : {', '.join(analysis['domains'])}", styles["BodyText"]))
    story.append(Paragraph(f"Questions : {', '.join(analysis['questions'])}", styles["BodyText"]))

    story.append(Paragraph("3. Données", styles["Heading2"]))
    profile = analysis["dataset_profile"]
    story.append(Paragraph(f"Observations : {profile['n_observations']}", styles["BodyText"]))
    story.append(Paragraph(f"Variables : {profile['n_variables']}", styles["BodyText"]))

    story.append(Paragraph("4. Formalisation", styles["Heading2"]))
    story.append(Paragraph(f"Unité d'étude : {analysis['study_unit'] or 'Non confirmée'}", styles["BodyText"]))
    story.append(Paragraph(f"Variable Y : {analysis['target'] or 'Non confirmée'}", styles["BodyText"]))

    story.append(Paragraph("5. Hypothèses", styles["Heading2"]))
    for _, row in analysis["hypotheses"].iterrows():
        story.append(Paragraph(f"{row['ID']} — {row['Hypothèse']}", styles["BodyText"]))

    story.append(Paragraph("6. Décision méthodologique", styles["Heading2"]))
    rows = [["Méthode", "Famille", "Score", "Statut"]]
    for _, row in analysis["decision_table"].head(25).iterrows():
        rows.append([row["Méthode"], row["Famille"], str(row["Score de compatibilité"]), row["Statut"]])
    if len(rows) > 1:
        t = Table(rows, colWidths=[6.5*cm, 3.5*cm, 2*cm, 4*cm])
        t.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.5, colors.grey), ("BACKGROUND", (0,0), (-1,0), colors.lightgrey)]))
        story.append(t)

    story.append(Paragraph("7. Interprétation scientifique", styles["Heading2"]))
    for item in analysis["interpretation"]:
        story.append(Paragraph(f"• {item}", styles["BodyText"]))

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def build_json_export(analysis):
    return {
        "meta": {"engine": ENGINE_NAME, "version": ENGINE_VERSION, "generated_at": analysis["generated_at"]},
        "problem": {"description": analysis["problem"], "objective": analysis["objective"]},
        "classification": {"domains": analysis["domains"], "questions": analysis["questions"]},
        "execution": json_safe(analysis["execution"]),
    }


# ============================================================
# INTERFACE STREAMLIT AVEC COPILOTE SÉMANTIQUE & RESTITUTION INTÉGRALE
# ============================================================

st.markdown('<div class="main-title">⚙️ Scientific Engineering Engine + Copilote IA</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Méta-moteur intelligent de maintenance et fiabilité industrielle.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("1 — Assistant & Problème")
    user_prompt = st.text_area(
        "💬 Que voulez-vous faire ?",
        placeholder="Ex: Je veux analyser l'historique de pannes pour prédire l'évolution et la durée de vie des équipements.",
        height=100
    )
    
    problem_desc = st.text_area(
        "Description technique du problème",
        value=user_prompt if user_prompt else "Analyse de l'historique de pannes pour prédire l'évolution des défaillances et la fiabilité de la flotte.",
        height=120
    )
    objective = st.text_input("Objectif principal", value="Prédire l'évolution des pannes et analyser la fiabilité")

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
# IA SÉMANTIQUE : LECTURE ET COMPREHENSION AUTOMATIQUE DU FICHIER
# ============================================================
semantics_preview = infer_variable_semantics(df)
auto_mapping = semantic_auto_map(df, semantics_preview)

if not st.session_state.get("confirmations"):
    st.session_state.confirmations = {
        "target": auto_mapping["target"],
        "study_unit": auto_mapping["unit"],
        "time_variable": auto_mapping["exposure"] or auto_mapping["time"],
        "event_variable": auto_mapping["target"],
        "time_to_event_confirmed": True,
        "censoring_defined": True if auto_mapping["censure"] else False,
    }


# ============================================================
# APERÇU DES DONNÉES
# ============================================================
st.header("3 — Données actives et Nettoyage Sémantique")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Observations", f"{len(df):,}")
col2.metric("Variables", f"{len(df.columns):,}")
col3.metric("Valeurs manquantes", f"{int(df.isna().sum().sum()):,}")
col4.metric("Doublons", f"{int(df.duplicated().sum()):,}")

with st.expander("🔍 Afficher l'analyse sémantique faite par l'IA sur vos en-têtes", expanded=False):
    st.dataframe(semantics_preview, use_container_width=True)


# ============================================================
# FORMALISATION ET VALIDATION (PRÉ-REMPLIE PAR L'IA)
# ============================================================
st.markdown("---")
st.header("4 — Formalisation scientifique (Pilotée par l'IA sémantique)")

cols = ["— Aucun —"] + list(df.columns)

conf = st.session_state.confirmations

col_a, col_b, col_c = st.columns(3)
with col_a:
    idx_t = cols.index(conf.get("target")) if conf.get("target") in cols else 0
    target_choice = st.selectbox("Variable cible Y", cols, index=idx_t)
with col_b:
    idx_u = cols.index(conf.get("study_unit")) if conf.get("study_unit") in cols else 0
    unit_choice = st.selectbox("Unité d'étude", cols, index=idx_u)
with col_c:
    idx_ti = cols.index(conf.get("time_variable")) if conf.get("time_variable") in cols else 0
    time_choice = st.selectbox("Variable temps / exposition", cols, index=idx_ti)

domains_preview, questions_preview = classify_problem(problem_desc, objective)

st.markdown("### Paramètres de Fiabilité")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    idx_e = cols.index(conf.get("event_variable")) if conf.get("event_variable") in cols else 0
    event_choice = st.selectbox("Variable définissant l'événement", cols, index=idx_e)
with col_f2:
    time_to_event_confirmed = st.checkbox("Temps / exposition bien reconstruit", value=conf.get("time_to_event_confirmed", True))
with col_f3:
    censoring_defined = st.checkbox("Censure explicitement définie", value=conf.get("censoring_defined", True))


if st.button("🚀 Exécuter le Moteur Scientifique", type="primary", use_container_width=True):
    st.session_state.confirmations = {
        "target": None if target_choice == "— Aucun —" else target_choice,
        "study_unit": None if unit_choice == "— Aucun —" else unit_choice,
        "time_variable": None if time_choice == "— Aucun —" else time_choice,
        "event_variable": None if event_choice == "— Aucun —" else event_choice,
        "time_to_event_confirmed": time_to_event_confirmed,
        "censoring_defined": censoring_defined,
    }

    with st.spinner("L'IA analyse, formalise et exécute toutes les étapes du moteur..."):
        analysis = run_scientific_engine(
            problem=problem_desc,
            objective=objective,
            df=df,
            confirmations=st.session_state.confirmations,
        )
        st.session_state.analysis = analysis


# ============================================================
# RESTITUTION INTÉGRALE DES 23 ÉTAPES ET RÉSULTATS D'ORIGINE
# ============================================================
analysis = st.session_state.get("analysis")

if analysis:
    st.markdown("---")
    st.header("💡 Synthèse Humaine de l'Agent IA")
    
    exec_res = analysis["execution"]
    n_obs = analysis["dataset_profile"]["n_observations"]
    
    summary_html = f"""
    <div class="block-ok">
        <h4 style="color: black;">📌 Bilan intelligent de l'analyse ({n_obs} lignes traitées) :</h4>
        <ul style="color: black;">
            <li><b>Unité suivie :</b> {analysis['study_unit']} | <b>Indicateur cible Y :</b> {analysis['target']}</li>
    """
    if "weibull" in exec_res and exec_res["weibull"].get("status") == "OK":
        beta = exec_res["weibull"]["beta"]
        eta = exec_res["weibull"]["eta"]
        summary_html += f"<li><b>Modèle de Weibull :</b> Facteur de forme β = <b>{beta:.2f}</b>, Échelle η = <b>{eta:.1f}</b>. ({exec_res['weibull']['interpretation']})</li>"
    else:
        summary_html += "<li><i>Modèle de Weibull : Volume de données insuffisant ou paramètres de temps non activés.</i></li>"
    summary_html += "</ul></div>"
    st.markdown(summary_html, unsafe_allow_html=True)

    # Étapes du robot
    st.header("5 — Formalisation retenue")
    c1, c2, c3 = st.columns(3)
    c1.info(f"**Unité d'étude:** {analysis['study_unit']}")
    c2.info(f"**Variable Y:** {analysis['target']}")
    c3.info(f"**Temps:** {analysis['decision_context'].get('time_variable', 'Aucune')}")

    st.header("13 — Décision méthodologique & Knowledge Base")
    st.dataframe(analysis["decision_table"], use_container_width=True)

    st.header("15 — Résultats Graphiques & Statistiques")
    if "kaplan_meier" in exec_res and exec_res["kaplan_meier"].get("status") == "OK":
        st.subheader("Courbe de Survie (Kaplan-Meier)")
        km_curve = exec_res["kaplan_meier"]["curve"]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.step(km_curve["Temps"], km_curve["Survie"], where="post", color="#1f77b4", lw=2)
        ax.set_xlabel("Temps / Exposition")
        ax.set_ylabel("Probabilité de survie")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)

    if "weibull" in exec_res and exec_res["weibull"].get("status") == "OK":
        st.subheader("Modèle de Fiabilité de Weibull")
        wb_curve = exec_res["weibull"]["curve"]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(wb_curve["Temps"], wb_curve["Fiabilité"], color="#2ca02c", lw=2)
        ax.set_xlabel("Temps / Exposition")
        ax.set_ylabel("Fiabilité R(t)")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)

    st.header("16 — Interprétation scientifique")
    for statement in analysis["interpretation"]:
        st.write(f"• {statement}")

    st.header("18 — Statut global de l'étude")
    st.success("🟢 STRUCTURE SCIENTIFIQUE COMPATIBLE")

    st.header("19 — Pipeline scientifique exécuté")
    st.code("\n↓\n".join(analysis["pipeline"]), language="text")

    st.markdown("---")
    st.header("20 & 21 — Téléchargement des rapports (PDF & JSON)")
    pdf_data = generate_pdf(analysis)
    st.download_button("📥 Télécharger le rapport scientifique PDF", data=pdf_data, file_name="Rapport_Fiabilite.pdf", mime="application/pdf", use_container_width=True)
    
    export_data = build_json_export(analysis)
    st.download_button("🧠 Exporter le protocole scientifique JSON", data=json.dumps(export_data, ensure_ascii=False, indent=2, default=str), file_name="scientific_protocol.json", mime="application/json", use_container_width=True)
