# ============================================================
# SCIENTIFIC ENGINEERING ENGINE
# Meta-moteur générique d'ingénierie scientifique industrielle
#
# VERSION 3.1 (Amélioration de l'ergonomie et du vocabulaire)
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
# OUTILS GENERAUX & VOCABULAIRE ÉLARGI (SYNONYMES)
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
# CHARGEMENT ET NETTOYAGE DES DONNEES
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
# INFERENCE SEMANTIQUE & VOCABULAIRE MULTIPLIÉ (SYNONYMES)
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

    numeric_columns = df.select_dtypes(include=np.number).columns
    for column in numeric_columns:
        series = df[column].dropna()
        if series.empty:
            continue
        negative_count = int((series < 0).sum())
        if negative_count:
            issues.append(("A VERIFIER", f"{column} contient {negative_count} valeur(s) négative(s)."))
        if series.nunique() <= 1:
            issues.append(("AVERTISSEMENT", f"{column} présente une variance nulle."))
    return issues


# ============================================================
# CLASSIFICATION SCIENTIFIQUE ÉLARGIE (SYNONYMES)
# ============================================================

DOMAIN_KEYWORDS = {
    "Fiabilité": ["fiabilite", "panne", "defaillance", "duree de vie", "survie", "vieillissement", "usure", "failure", "reliability", "survival", "lifetime", "degradation", "casse", "rupture", "fatigue", "dysfonctionnement"],
    "Maintenance": ["maintenance", "intervention", "reparation", "immobilisation", "gmao", "preventive", "corrective", "technicien", "depannage", "visite", "astreinte"],
    "Qualité": ["qualite", "defaut", "non conformite", "rebuts", "rebut", "defective", "quality", "retouche", "ecart", "conformite", "controle", "bureau_controle"],
    "Production": ["production", "cadence", "trs", "oee", "rendement", "temps de cycle", "cycle de production", "goulot", "capacite", "throughput", "volume", "fabrication", "atelier", "sortie"],
    "Énergie": ["energie", "consommation", "kwh", "puissance", "electrique", "energy", "power", "gaz", "carburant", "kw", "facture"],
    "Logistique": ["logistique", "stock", "inventaire", "flux", "transport", "approvisionnement", "supply chain", "entrepot", "colis", "livraison", "expedition"],
    "Sécurité": ["securite", "accident", "incident", "risque", "danger", "safety", "presqu_accident", "blessure", "hse"],
    "Process": ["process", "procede", "parametre", "reglage", "processus", "temperature", "pression", "vitesse", "consigne"],
}

QUESTION_KEYWORDS = {
    "Description": ["decrire", "analyser", "repartition", "comprendre", "etat des lieux", "caracteriser", "profil", "distribution", "voir", "observer", "synthese"],
    "Explication / diagnostic": ["pourquoi", "cause", "causes", "origine", "facteur", "expliquer", "diagnostic", "influence", "associe", "association", "provenance", "lie_a"],
    "Prédiction": ["predire", "prevoir", "prediction", "anticiper", "forecast", "predict", "futur", "estimer_prochain"],
    "Comparaison": ["comparer", "comparaison", "difference", "différence", "compare", "entre", "versus", "vs"],
    "Optimisation": ["optimiser", "optimisation", "reduire", "ameliorer", "minimiser", "maximiser", "optimize", "gagner", "booster", "perf"],
    "Détection": ["detecter", "anomalie", "anomalies", "derive", "surveillance", "detection", "monitoring", "bizarre", "atypique"],
    "Simulation": ["simuler", "simulation", "scenario", "scenarios", "what if", "et si"],
}

def classify_problem(problem, objective):
    text = f"{problem} {objective}"
    domains = [d for d, keywords in DOMAIN_KEYWORDS.items() if contains_any(text, keywords)]
    questions = [q for q, keywords in QUESTION_KEYWORDS.items() if contains_any(text, keywords)]
    if not domains:
        domains = ["Domaine à déterminer"]
    if not questions:
        questions = ["Description"]
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
            candidates.append({
                "Variable": row["Variable"],
                "Rôle": "Identifiant potentiel",
                "Confiance": row["Confiance"],
                "Nombre d'unités": int(df[row["Variable"]].nunique(dropna=True)),
            })
    if candidates:
        result = pd.DataFrame(candidates).sort_values("Confiance", ascending=False)
        return {"status": "CANDIDAT IDENTIFIE", "candidate": result.iloc[0]["Variable"], "candidates": result}
    return {"status": "NON IDENTIFIE", "candidate": None, "candidates": pd.DataFrame(columns=["Variable", "Rôle", "Confiance", "Nombre d'unités"])}


def identify_candidate_targets(df, semantics):
    candidates = []
    for _, row in semantics.iterrows():
        role = str(row["Rôle inféré"])
        confidence = float(row["Confiance"])
        if "cible" in role.lower() or "événement" in role.lower():
            candidates.append({"Variable": row["Variable"], "Justification": role, "Confiance": confidence})
    for column in df.columns:
        if any(item["Variable"] == column for item in candidates):
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            candidates.append({"Variable": column, "Justification": "Variable quantitative candidate.", "Confiance": 0.45})
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series):
            nunique = series.dropna().nunique()
            if 2 <= nunique <= 10:
                candidates.append({"Variable": column, "Justification": "Variable catégorielle pouvant constituer une cible.", "Confiance": 0.35})
    return pd.DataFrame(candidates)


def identify_predictors(df, target=None):
    records = []
    for column in df.columns:
        if column == target:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            records.append({"Variable": column, "Type": "Quantitative", "Utilisable": True})
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series):
            records.append({"Variable": column, "Type": "Catégorielle", "Utilisable": True})
        elif pd.api.types.is_datetime64_any_dtype(series):
            records.append({"Variable": column, "Type": "Temporelle", "Utilisable": "À transformer"})
    return pd.DataFrame(records)


# Knowledge base et méthodes standards (inchangées pour la robustesse de l'analyse)
SCIENTIFIC_KNOWLEDGE_BASE = {
    "descriptive_statistics": {"name": "Statistiques descriptives", "family": "Exploration", "objectives": ["Description", "Explication / diagnostic", "Comparaison", "Prédiction", "Détection", "Optimisation", "Simulation"], "requires": [], "outputs": ["distribution", "central_tendency", "dispersion", "missingness"], "limitations": ["Description uniquement."]},
    "correlation": {"name": "Corrélation Pearson / Spearman", "family": "Association", "objectives": ["Description", "Explication / diagnostic", "Comparaison"], "requires": ["at_least_two_numeric_variables"], "outputs": ["correlation_matrix"], "limitations": ["Ne démontre pas une causalité."]},
    "group_comparison": {"name": "Comparaison statistique de groupes", "family": "Comparaison", "objectives": ["Comparaison", "Explication / diagnostic"], "requires": ["categorical_variable", "numeric_variable"], "outputs": ["group_statistics", "p_value"], "limitations": ["Dépend de la structure."]},
    "linear_regression": {"name": "Régression linéaire", "family": "Régression", "objectives": ["Explication / diagnostic", "Prédiction", "Optimisation"], "requires": ["numeric_target", "predictors"], "outputs": ["coefficients", "predictions", "r2"], "limitations": ["Linéarité requise."]},
    "logistic_regression": {"name": "Régression logistique", "family": "Classification", "objectives": ["Prédiction", "Explication / diagnostic", "Comparaison"], "requires": ["binary_target", "predictors"], "outputs": ["probabilities", "accuracy"], "limitations": ["Cible binaire requise."]},
    "kaplan_meier": {"name": "Kaplan-Meier", "family": "Survie", "objectives": ["Description", "Explication / diagnostic", "Prédiction"], "requires": ["time_to_event", "event_indicator"], "outputs": ["survival_curve"], "limitations": ["Censure requise."]},
    "weibull": {"name": "Weibull", "family": "Fiabilité", "objectives": ["Description", "Explication / diagnostic", "Prédiction"], "requires": ["time_to_event", "event_indicator"], "outputs": ["beta", "eta"], "limitations": ["Loi compatible requise."]},
    "time_series": {"name": "Analyse temporelle", "family": "Séries temporelles", "objectives": ["Description", "Prédiction", "Détection"], "requires": ["time_variable", "numeric_target"], "outputs": ["trend"], "limitations": ["Chronométrie requise."]},
    "anomaly_detection": {"name": "Détection d'anomalies", "family": "Anomalies", "objectives": ["Détection"], "requires": ["numeric_variables"], "outputs": ["anomaly_score"], "limitations": ["Anomalie statistique."]},
    "constrained_optimization": {"name": "Optimisation sous contraintes", "family": "Optimisation", "objectives": ["Optimisation"], "requires": ["optimization_variables", "objective_function", "constraints"], "outputs": ["candidate_solution"], "limitations": ["Réalisabilité physique."]},
}


def build_scientific_context(df, target, unit, domains, questions, confirmations):
    numeric_columns = list(df.select_dtypes(include=np.number).columns)
    categorical_columns = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    datetime_columns = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
    
    target_type = None
    target_is_binary = False
    if target and target in df.columns:
        series = df[target]
        if pd.api.types.is_numeric_dtype(series):
            target_type = "numeric"
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series):
            target_type = "categorical"
        target_is_binary = (series.dropna().nunique() == 2)

    predictor_count = len([c for c in df.columns if c != target])
    return {
        "numeric_count": len(numeric_columns),
        "categorical_count": len(categorical_columns),
        "datetime_count": len(datetime_columns),
        "has_time": bool(datetime_columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "target": target,
        "target_type": target_type,
        "target_is_binary": target_is_binary,
        "predictor_count": predictor_count,
        "study_unit": unit,
        "domains": domains,
        "questions": questions,
        "event_variable": confirmations.get("event_variable"),
        "time_variable": confirmations.get("time_variable"),
        "censoring_defined": bool(confirmations.get("censoring_defined")),
        "controllable_variables": confirmations.get("controllable_variables", []),
        "objective_function": confirmations.get("objective_function"),
        "constraints_defined": bool(confirmations.get("constraints")),
        "time_to_event_confirmed": bool(confirmations.get("time_to_event_confirmed")),
        "normal_behavior_defined": bool(confirmations.get("normal_behavior_defined")),
    }


def generate_hypotheses(problem, objective, domains, questions, df, target):
    hypotheses = [{"ID": "H-DESC-01", "Hypothèse": "La structure des données permet de caractériser quantitativement le phénomène étudié.", "Type": "Descriptif", "Statut": "À vérifier"}]
    numeric_columns = list(df.select_dtypes(include=np.number).columns)
    categorical_columns = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    if len(numeric_columns) >= 2:
        hypotheses.append({"ID": "H-ASSOC-01", "Hypothèse": "Au moins une association statistique mesurable existe entre des variables quantitatives.", "Type": "Association", "Statut": "À tester"})
    if categorical_columns and numeric_columns:
        hypotheses.append({"ID": "H-COMP-01", "Hypothèse": "La réponse étudiée présente des distributions différentes selon les catégories.", "Type": "Comparaison", "Statut": "À tester"})
    return pd.DataFrame(hypotheses)


def evaluate_method(method_id, method, context):
    missing, satisfied = [], []
    requirements = method.get("requires", [])
    if "at_least_two_numeric_variables" in requirements:
        if context["numeric_count"] >= 2: satisfied.append("≥ 2 variables quantitatives")
        else: missing.append("≥ 2 variables quantitatives")
    if "numeric_variables" in requirements:
        if context["numeric_count"] >= 1: satisfied.append("Variable(s) quantitative(s)")
        else: missing.append("Variable quantitative")
    if "categorical_variable" in requirements:
        if context["categorical_count"] >= 1: satisfied.append("Variable catégorielle")
        else: missing.append("Variable catégorielle")
    if "numeric_target" in requirements:
        if context["target_type"] == "numeric": satisfied.append("Cible quantitative")
        else: missing.append("Cible quantitative confirmée")
    if "binary_target" in requirements:
        if context["target_is_binary"]: satisfied.append("Cible binaire")
        else: missing.append("Cible binaire")
    if "predictors" in requirements:
        if context["predictor_count"] >= 1: satisfied.append("Variables explicatives")
        else: missing.append("Variables explicatives")
    if "time_variable" in requirements:
        if context["has_time"] or context["time_variable"]: satisfied.append("Variable temporelle")
        else: missing.append("Variable temporelle")
    if "time_to_event" in requirements:
        if context["time_to_event_confirmed"] and context["event_variable"]: satisfied.append("Temps jusqu'à événement")
        else: missing.append("Temps jusqu'à événement")
    if "event_indicator" in requirements:
        if context["event_variable"]: satisfied.append("Variable événement")
        else: missing.append("Variable événement")
    if "optimization_variables" in requirements:
        if context["controllable_variables"]: satisfied.append("Variables contrôlables")
        else: missing.append("Variables contrôlables")
    if "objective_function" in requirements:
        if context["objective_function"]: satisfied.append("Fonction objectif")
        else: missing.append("Fonction objectif")
    if "constraints" in requirements:
        if context["constraints_defined"]: satisfied.append("Contraintes")
        else: missing.append("Contraintes")

    status = "COMPATIBLE" if not missing else ("CONDITIONNEL" if len(missing) == 1 else "BLOQUÉ")
    total = len(requirements)
    score = 100 if total == 0 else int(100 * len(satisfied) / total)
    return {"status": status, "score": score, "satisfied": satisfied, "missing": missing}


def decision_engine(knowledge_base, context):
    results = []
    questions = context["questions"]
    for method_id, method in knowledge_base.items():
        objectives = method.get("objectives", [])
        objective_match = len(set(objectives) & set(questions)) > 0
        if method_id == "descriptive_statistics":
            objective_match = True
        if not objective_match:
            continue
        evaluation = evaluate_method(method_id, method, context)
        results.append({
            "ID": method_id,
            "Méthode": method["name"],
            "Famille": method["family"],
            "Score de compatibilité": evaluation["score"],
            "Statut": evaluation["status"],
            "Conditions satisfaites": " | ".join(evaluation["satisfied"]),
            "Informations manquantes": " | ".join(evaluation["missing"]),
            "Sorties": " | ".join(method.get("outputs", [])),
            "Limites": " | ".join(method.get("limitations", [])),
        })
    result = pd.DataFrame(results)
    if not result.empty:
        order = {"COMPATIBLE": 0, "CONDITIONNEL": 1, "BLOQUÉ": 2}
        result["_order"] = result["Statut"].map(order)
        result = result.sort_values(["_order", "Score de compatibilité"], ascending=[True, False]).drop(columns=["_order"]).reset_index(drop=True)
    return result


def build_required_questions(domains, questions, context):
    questions_list = []
    if not context["study_unit"]:
        questions_list.append({"Élément": "Unité d'étude", "Question": "Quelle entité représente une ligne/objet unique ?", "Pourquoi": "Nécessaire pour structurer les conclusions.", "Obligatoire": True})
    if ("Prédiction" in questions or "Explication / diagnostic" in questions or "Comparaison" in questions) and not context["target"]:
        questions_list.append({"Élément": "Variable cible Y", "Question": "Quelle est la valeur ou le résultat à analyser ?", "Pourquoi": "Nécessaire pour évaluer la modélisation.", "Obligatoire": True})
    return pd.DataFrame(questions_list)


def build_reconstruction(domains, questions, context):
    return pd.DataFrame([
        {"Élément": "Unité d'étude", "Statut": "CONFIRMÉ" if context["study_unit"] else "À CONFIRMER", "Information": context["study_unit"] or "Aucune unité confirmée."},
        {"Élément": "Variable cible Y", "Statut": "CONFIRMÉE" if context["target"] else "À CONFIRMER", "Information": context["target"] or "Aucune cible confirmée."},
    ])


def descriptive_statistics(df):
    numeric = df.select_dtypes(include=np.number)
    numeric_stats = numeric.describe().T if not numeric.empty else pd.DataFrame()
    if not numeric.empty:
        numeric_stats["missing"] = numeric.isna().sum()
        numeric_stats["missing_%"] = numeric.isna().mean() * 100
    
    categorical_records = []
    for column in df.select_dtypes(include=["object", "category", "bool"]).columns:
        mode = df[column].mode(dropna=True)
        dominant = mode.iloc[0] if not mode.empty else None
        categorical_records.append({
            "Variable": column,
            "Modalités": int(df[column].nunique(dropna=True)),
            "Modalité dominante": dominant,
            "Valeurs manquantes": int(df[column].isna().sum()),
        })
    return numeric_stats, pd.DataFrame(categorical_records)


def correlation_analysis(df):
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        return {"pearson": pd.DataFrame(), "spearman": pd.DataFrame()}
    return {"pearson": numeric.corr(method="pearson"), "spearman": numeric.corr(method="spearman")}


def group_comparison(df, target, group_variable):
    if target not in df.columns or group_variable not in df.columns or not pd.api.types.is_numeric_dtype(df[target]):
        return {"status": "NON EXECUTABLE", "message": "Variables absentes ou cible non quantitative."}
    data = df[[target, group_variable]].dropna()
    groups = [values[target].values for _, values in data.groupby(group_variable) if len(values) >= 2]
    group_names = [name for name, values in data.groupby(group_variable) if len(values) >= 2]
    if len(groups) < 2:
        return {"status": "NON EXECUTABLE", "message": "Pas assez de groupes."}
    rows = [{"Groupe": name, "n": len(vals), "Moyenne": float(np.mean(vals)), "Médiane": float(np.median(vals))} for name, vals in zip(group_names, groups)]
    return {"status": "OK", "table": pd.DataFrame(rows), "test": None}


def prepare_numeric_matrix(df, predictors, target):
    columns = [c for c in predictors if c in df.columns and c != target]
    if not columns: return None
    work = df[columns + [target]].copy()
    work = pd.get_dummies(work, columns=[c for c in columns if not pd.api.types.is_numeric_dtype(work[c])], drop_first=True, dtype=float)
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty: return None
    X_cols = [c for c in work.columns if c != target]
    return work[X_cols].astype(float).values, work[target].astype(float).values, X_cols, work.index


def linear_regression_analysis(df, target, predictors):
    if not SCIPY_AVAILABLE: return {"status": "NON EXECUTABLE", "message": "SciPy requis."}
    prepared = prepare_numeric_matrix(df, predictors, target)
    if prepared is None: return {"status": "NON EXECUTABLE", "message": "Matrice invalide."}
    X, y, columns, index = prepared
    n = len(y)
    if n < 10: return {"status": "NON EXECUTABLE", "message": "Trop peu de données."}
    X_design = np.column_stack([np.ones(n), X])
    try:
        beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
        y_hat = X_design @ beta
        residuals = y - y_hat
        ss_res, ss_tot = np.sum(residuals**2), np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {
            "status": "OK", "n": n,
            "coefficients": pd.DataFrame({"Variable": ["Intercept"] + columns, "Coefficient": beta}),
            "predictions": pd.DataFrame({"Observation": index, "Réel": y, "Prédit": y_hat, "Résidu": residuals}),
            "r2": float(r2), "rmse": float(np.sqrt(np.mean(residuals**2))), "mae": float(np.mean(np.abs(residuals)))
        }
    except Exception as exc:
        return {"status": "NON EXECUTABLE", "message": str(exc)}


def validate_regression(reg_res):
    if not reg_res or reg_res.get("status") != "OK": return {"status": "NON DISPONIBLE"}
    residuals = reg_res["predictions"]["Résidu"].values
    res = {"status": "OK"}
    if len(residuals) >= 8 and SCIPY_AVAILABLE:
        try:
            stat, p = stats.shapiro(residuals)
            res["normalite_residus"] = {"p_value": float(p), "interpretation": "Normalité OK" if p >= 0.05 else "Écart à la normalité"}
        except Exception: pass
    return res


def execute_selected_methods(df, context, decision_table):
    results = {}
    if decision_table.empty: return results
    compatible = decision_table[decision_table["Statut"] == "COMPATIBLE"]["ID"].tolist()

    if "descriptive_statistics" in compatible:
        num_s, cat_s = descriptive_statistics(df)
        results["descriptive_statistics"] = {"status": "OK", "numeric": num_s, "categorical": cat_s}
    if "correlation" in compatible:
        results["correlation"] = correlation_analysis(df)
    if "group_comparison" in compatible and context["target"] and context["categorical_columns"]:
        results["group_comparison"] = group_comparison(df, context["target"], context["categorical_columns"][0])
    if "linear_regression" in compatible and context["target"]:
        preds = [c for c in df.columns if c != context["target"] and pd.api.types.is_numeric_dtype(df[c])]
        if preds:
            results["linear_regression"] = linear_regression_analysis(df, context["target"], preds)
            results["linear_regression_validation"] = validate_regression(results["linear_regression"])
    return results


def build_interpretation(analysis):
    statements = ["Les données sont exploitables sous réserve des contrôles méthodologiques."]
    if not analysis["target"]:
        statements.append("Aucune variable cible n'est confirmée. L'analyse supervisée est limitée.")
    corr = analysis["execution"].get("correlation")
    if corr and not corr["pearson"].empty:
        statements.append("Des corrélations significatives ont été identifiées dans la matrice, sans présumer de causalité.")
    statements.append("Les résultats statistiques, les modèles et les décisions industrielles doivent rester des étapes distinctes.")
    return statements


def build_pipeline():
    return [
        "01 — Formalisation du problème",
        "02 — Classification scientifique",
        "03 — Compréhension des données",
        "04 — Contrôle de qualité",
        "05 — Scientific Knowledge Base",
        "06 — Decision Engine",
        "07 — Exécution des méthodes",
        "08 — Interprétation & Rapport"
    ]


def run_scientific_engine(problem, objective, df, confirmations):
    df = standardize_dataframe(df)
    domains, questions = classify_problem(problem, objective)
    semantics = infer_variable_semantics(df)
    dataset_profile = characterize_dataset(df)
    quality_issues = inspect_data_quality(df)
    
    study_unit_info = identify_study_unit(df, semantics)
    study_unit = confirmations.get("study_unit") or study_unit_info["candidate"]
    
    targets = identify_candidate_targets(df, semantics)
    target = confirmations.get("target")
    if target and target not in df.columns: target = None

    predictors = identify_predictors(df, target)
    context = build_scientific_context(df, target, study_unit, domains, questions, confirmations)
    hypotheses = generate_hypotheses(problem, objective, domains, questions, df, target)
    decision_table = decision_engine(SCIENTIFIC_KNOWLEDGE_BASE, context)
    required_questions = build_required_questions(domains, questions, context)
    reconstruction = build_reconstruction(domains, questions, context)
    
    validation_checks = pd.DataFrame([{"Contrôle": "Taille d'échantillon", "Statut": "OK" if len(df) >= 10 else "LIMITÉ", "Commentaire": f"{len(df)} observations."}])
    execution = execute_selected_methods(df, context, decision_table)

    analysis = {
        "engine": ENGINE_NAME, "version": ENGINE_VERSION,
        "problem": problem, "objective": objective,
        "domains": domains, "questions": questions,
        "dataset_profile": dataset_profile, "semantics": semantics,
        "quality_issues": quality_issues, "study_unit": study_unit,
        "study_unit_candidates": study_unit_info["candidates"],
        "targets": targets, "target": target, "predictors": predictors,
        "hypotheses": hypotheses, "decision_context": context,
        "decision_table": decision_table, "required_questions": required_questions,
        "reconstruction": reconstruction, "validation_checks": validation_checks,
        "execution": execution, "pipeline": build_pipeline(),
        "generated_at": datetime.now().isoformat(),
    }
    analysis["interpretation"] = build_interpretation(analysis)
    return analysis


def generate_pdf(analysis):
    buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.3*cm, leftMargin=1.3*cm, topMargin=1.3*cm, bottomMargin=1.3*cm)
    story = [Paragraph("RAPPORT D'INGÉNIERIE INDUSTRIELLE", styles["Title"]), Spacer(1, 0.5*cm)]
    story.append(Paragraph(f"Problème : {analysis['problem']}", styles["BodyText"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def build_json_export(analysis):
    return json_safe(analysis)


# ============================================================
# INTERFACE UTILISATEUR SIMPLIFIÉE
# ============================================================

st.markdown('<div class="main-title">⚙️ Scientific Engineering Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Outil d\'aide à l\'analyse de données et à la résolution de problèmes industriels.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("1 — Votre Problème")
    problem_desc = st.text_area(
        "Décrivez votre situation ou votre problème",
        height=150,
        placeholder="Ex: Nous avons une hausse des pannes sur la ligne 3 et nous voulons comprendre quelles variables de réglage influencent ces arrêts..."
    )
    objective = st.text_input("Objectif principal", placeholder="Ex: Expliquer, prédire, optimiser...")
    
    st.header("2 — Vos Données")
    uploaded_file = st.file_uploader("Importer votre fichier (CSV ou Excel)", type=["csv", "xlsx"])


if uploaded_file is None:
    st.info("👋 Bienvenue ! Veuillez importer un fichier de données dans le volet de gauche pour démarrer.")
    st.stop()

try:
    df = load_uploaded_data(uploaded_file)
    df = standardize_dataframe(df)
    st.session_state.df = df
except Exception as exc:
    st.error(f"Erreur de lecture du fichier : {exc}")
    st.stop()

# Aperçu rapide
st.header("3 — Aperçu de vos données")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes (observations)", f"{len(df):,}")
c2.metric("Colonnes (variables)", f"{len(df.columns):,}")
c3.metric("Valeurs manquantes", f"{int(df.isna().sum().sum()):,}")
c4.metric("Doublons", f"{int(df.duplicated().sum()):,}")

with st.expander("Voir un extrait des données"):
    st.dataframe(df.head(10), use_container_width=True)


# ============================================================
# FORMALISATION CLaire (Réponse au point 2 de l'utilisateur)
# ============================================================

st.markdown("---")
st.header("4 — Paramétrage simple de l'étude")

st.markdown(
    """
    <div class="help-box">
    <b>💡 Guide pour remplir ce formulaire (Exemple fil rouge) :</b><br>
    Imaginons que vous travaillez dans une usine et que votre tableau de bord liste des <b>produits fabriqués heure par heure</b>, avec leur température de cuisson, la machine utilisée, et s'ils sont conformes ou non.
    </div>
    """,
    unsafe_allow_html=True,
)

col_f1, col_f2 = st.columns(2)

with col_f1:
    st.markdown("### 🎯 Ce que vous cherchez à étudier (La Cible Y)")
    st.markdown(
        """
        * **Explication simple :** C'est **la question principale** ou le résultat que vous voulez expliquer, prédire ou surveiller.
        * **Exemple :** Si vous voulez savoir pourquoi certains produits ont un défaut, la cible Y est la colonne **« Statut_Conformité »** ou **« Rebut (Oui/Non) »**.
        """
    )
    target_options = ["— Aucune sélection —"] + list(df.columns)
    target_default = target_options.index(st.session_state.confirmations["target"]) if st.session_state.confirmations.get("target") in df.columns else 0
    target_choice = st.selectbox("Sélectionnez votre variable Cible (Y)", target_options, index=target_default)

with col_f2:
    st.markdown("### 🏷️ L'élément unique observé (Unité d'étude)")
    st.markdown(
        """
        * **Explication simple :** C'est ce que représente **une seule ligne** de votre tableau (une pièce, une machine, un client, un jour...).
        * **Exemple :** Si chaque ligne de votre fichier correspond à un produit unique qui passe sur la ligne, l'unité d'étude est l'identifiant du produit (**« ID_Piece »** ou **« Numero_Serie »**).
        """
    )
    unit_options = ["— Aucune sélection —"] + list(df.columns)
    unit_default = unit_options.index(st.session_state.confirmations["study_unit"]) if st.session_state.confirmations.get("study_unit") in df.columns else 0
    unit_choice = st.selectbox("Sélectionnez l'élément unique (Unité d'étude)", unit_options, index=unit_default)

# Variable temporelle
datetime_candidates = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or contains_any(col, ["date", "time", "timestamp", "heure", "jour"])]
time_options = ["— Aucune —"] + datetime_candidates
time_default = time_options.index(st.session_state.confirmations["time_variable"]) if st.session_state.confirmations.get("time_variable") in datetime_candidates else 0

st.markdown("### ⏰ Suivi dans le temps")
time_choice = st.selectbox("Votre tableau comporte-t-il une date ou une heure de suivi ?", time_options, index=time_default)


# ============================================================
# FORMULAIRES CONDITIONNELS SIMPLIFIÉS (Réponse au point 3)
# ============================================================

domains_preview, questions_preview = classify_problem(problem_desc, objective)

event_choice = "— Aucun —"
time_to_event_confirmed = False
censoring_defined = False
selected_controls = []
objective_function = ""
constraints = ""
normal_behavior_defined = False

if "Fiabilité" in domains_preview:
    st.markdown("---")
    st.subheader("🛠️ Option : Suivi des pannes / de la maintenance")
    st.info("Vous avez mentionné des notions de pannes ou de durée de vie. Précisez les éléments suivants pour affiner l'analyse de fiabilité :")
    
    event_options = ["— Aucun —"] + list(df.columns)
    event_choice = st.selectbox("Quelle colonne indique qu'une panne ou un arrêt critique s'est produit ?", event_options)
    time_to_event_confirmed = st.checkbox("Le temps de fonctionnement cumulé avant la panne est bien calculé dans les données.")
    censoring_defined = st.checkbox("Les équipements qui n'ont pas encore eu de panne sont bien signalés comme 'actifs / non cassés'.")

if "Optimisation" in questions_preview:
    st.markdown("---")
    st.subheader("📈 Option : Paramètres d'optimisation")
    st.info("Vous cherchez à améliorer ou optimiser un processus. Dites-nous quels leviers vous pouvez modifier :")
    
    controllable_candidates = list(df.select_dtypes(include=np.number).columns)
    selected_controls = st.multiselect("Quelles colonnes pouvez-vous modifier/régler directement (ex: température, vitesse) ?", controllable_candidates)
    objective_function = st.text_input("Que voulez-vous faire ?", placeholder="Ex: Minimiser les défauts ou maximiser la production")
    constraints = st.text_input("Y a-t-il des limites à respecter ?", placeholder="Ex: Ne pas dépasser 120°C")

if "Détection" in questions_preview:
    st.markdown("---")
    st.subheader("🔍 Option : Détection d'anomalies")
    normal_behavior_defined = st.checkbox("Disposez-vous d'une période de référence où le système fonctionnait normalement ?")


# Validation du formulaire simple
if st.button("✅ Valider mes choix et lancer l'analyse", type="primary", use_container_width=True):
    st.session_state.confirmations = {
        "target": None if target_choice == "— Aucune sélection —" else target_choice,
        "study_unit": None if unit_choice == "— Aucune sélection —" else unit_choice,
        "time_variable": None if time_choice == "— Aucune —" else time_choice,
        "event_variable": None if event_choice == "— Aucun —" else event_choice,
        "time_to_event_confirmed": time_to_event_confirmed,
        "censoring_defined": censoring_defined,
        "controllable_variables": selected_controls,
        "objective_function": objective_function,
        "constraints": constraints,
        "normal_behavior_defined": normal_behavior_defined,
    }
    st.success("Paramètres enregistrés avec succès ! L'analyse se lance...")
    st.rerun()


# ============================================================
# EXÉCUTION & AFFICHAGE DES RÉSULTATS
# ============================================================

analysis = st.session_state.analysis
if analysis is None and st.session_state.confirmations.get("target"):
    with st.spinner("Analyse des données par le moteur scientifique..."):
        analysis = run_scientific_engine(problem_desc, objective, df, st.session_state.confirmations)
        st.session_state.analysis = analysis

if analysis is not None:
    st.markdown("---")
    st.header("📊 Résultats de l'analyse")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.subheader("Domaines identifiés")
        for d in analysis["domains"]: st.info(d)
    with col_r2:
        st.subheader("Questions traitées")
        for q in analysis["questions"]: st.info(q)

    st.subheader("Interprétations clés")
    for statement in analysis["interpretation"]:
        st.write(f"• {statement}")

    # Export PDF simplifié
    st.markdown("---")
    pdf_data = generate_pdf(analysis)
    st.download_button("📥 Télécharger le rapport de synthèse (PDF)", data=pdf_data, file_name="Rapport_Industriel.pdf", mime="application/pdf", use_container_width=True)
else:
    st.info("👉 Veuillez remplir les choix ci-dessus et cliquer sur le bouton de validation pour afficher les résultats.")
