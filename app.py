# ============================================================
# SCIENTIFIC ENGINEERING ENGINE
# Meta-moteur générique d'ingénierie scientifique industrielle
#
# VERSION 3.0
#
# ARCHITECTURE
#
# PROBLEME
#    ↓
# FORMALISATION
#    ↓
# CLASSIFICATION
#    ↓
# COMPREHENSION DES DONNEES
#    ↓
# DATA QUALITY
#    ↓
# RECONSTRUCTION DU PHENOMENE
#    ↓
# UNITE D'ETUDE
#    ↓
# Y / X
#    ↓
# HYPOTHESES
#    ↓
# SCIENTIFIC KNOWLEDGE BASE
#    ↓
# DECISION ENGINE
#    ↓
# VALIDATION DES CONDITIONS
#    ↓
# ANALYSE
#    ↓
# VALIDATION DES RESULTATS
#    ↓
# INTERPRETATION
#    ↓
# DECISION
#    ↓
# RAPPORT SCIENTIFIQUE
#
# PRINCIPES
#
# - aucun domaine industriel prédéfini
# - aucun dataset de démonstration
# - aucune hypothèse métier codée en dur
# - aucune cible imposée
# - aucune causalité déduite automatiquement
# - aucune méthode avancée exécutée sans conditions
# - les informations ambiguës deviennent des demandes
#   de confirmation utilisateur
# - les résultats sont séparés des interprétations
#
# DEPENDANCES
#
# pip install streamlit pandas numpy matplotlib reportlab scipy
#
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
ENGINE_VERSION = "3.0"

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
# OUTILS GENERAUX
# ============================================================

def normalize_name(value):

    value = str(value).strip().lower()

    replacements = {
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "à": "a",
        "â": "a",
        "ä": "a",
        "î": "i",
        "ï": "i",
        "ô": "o",
        "ö": "o",
        "ù": "u",
        "û": "u",
        "ü": "u",
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


def json_safe(value):

    if isinstance(value, pd.DataFrame):
        return value.to_dict(
            orient="records"
        )

    if isinstance(value, pd.Series):
        return value.to_dict()

    if isinstance(
        value,
        (
            np.integer,
            np.int64,
            np.int32,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            np.float64,
            np.float32,
        ),
    ):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(
        value,
        (
            datetime,
            pd.Timestamp,
        ),
    ):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            json_safe(v)
            for v in value
        ]

    return value


def dataframe_to_json_records(df):

    if df is None:
        return []

    if df.empty:
        return []

    return json_safe(
        df.replace(
            {
                np.nan: None,
                np.inf: None,
                -np.inf: None,
            }
        )
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
            return pd.read_csv(
                uploaded_file
            )
        except Exception:

            uploaded_file.seek(0)

            try:
                return pd.read_csv(
                    uploaded_file,
                    sep=";",
                )
            except Exception as exc:

                raise ValueError(
                    f"Impossible de lire le CSV : {exc}"
                )

    if filename.endswith(".xlsx"):

        uploaded_file.seek(0)

        try:
            return pd.read_excel(
                uploaded_file
            )
        except Exception as exc:

            raise ValueError(
                f"Impossible de lire le fichier Excel : {exc}"
            )

    raise ValueError(
        "Format non supporté. Utilisez CSV ou XLSX."
    )


# ============================================================
# NETTOYAGE STRUCTUREL
# ============================================================

def standardize_dataframe(df):

    df = df.copy()

    # Supprime colonnes totalement vides
    empty_columns = [
        col
        for col in df.columns
        if df[col].notna().sum() == 0
    ]

    if empty_columns:
        df = df.drop(
            columns=empty_columns
        )

    # Nettoyage des noms
    new_columns = []

    for col in df.columns:

        clean = str(col).strip()

        if not clean:
            clean = "variable"

        new_columns.append(clean)

    df.columns = new_columns

    return df


# ============================================================
# INFERENCE SEMANTIQUE
# ============================================================

def infer_variable_semantics(df):

    records = []

    for column in df.columns:

        series = df[column]
        normalized = normalize_name(
            column
        )

        role = "Variable générale"
        confidence = 0.30

        # ----------------------------------------------------
        # DATETIME NATIF
        # ----------------------------------------------------

        if pd.api.types.is_datetime64_any_dtype(
            series
        ):

            role = "Variable temporelle"
            confidence = 0.99

        # ----------------------------------------------------
        # TEMPS DETECTE PAR NOM
        # ----------------------------------------------------

        elif contains_any(
            normalized,
            [
                "date",
                "datetime",
                "timestamp",
                "heure",
                "jour",
                "annee",
                "year",
                "time",
            ],
        ):

            parsed = pd.to_datetime(
                series,
                errors="coerce",
            )

            if (
                parsed.notna().mean()
                >= 0.70
            ):

                role = "Variable temporelle"
                confidence = 0.90

        # ----------------------------------------------------
        # IDENTIFIANT
        # ----------------------------------------------------

        elif contains_any(
            normalized,
            [
                "id",
                "identifiant",
                "asset_id",
                "equipment_id",
                "machine_id",
                "unit_id",
                "sample_id",
                "observation_id",
            ],
        ):

            role = "Identifiant potentiel"
            confidence = 0.85

        # ----------------------------------------------------
        # EVENEMENT / CIBLE
        # ----------------------------------------------------

        elif contains_any(
            normalized,
            [
                "event",
                "failure",
                "fault",
                "defaut",
                "defaillance",
                "incident",
                "panne",
                "target",
                "label",
                "class",
                "classe",
                "outcome",
                "result",
                "resultat",
            ],
        ):

            role = "Événement / cible potentielle"
            confidence = 0.85

        # ----------------------------------------------------
        # EXPOSITION
        # ----------------------------------------------------

        elif contains_any(
            normalized,
            [
                "exposition",
                "exposure",
                "distance",
                "kilometrage",
                "mileage",
                "hours",
                "heures",
                "cycles",
                "cycle",
                "volume",
                "quantite",
                "production",
            ],
        ):

            role = "Exposition potentielle"
            confidence = 0.80

        # ----------------------------------------------------
        # DUREE
        # ----------------------------------------------------

        elif contains_any(
            normalized,
            [
                "duration",
                "duree",
                "delay",
                "delai",
            ],
        ):

            role = "Durée potentielle"
            confidence = 0.72

        # ----------------------------------------------------
        # NUMERIQUE
        # ----------------------------------------------------

        elif pd.api.types.is_numeric_dtype(
            series
        ):

            role = "Variable quantitative"
            confidence = 0.60

        # ----------------------------------------------------
        # CATEGORIEL
        # ----------------------------------------------------

        elif (
            pd.api.types.is_object_dtype(
                series
            )
            or pd.api.types.is_categorical_dtype(
                series
            )
            or pd.api.types.is_bool_dtype(
                series
            )
        ):

            role = "Variable catégorielle"
            confidence = 0.55

        records.append(
            {
                "Variable": column,
                "Type": str(series.dtype),
                "Rôle inféré": role,
                "Confiance": round(
                    confidence,
                    2,
                ),
                "Valeurs manquantes (%)":
                    round(
                        series.isna().mean()
                        * 100,
                        2,
                    ),
                "Valeurs uniques":
                    int(
                        series.nunique(
                            dropna=True
                        )
                    ),
                "Min":
                    safe_float(
                        series.min()
                    )
                    if pd.api.types.is_numeric_dtype(
                        series
                    )
                    else None,
                "Max":
                    safe_float(
                        series.max()
                    )
                    if pd.api.types.is_numeric_dtype(
                        series
                    )
                    else None,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# DATA QUALITY ENGINE
# ============================================================

def inspect_data_quality(df):

    issues = []

    if df is None:

        return [
            (
                "CRITIQUE",
                "Aucune donnée fournie.",
            )
        ]

    if df.empty:

        return [
            (
                "CRITIQUE",
                "Le jeu de données est vide.",
            )
        ]

    # --------------------------------------------------------
    # NOMBRE D'OBSERVATIONS
    # --------------------------------------------------------

    if len(df) < 5:

        issues.append(
            (
                "CRITIQUE",
                "Le jeu de données contient moins de 5 observations.",
            )
        )

    elif len(df) < 10:

        issues.append(
            (
                "IMPORTANT",
                "Le jeu de données contient très peu d'observations.",
            )
        )

    # --------------------------------------------------------
    # DUPLICATS
    # --------------------------------------------------------

    duplicates = int(
        df.duplicated().sum()
    )

    if duplicates:

        issues.append(
            (
                "AVERTISSEMENT",
                f"{duplicates} ligne(s) dupliquée(s).",
            )
        )

    # --------------------------------------------------------
    # MANQUANTS
    # --------------------------------------------------------

    for column in df.columns:

        missing_rate = (
            df[column]
            .isna()
            .mean()
        )

        if missing_rate >= 0.80:

            issues.append(
                (
                    "IMPORTANT",
                    f"{column} contient "
                    f"{missing_rate * 100:.1f}% "
                    "de valeurs manquantes.",
                )
            )

        elif missing_rate >= 0.50:

            issues.append(
                (
                    "IMPORTANT",
                    f"{column} contient "
                    f"{missing_rate * 100:.1f}% "
                    "de valeurs manquantes.",
                )
            )

        elif missing_rate > 0:

            issues.append(
                (
                    "AVERTISSEMENT",
                    f"{column} contient "
                    f"{missing_rate * 100:.1f}% "
                    "de valeurs manquantes.",
                )
            )

    # --------------------------------------------------------
    # NUMERIQUES
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        series = df[column].dropna()

        if series.empty:
            continue

        negative_count = int(
            (series < 0).sum()
        )

        if negative_count:

            issues.append(
                (
                    "A VERIFIER",
                    f"{column} contient "
                    f"{negative_count} valeur(s) négative(s).",
                )
            )

        if series.nunique() <= 1:

            issues.append(
                (
                    "AVERTISSEMENT",
                    f"{column} présente une variance nulle.",
                )
            )

        # Valeurs extrêmes par IQR
        if len(series) >= 10:

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)

            iqr = q3 - q1

            if iqr > 0:

                lower = q1 - 3 * iqr
                upper = q3 + 3 * iqr

                extreme_count = int(
                    (
                        (series < lower)
                        | (series > upper)
                    ).sum()
                )

                if extreme_count:

                    issues.append(
                        (
                            "A VERIFIER",
                            f"{column} présente "
                            f"{extreme_count} valeur(s) extrêmement éloignée(s).",
                        )
                    )

    # --------------------------------------------------------
    # COLONNES VIDES
    # --------------------------------------------------------

    for column in df.columns:

        if df[column].notna().sum() == 0:

            issues.append(
                (
                    "IMPORTANT",
                    f"{column} est entièrement vide.",
                )
            )

    return issues


# ============================================================
# CLASSIFICATION SCIENTIFIQUE
# ============================================================

DOMAIN_KEYWORDS = {

    "Fiabilité": [
        "fiabilite",
        "panne",
        "defaillance",
        "duree de vie",
        "survie",
        "vieillissement",
        "usure",
        "failure",
        "reliability",
        "survival",
        "lifetime",
        "degradation",
    ],

    "Maintenance": [
        "maintenance",
        "intervention",
        "reparation",
        "immobilisation",
        "gmao",
        "preventive",
        "corrective",
    ],

    "Qualité": [
        "qualite",
        "defaut",
        "non conformite",
        "rebuts",
        "rebut",
        "defective",
        "quality",
    ],

    "Production": [
        "production",
        "cadence",
        "trs",
        "oee",
        "rendement",
        "temps de cycle",
        "cycle de production",
        "goulot",
        "capacite",
        "throughput",
    ],

    "Énergie": [
        "energie",
        "consommation",
        "kwh",
        "puissance",
        "electrique",
        "energy",
        "power",
    ],

    "Logistique": [
        "logistique",
        "stock",
        "inventaire",
        "flux",
        "transport",
        "approvisionnement",
        "supply chain",
    ],

    "Sécurité": [
        "securite",
        "accident",
        "incident",
        "risque",
        "danger",
        "safety",
    ],

    "Process": [
        "process",
        "procede",
        "parametre",
        "reglage",
        "processus",
    ],
}


QUESTION_KEYWORDS = {

    "Description": [
        "decrire",
        "analyser",
        "repartition",
        "comprendre",
        "etat des lieux",
        "caracteriser",
        "profil",
        "distribution",
    ],

    "Explication / diagnostic": [
        "pourquoi",
        "cause",
        "causes",
        "origine",
        "facteur",
        "expliquer",
        "diagnostic",
        "influence",
        "associe",
        "association",
    ],

    "Prédiction": [
        "predire",
        "prevoir",
        "prediction",
        "anticiper",
        "forecast",
        "predict",
    ],

    "Comparaison": [
        "comparer",
        "comparaison",
        "difference",
        "différence",
        "compare",
    ],

    "Optimisation": [
        "optimiser",
        "optimisation",
        "reduire",
        "ameliorer",
        "minimiser",
        "maximiser",
        "optimize",
    ],

    "Détection": [
        "detecter",
        "anomalie",
        "anomalies",
        "derive",
        "surveillance",
        "detection",
        "monitoring",
    ],

    "Simulation": [
        "simuler",
        "simulation",
        "scenario",
        "scenarios",
        "what if",
    ],
}


def classify_problem(
    problem,
    objective,
):

    text = (
        f"{problem} {objective}"
    )

    domains = []
    questions = []

    for domain, keywords in DOMAIN_KEYWORDS.items():

        if contains_any(
            text,
            keywords,
        ):

            domains.append(domain)

    for question, keywords in QUESTION_KEYWORDS.items():

        if contains_any(
            text,
            keywords,
        ):

            questions.append(question)

    if not domains:

        domains = [
            "Domaine à déterminer"
        ]

    if not questions:

        questions = [
            "Description"
        ]

    return domains, questions


# ============================================================
# PROFIL DATASET
# ============================================================

def characterize_dataset(df):

    numeric = list(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    categorical = list(
        df.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        ).columns
    )

    datetime_columns = list(
        df.select_dtypes(
            include=[
                "datetime",
                "datetimetz",
            ]
        ).columns
    )

    return {
        "n_observations": len(df),
        "n_variables": len(df.columns),
        "numeric_variables": numeric,
        "categorical_variables": categorical,
        "datetime_variables": datetime_columns,
        "numeric_count": len(numeric),
        "categorical_count": len(categorical),
        "has_time": bool(
            datetime_columns
        ),
    }


# ============================================================
# CANDIDATS UNITE D'ETUDE
# ============================================================

def identify_study_unit(
    df,
    semantics,
):

    candidates = []

    for _, row in semantics.iterrows():

        role = str(
            row["Rôle inféré"]
        ).lower()

        if "identifiant" in role:

            candidates.append(
                {
                    "Variable":
                        row["Variable"],
                    "Rôle":
                        "Identifiant potentiel",
                    "Confiance":
                        row["Confiance"],
                    "Nombre d'unités":
                        int(
                            df[
                                row["Variable"]
                            ]
                            .nunique(
                                dropna=True
                            )
                        ),
                }
            )

    if candidates:

        result = pd.DataFrame(
            candidates
        )

        result = result.sort_values(
            "Confiance",
            ascending=False,
        )

        return {
            "status":
                "CANDIDAT IDENTIFIE",
            "candidate":
                result.iloc[0]["Variable"],
            "candidates":
                result,
        }

    return {
        "status":
            "NON IDENTIFIE",
        "candidate":
            None,
        "candidates":
            pd.DataFrame(
                columns=[
                    "Variable",
                    "Rôle",
                    "Confiance",
                    "Nombre d'unités",
                ]
            ),
    }


# ============================================================
# CANDIDATS Y
# ============================================================

def identify_candidate_targets(
    df,
    semantics,
):

    candidates = []

    for _, row in semantics.iterrows():

        role = str(
            row["Rôle inféré"]
        )

        confidence = float(
            row["Confiance"]
        )

        role_lower = role.lower()

        if (
            "cible" in role_lower
            or "événement" in role_lower
        ):

            candidates.append(
                {
                    "Variable":
                        row["Variable"],
                    "Justification":
                        role,
                    "Confiance":
                        confidence,
                }
            )

    for column in df.columns:

        if any(
            item["Variable"] == column
            for item in candidates
        ):
            continue

        series = df[column]

        if pd.api.types.is_numeric_dtype(
            series
        ):

            candidates.append(
                {
                    "Variable":
                        column,
                    "Justification":
                        "Variable quantitative candidate.",
                    "Confiance":
                        0.45,
                }
            )

        elif (
            pd.api.types.is_object_dtype(
                series
            )
            or pd.api.types.is_categorical_dtype(
                series
            )
            or pd.api.types.is_bool_dtype(
                series
            )
        ):

            nunique = (
                series
                .dropna()
                .nunique()
            )

            if 2 <= nunique <= 10:

                candidates.append(
                    {
                        "Variable":
                            column,
                        "Justification":
                            "Variable catégorielle pouvant constituer une cible.",
                        "Confiance":
                            0.35,
                    }
                )

    return pd.DataFrame(
        candidates
    )


# ============================================================
# PREDICTEURS
# ============================================================

def identify_predictors(
    df,
    target=None,
):

    records = []

    for column in df.columns:

        if column == target:
            continue

        series = df[column]

        if pd.api.types.is_numeric_dtype(
            series
        ):

            records.append(
                {
                    "Variable":
                        column,
                    "Type":
                        "Quantitative",
                    "Utilisable":
                        True,
                }
            )

        elif (
            pd.api.types.is_object_dtype(
                series
            )
            or pd.api.types.is_categorical_dtype(
                series
            )
            or pd.api.types.is_bool_dtype(
                series
            )
        ):

            records.append(
                {
                    "Variable":
                        column,
                    "Type":
                        "Catégorielle",
                    "Utilisable":
                        True,
                }
            )

        elif pd.api.types.is_datetime64_any_dtype(
            series
        ):

            records.append(
                {
                    "Variable":
                        column,
                    "Type":
                        "Temporelle",
                    "Utilisable":
                        "À transformer",
                }
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# SCIENTIFIC KNOWLEDGE BASE
# ============================================================

SCIENTIFIC_KNOWLEDGE_BASE = {

    "descriptive_statistics": {

        "name":
            "Statistiques descriptives",

        "family":
            "Exploration",

        "objectives": [
            "Description",
            "Explication / diagnostic",
            "Comparaison",
            "Prédiction",
            "Détection",
            "Optimisation",
            "Simulation",
        ],

        "requires": [],

        "outputs": [
            "distribution",
            "central_tendency",
            "dispersion",
            "missingness",
        ],

        "limitations": [
            "Description uniquement.",
            "Ne démontre pas une causalité.",
        ],
    },

    "correlation": {

        "name":
            "Corrélation Pearson / Spearman",

        "family":
            "Association",

        "objectives": [
            "Description",
            "Explication / diagnostic",
            "Comparaison",
        ],

        "requires": [
            "at_least_two_numeric_variables",
        ],

        "outputs": [
            "correlation_matrix",
            "association_strength",
        ],

        "limitations": [
            "Une corrélation ne démontre pas une causalité.",
            "Les valeurs extrêmes peuvent influencer le résultat.",
        ],
    },

    "group_comparison": {

        "name":
            "Comparaison statistique de groupes",

        "family":
            "Comparaison",

        "objectives": [
            "Comparaison",
            "Explication / diagnostic",
        ],

        "requires": [
            "categorical_variable",
            "numeric_variable",
        ],

        "outputs": [
            "group_statistics",
            "effect_size",
            "p_value",
        ],

        "limitations": [
            "Le test exact dépend de la structure des données.",
            "La significativité statistique ne mesure pas seule l'importance pratique.",
        ],
    },

    "linear_regression": {

        "name":
            "Régression linéaire",

        "family":
            "Régression",

        "objectives": [
            "Explication / diagnostic",
            "Prédiction",
            "Optimisation",
        ],

        "requires": [
            "numeric_target",
            "predictors",
        ],

        "outputs": [
            "coefficients",
            "predictions",
            "r2",
            "rmse",
            "mae",
            "residuals",
        ],

        "limitations": [
            "La linéarité doit être examinée.",
            "Les résidus doivent être contrôlés.",
            "Les coefficients ne prouvent pas une causalité.",
        ],
    },

    "logistic_regression": {

        "name":
            "Régression logistique",

        "family":
            "Classification",

        "objectives": [
            "Prédiction",
            "Explication / diagnostic",
            "Comparaison",
        ],

        "requires": [
            "binary_target",
            "predictors",
        ],

        "outputs": [
            "probabilities",
            "classification",
            "accuracy",
            "precision",
            "recall",
            "roc_auc",
        ],

        "limitations": [
            "La cible doit être réellement binaire.",
            "La fuite d'information doit être contrôlée.",
        ],
    },

    "kaplan_meier": {

        "name":
            "Kaplan-Meier",

        "family":
            "Survie",

        "objectives": [
            "Description",
            "Explication / diagnostic",
            "Prédiction",
        ],

        "requires": [
            "time_to_event",
            "event_indicator",
        ],

        "outputs": [
            "survival_curve",
            "median_survival",
            "survival_probability",
        ],

        "limitations": [
            "Le temps jusqu'à événement doit être correctement reconstruit.",
            "La censure doit être explicitement définie.",
        ],
    },

    "weibull": {

        "name":
            "Weibull",

        "family":
            "Fiabilité",

        "objectives": [
            "Description",
            "Explication / diagnostic",
            "Prédiction",
        ],

        "requires": [
            "time_to_event",
            "event_indicator",
        ],

        "outputs": [
            "beta",
            "eta",
            "reliability",
            "hazard",
        ],

        "limitations": [
            "La loi de Weibull doit être compatible avec les données.",
            "Les cycles de vie doivent être correctement reconstruits.",
        ],
    },

    "time_series": {

        "name":
            "Analyse temporelle",

        "family":
            "Séries temporelles",

        "objectives": [
            "Description",
            "Prédiction",
            "Détection",
        ],

        "requires": [
            "time_variable",
            "numeric_target",
        ],

        "outputs": [
            "trend",
            "rolling_statistics",
            "temporal_dependence",
        ],

        "limitations": [
            "La validation doit respecter la chronologie.",
            "Les observations temporelles peuvent être dépendantes.",
        ],
    },

    "anomaly_detection": {

        "name":
            "Détection d'anomalies par score robuste",

        "family":
            "Anomalies",

        "objectives": [
            "Détection",
        ],

        "requires": [
            "numeric_variables",
        ],

        "outputs": [
            "anomaly_score",
            "anomaly_flag",
        ],

        "limitations": [
            "Une anomalie statistique n'est pas nécessairement une anomalie physique.",
            "Le comportement nominal doit être défini.",
        ],
    },

    "constrained_optimization": {

        "name":
            "Optimisation sous contraintes",

        "family":
            "Optimisation",

        "objectives": [
            "Optimisation",
        ],

        "requires": [
            "optimization_variables",
            "objective_function",
            "constraints",
        ],

        "outputs": [
            "candidate_solution",
            "objective_value",
            "constraint_status",
        ],

        "limitations": [
            "La solution mathématique doit rester physiquement réalisable.",
            "Les contraintes doivent être correctement formulées.",
        ],
    },
}


# ============================================================
# CONTEXTE SCIENTIFIQUE
# ============================================================

def build_scientific_context(
    df,
    target,
    unit,
    domains,
    questions,
    confirmations,
):

    numeric_columns = list(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    categorical_columns = list(
        df.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        ).columns
    )

    datetime_columns = list(
        df.select_dtypes(
            include=[
                "datetime",
                "datetimetz",
            ]
        ).columns
    )

    target_type = None
    target_is_binary = False

    if (
        target
        and target in df.columns
    ):

        series = df[target]

        if pd.api.types.is_numeric_dtype(
            series
        ):

            target_type = "numeric"

        elif (
            pd.api.types.is_object_dtype(
                series
            )
            or pd.api.types.is_categorical_dtype(
                series
            )
            or pd.api.types.is_bool_dtype(
                series
            )
        ):

            target_type = "categorical"

        unique_values = (
            series
            .dropna()
            .nunique()
        )

        target_is_binary = (
            unique_values == 2
        )

    predictor_count = len(
        [
            c
            for c in df.columns
            if c != target
        ]
    )

    return {

        "numeric_count":
            len(numeric_columns),

        "categorical_count":
            len(categorical_columns),

        "datetime_count":
            len(datetime_columns),

        "has_time":
            bool(datetime_columns),

        "numeric_columns":
            numeric_columns,

        "categorical_columns":
            categorical_columns,

        "datetime_columns":
            datetime_columns,

        "target":
            target,

        "target_type":
            target_type,

        "target_is_binary":
            target_is_binary,

        "predictor_count":
            predictor_count,

        "study_unit":
            unit,

        "domains":
            domains,

        "questions":
            questions,

        "event_variable":
            confirmations.get(
                "event_variable"
            ),

        "time_variable":
            confirmations.get(
                "time_variable"
            ),

        "censoring_defined":
            bool(
                confirmations.get(
                    "censoring_defined"
                )
            ),

        "controllable_variables":
            confirmations.get(
                "controllable_variables",
                [],
            ),

        "objective_function":
            confirmations.get(
                "objective_function"
            ),

        "constraints_defined":
            bool(
                confirmations.get(
                    "constraints"
                )
            ),

        "time_to_event_confirmed":
            bool(
                confirmations.get(
                    "time_to_event_confirmed"
                )
            ),

        "normal_behavior_defined":
            bool(
                confirmations.get(
                    "normal_behavior_defined"
                )
            ),
    }


# ============================================================
# GENERATION DES HYPOTHESES
# ============================================================

def generate_hypotheses(
    problem,
    objective,
    domains,
    questions,
    df,
    target,
):

    hypotheses = []

    numeric_columns = list(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    categorical_columns = list(
        df.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        ).columns
    )

    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

    hypotheses.append(
        {
            "ID":
                "H-DESC-01",

            "Hypothèse":
                "La structure des données permet de caractériser "
                "quantitativement le phénomène étudié.",

            "Type":
                "Descriptif",

            "Statut":
                "À vérifier",
        }
    )

    # --------------------------------------------------------
    # ASSOCIATION
    # --------------------------------------------------------

    if len(numeric_columns) >= 2:

        hypotheses.append(
            {
                "ID":
                    "H-ASSOC-01",

                "Hypothèse":
                    "Au moins une association statistique "
                    "mesurable existe entre des variables quantitatives.",

                "Type":
                    "Association",

                "Statut":
                    "À tester",
            }
        )

    # --------------------------------------------------------
    # COMPARAISON
    # --------------------------------------------------------

    if (
        categorical_columns
        and numeric_columns
    ):

        hypotheses.append(
            {
                "ID":
                    "H-COMP-01",

                "Hypothèse":
                    "La réponse étudiée peut présenter "
                    "des distributions différentes selon une "
                    "ou plusieurs catégories.",

                "Type":
                    "Comparaison",

                "Statut":
                    "À tester",
            }
        )

    # --------------------------------------------------------
    # TEMPS
    # --------------------------------------------------------

    has_time = any(
        pd.api.types.is_datetime64_any_dtype(
            df[col]
        )
        for col in df.columns
    )

    if has_time:

        hypotheses.append(
            {
                "ID":
                    "H-TIME-01",

                "Hypothèse":
                    "Le temps apporte une structure exploitable "
                    "dans l'évolution du phénomène.",

                "Type":
                    "Temporalité",

                "Statut":
                    "À tester",
            }
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if "Prédiction" in questions:

        hypotheses.append(
            {
                "ID":
                    "H-PRED-01",

                "Hypothèse":
                    "Les variables disponibles avant la réalisation "
                    "de la cible contiennent une information prédictive.",

                "Type":
                    "Prédiction",

                "Statut":
                    "À tester hors échantillon",
            }
        )

    # --------------------------------------------------------
    # DETECTION
    # --------------------------------------------------------

    if "Détection" in questions:

        hypotheses.append(
            {
                "ID":
                    "H-ANOM-01",

                "Hypothèse":
                    "Un comportement nominal suffisamment défini "
                    "permet de distinguer des observations atypiques.",

                "Type":
                    "Détection",

                "Statut":
                    "À vérifier",
            }
        )

    # --------------------------------------------------------
    # FIABILITE
    # --------------------------------------------------------

    if "Fiabilité" in domains:

        hypotheses.append(
            {
                "ID":
                    "H-REL-01",

                "Hypothèse":
                    "Le temps ou l'exposition jusqu'à l'événement "
                    "peut être reconstruit sans ambiguïté.",

                "Type":
                    "Fiabilité",

                "Statut":
                    "À confirmer",
            }
        )

    # --------------------------------------------------------
    # OPTIMISATION
    # --------------------------------------------------------

    if "Optimisation" in questions:

        hypotheses.append(
            {
                "ID":
                    "H-OPT-01",

                "Hypothèse":
                    "Des variables réellement contrôlables peuvent "
                    "être modifiées afin d'améliorer une fonction objectif.",

                "Type":
                    "Optimisation",

                "Statut":
                    "À formaliser",
            }
        )

    return pd.DataFrame(
        hypotheses
    )


# ============================================================
# EVALUATION DES CONDITIONS
# ============================================================

def evaluate_method(
    method_id,
    method,
    context,
):

    missing = []
    satisfied = []

    requirements = method.get(
        "requires",
        [],
    )

    # --------------------------------------------------------
    # VARIABLES NUMERIQUES
    # --------------------------------------------------------

    if (
        "at_least_two_numeric_variables"
        in requirements
    ):

        if context[
            "numeric_count"
        ] >= 2:

            satisfied.append(
                "≥ 2 variables quantitatives"
            )

        else:

            missing.append(
                "≥ 2 variables quantitatives"
            )

    if (
        "numeric_variables"
        in requirements
    ):

        if context[
            "numeric_count"
        ] >= 1:

            satisfied.append(
                "Variable(s) quantitative(s)"
            )

        else:

            missing.append(
                "Variable quantitative"
            )

    # --------------------------------------------------------
    # CATEGORIEL
    # --------------------------------------------------------

    if (
        "categorical_variable"
        in requirements
    ):

        if context[
            "categorical_count"
        ] >= 1:

            satisfied.append(
                "Variable catégorielle"
            )

        else:

            missing.append(
                "Variable catégorielle"
            )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    if (
        "numeric_target"
        in requirements
    ):

        if context[
            "target_type"
        ] == "numeric":

            satisfied.append(
                "Cible quantitative"
            )

        else:

            missing.append(
                "Cible quantitative confirmée"
            )

    if (
        "binary_target"
        in requirements
    ):

        if context[
            "target_is_binary"
        ]:

            satisfied.append(
                "Cible binaire"
            )

        else:

            missing.append(
                "Cible binaire"
            )

    # --------------------------------------------------------
    # PREDICTEURS
    # --------------------------------------------------------

    if (
        "predictors"
        in requirements
    ):

        if context[
            "predictor_count"
        ] >= 1:

            satisfied.append(
                "Variables explicatives disponibles"
            )

        else:

            missing.append(
                "Variables explicatives"
            )

    # --------------------------------------------------------
    # TEMPS
    # --------------------------------------------------------

    if (
        "time_variable"
        in requirements
    ):

        if (
            context[
                "has_time"
            ]
            or context[
                "time_variable"
            ]
        ):

            satisfied.append(
                "Variable temporelle"
            )

        else:

            missing.append(
                "Variable temporelle"
            )

    # --------------------------------------------------------
    # SURVIE
    # --------------------------------------------------------

    if (
        "time_to_event"
        in requirements
    ):

        if (
            context[
                "time_to_event_confirmed"
            ]
            and context[
                "event_variable"
            ]
        ):

            satisfied.append(
                "Temps jusqu'à événement confirmé"
            )

        else:

            missing.append(
                "Temps jusqu'à événement confirmé"
            )

    if (
        "event_indicator"
        in requirements
    ):

        if context[
            "event_variable"
        ]:

            satisfied.append(
                "Variable événement"
            )

        else:

            missing.append(
                "Variable événement"
            )

    # --------------------------------------------------------
    # OPTIMISATION
    # --------------------------------------------------------

    if (
        "optimization_variables"
        in requirements
    ):

        if context[
            "controllable_variables"
        ]:

            satisfied.append(
                "Variables contrôlables"
            )

        else:

            missing.append(
                "Variables contrôlables"
            )

    if (
        "objective_function"
        in requirements
    ):

        if context[
            "objective_function"
        ]:

            satisfied.append(
                "Fonction objectif"
            )

        else:

            missing.append(
                "Fonction objectif"
            )

    if (
        "constraints"
        in requirements
    ):

        if context[
            "constraints_defined"
        ]:

            satisfied.append(
                "Contraintes"
            )

        else:

            missing.append(
                "Contraintes"
            )

    # --------------------------------------------------------
    # STATUT
    # --------------------------------------------------------

    if not missing:

        status = "COMPATIBLE"

    elif len(missing) == 1:

        status = "CONDITIONNEL"

    else:

        status = "BLOQUÉ"

    # Score indicatif.
    # Ce score n'est PAS un score de qualité scientifique.
    total = len(requirements)

    if total == 0:

        score = 100

    else:

        score = int(
            100
            * len(satisfied)
            / total
        )

    return {
        "status":
            status,

        "score":
            score,

        "satisfied":
            satisfied,

        "missing":
            missing,
    }


# ============================================================
# DECISION ENGINE
# ============================================================

def decision_engine(
    knowledge_base,
    context,
):

    results = []

    questions = context[
        "questions"
    ]

    for method_id, method in knowledge_base.items():

        objectives = method.get(
            "objectives",
            [],
        )

        objective_match = (
            len(
                set(objectives)
                & set(questions)
            )
            > 0
        )

        # Les statistiques descriptives restent
        # une méthode générale d'exploration.
        if method_id == "descriptive_statistics":

            objective_match = True

        if not objective_match:
            continue

        evaluation = evaluate_method(
            method_id,
            method,
            context,
        )

        results.append(
            {
                "ID":
                    method_id,

                "Méthode":
                    method["name"],

                "Famille":
                    method["family"],

                "Score de compatibilité":
                    evaluation["score"],

                "Statut":
                    evaluation["status"],

                "Conditions satisfaites":
                    " | ".join(
                        evaluation[
                            "satisfied"
                        ]
                    ),

                "Informations manquantes":
                    " | ".join(
                        evaluation[
                            "missing"
                        ]
                    ),

                "Sorties":
                    " | ".join(
                        method.get(
                            "outputs",
                            [],
                        )
                    ),

                "Limites":
                    " | ".join(
                        method.get(
                            "limitations",
                            [],
                        )
                    ),
            }
        )

    result = pd.DataFrame(
        results
    )

    if not result.empty:

        order = {
            "COMPATIBLE": 0,
            "CONDITIONNEL": 1,
            "BLOQUÉ": 2,
        }

        result[
            "_order"
        ] = result[
            "Statut"
        ].map(order)

        result = result.sort_values(
            [
                "_order",
                "Score de compatibilité",
            ],
            ascending=[
                True,
                False,
            ],
        ).drop(
            columns=[
                "_order"
            ]
        )

        result = result.reset_index(
            drop=True
        )

    return result


# ============================================================
# QUESTIONS SCIENTIFIQUES NECESSAIRES
# ============================================================

def build_required_questions(
    domains,
    questions,
    context,
):

    questions_list = []

    # --------------------------------------------------------
    # UNITE
    # --------------------------------------------------------

    if not context[
        "study_unit"
    ]:

        questions_list.append(
            {
                "Élément":
                    "Unité d'étude",

                "Question":
                    "Quelle entité représente une observation scientifique ?",

                "Pourquoi":
                    "Le moteur doit connaître l'unité sur laquelle "
                    "les conclusions seront formulées.",

                "Obligatoire":
                    True,
            }
        )

    # --------------------------------------------------------
    # CIBLE
    # --------------------------------------------------------

    if (
        "Prédiction" in questions
        or "Explication / diagnostic" in questions
        or "Comparaison" in questions
    ):

        if not context[
            "target"
        ]:

            questions_list.append(
                {
                    "Élément":
                        "Variable cible Y",

                    "Question":
                        "Quelle grandeur ou quel événement constitue la réponse étudiée ?",

                    "Pourquoi":
                        "Sans Y correctement défini, une modélisation "
                        "supervisée ne peut pas être validée.",

                    "Obligatoire":
                        True,
                }
            )

    # --------------------------------------------------------
    # TEMPS
    # --------------------------------------------------------

    if (
        "Prédiction" in questions
        and context["has_time"]
    ):

        if not context[
            "time_variable"
        ]:

            questions_list.append(
                {
                    "Élément":
                        "Variable temporelle",

                    "Question":
                        "Quelle variable définit la chronologie ?",

                    "Pourquoi":
                        "La validation prédictive doit respecter "
                        "l'ordre temporel si la question porte sur le futur.",

                    "Obligatoire":
                        True,
                }
            )

    # --------------------------------------------------------
    # FIABILITE
    # --------------------------------------------------------

    if "Fiabilité" in domains:

        if not context[
            "event_variable"
        ]:

            questions_list.append(
                {
                    "Élément":
                        "Événement",

                    "Question":
                        "Quelle variable définit explicitement l'événement étudié ?",

                    "Pourquoi":
                        "Une intervention, une observation ou une opération "
                        "ne constitue pas automatiquement une défaillance.",

                    "Obligatoire":
                        True,
                }
            )

        if not context[
            "time_to_event_confirmed"
        ]:

            questions_list.append(
                {
                    "Élément":
                        "Temps jusqu'à événement",

                    "Question":
                        "Le temps/exposition entre origine du cycle "
                        "et événement est-il correctement reconstruit ?",

                    "Pourquoi":
                        "Kaplan-Meier et Weibull nécessitent une durée "
                        "d'observation scientifiquement définie.",

                    "Obligatoire":
                        True,
                }
            )

        if not context[
            "censoring_defined"
        ]:

            questions_list.append(
                {
                    "Élément":
                        "Censure",

                    "Question":
                        "Les unités n'ayant pas connu l'événement "
                        "sont-elles identifiées comme censurées ?",

                    "Pourquoi":
                        "Ignorer la censure peut biaiser l'analyse de survie.",

                    "Obligatoire":
                        True,
                }
            )

    # --------------------------------------------------------
    # OPTIMISATION
    # --------------------------------------------------------

    if "Optimisation" in questions:

        if not context[
            "controllable_variables"
        ]:

            questions_list.append(
                {
                    "Élément":
                        "Variables contrôlables",

                    "Question":
                        "Quelles variables peuvent réellement être modifiées ?",

                    "Pourquoi":
                        "Une variable observée n'est pas nécessairement "
                        "une variable de décision.",

                    "Obligatoire":
                        True,
                }
            )

        if not context[
            "objective_function"
        ]:

            questions_list.append(
                {
                    "Élément":
                        "Fonction objectif",

                    "Question":
                        "Quelle grandeur doit être minimisée, maximisée "
                        "ou rapprochée d'une cible ?",

                    "Pourquoi":
                        "Une optimisation nécessite une fonction objectif.",

                    "Obligatoire":
                        True,
                }
            )

        if not context[
            "constraints_defined"
        ]:

            questions_list.append(
                {
                    "Élément":
                        "Contraintes",

                    "Question":
                        "Quelles limites physiques, économiques, "
                        "réglementaires ou opérationnelles doivent être respectées ?",

                    "Pourquoi":
                        "Une solution sans contraintes peut être mathématiquement "
                        "optimale mais industriellement irréalisable.",

                    "Obligatoire":
                        True,
                }
            )

    return pd.DataFrame(
        questions_list
    )


# ============================================================
# RECONSTRUCTION DU PHENOMENE
# ============================================================

def build_reconstruction(
    domains,
    questions,
    context,
):

    records = []

    records.append(
        {
            "Élément":
                "Unité d'étude",

            "Statut":
                "CONFIRMÉ"
                if context[
                    "study_unit"
                ]
                else
                "À CONFIRMER",

            "Information":
                context[
                    "study_unit"
                ]
                or
                "Aucune unité confirmée.",
        }
    )

    records.append(
        {
            "Élément":
                "Variable cible Y",

            "Statut":
                "CONFIRMÉE"
                if context[
                    "target"
                ]
                else
                "À CONFIRMER",

            "Information":
                context[
                    "target"
                ]
                or
                "Aucune cible confirmée.",
        }
    )

    if context[
        "has_time"
    ]:

        records.append(
            {
                "Élément":
                    "Temporalité",

                "Statut":
                    "PRÉSENTE",

                "Information":
                    "Une ou plusieurs variables temporelles ont été détectées.",
            }
        )

    if "Fiabilité" in domains:

        records.extend(
            [
                {
                    "Élément":
                        "Définition événement",

                    "Statut":
                        "CONFIRMÉE"
                        if context[
                            "event_variable"
                        ]
                        else
                        "À CONFIRMER",

                    "Information":
                        context[
                            "event_variable"
                        ]
                        or
                        "Événement non confirmé.",
                },

                {
                    "Élément":
                        "Temps jusqu'à événement",

                    "Statut":
                        "CONFIRMÉ"
                        if context[
                            "time_to_event_confirmed"
                        ]
                        else
                        "À CONFIRMER",

                    "Information":
                        "Structure nécessaire à l'analyse de survie.",
                },

                {
                    "Élément":
                        "Censure",

                    "Statut":
                        "CONFIRMÉE"
                        if context[
                            "censoring_defined"
                        ]
                        else
                        "À CONFIRMER",

                    "Information":
                        "Les observations sans événement doivent être distinguées.",
                },
            ]
        )

    if "Prédiction" in questions:

        records.append(
            {
                "Élément":
                    "Fuite d'information",

                "Statut":
                    "À CONTRÔLER",

                "Information":
                    "Les variables disponibles après la date de prédiction "
                    "ne doivent pas entrer dans les prédicteurs.",
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# STATISTIQUES DESCRIPTIVES
# ============================================================

def descriptive_statistics(df):

    numeric = df.select_dtypes(
        include=np.number
    )

    if numeric.empty:

        numeric_stats = pd.DataFrame()

    else:

        numeric_stats = numeric.describe().T

        numeric_stats[
            "missing"
        ] = numeric.isna().sum()

        numeric_stats[
            "missing_%"
        ] = (
            numeric.isna().mean()
            * 100
        )

    categorical_records = []

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool",
        ]
    ).columns

    for column in categorical_columns:

        mode = (
            df[column]
            .mode(
                dropna=True
            )
        )

        dominant = (
            mode.iloc[0]
            if not mode.empty
            else None
        )

        categorical_records.append(
            {
                "Variable":
                    column,

                "Modalités":
                    int(
                        df[column]
                        .nunique(
                            dropna=True
                        )
                    ),

                "Modalité dominante":
                    dominant,

                "Valeurs manquantes":
                    int(
                        df[column]
                        .isna()
                        .sum()
                    ),
            }
        )

    return (
        numeric_stats,
        pd.DataFrame(
            categorical_records
        ),
    )


# ============================================================
# CORRELATIONS
# ============================================================

def correlation_analysis(df):

    numeric = df.select_dtypes(
        include=np.number
    )

    if numeric.shape[1] < 2:

        return {
            "pearson":
                pd.DataFrame(),

            "spearman":
                pd.DataFrame(),
        }

    return {
        "pearson":
            numeric.corr(
                method="pearson"
            ),

        "spearman":
            numeric.corr(
                method="spearman"
            ),
    }


# ============================================================
# COMPARAISON DE GROUPES
# ============================================================

def group_comparison(
    df,
    target,
    group_variable,
):

    if (
        target not in df.columns
        or group_variable not in df.columns
    ):

        return {
            "status":
                "NON EXECUTABLE",
            "message":
                "Variable absente.",
        }

    if not pd.api.types.is_numeric_dtype(
        df[target]
    ):

        return {
            "status":
                "NON EXECUTABLE",
            "message":
                "La cible doit être quantitative.",
        }

    data = df[
        [
            target,
            group_variable,
        ]
    ].dropna()

    groups = [
        values[target].values
        for _, values
        in data.groupby(
            group_variable
        )
        if len(values) >= 2
    ]

    group_names = [
        name
        for name, values
        in data.groupby(
            group_variable
        )
        if len(values) >= 2
    ]

    if len(groups) < 2:

        return {
            "status":
                "NON EXECUTABLE",
            "message":
                "Au moins deux groupes avec suffisamment d'observations sont nécessaires.",
        }

    rows = []

    for name, values in zip(
        group_names,
        groups,
    ):

        rows.append(
            {
                "Groupe":
                    name,

                "n":
                    len(values),

                "Moyenne":
                    float(
                        np.mean(values)
                    ),

                "Médiane":
                    float(
                        np.median(values)
                    ),

                "Écart-type":
                    float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    )
                    if len(values) > 1
                    else np.nan,
            }
        )

    result = {
        "status":
            "OK",

        "table":
            pd.DataFrame(rows),

        "test":
            None,
    }

    if not SCIPY_AVAILABLE:

        result["test"] = {
            "message":
                "SciPy indisponible : test statistique non calculé."
        }

        return result

    if len(groups) == 2:

        statistic, p_value = (
            stats.mannwhitneyu(
                groups[0],
                groups[1],
                alternative="two-sided",
            )
        )

        result["test"] = {
            "méthode":
                "Mann-Whitney U",

            "statistique":
                float(statistic),

            "p_value":
                float(p_value),
        }

    else:

        statistic, p_value = (
            stats.kruskal(
                *groups
            )
        )

        result["test"] = {
            "méthode":
                "Kruskal-Wallis",

            "statistique":
                float(statistic),

            "p_value":
                float(p_value),
        }

    return result


# ============================================================
# REGRESSION LINEAIRE
# ============================================================

def prepare_numeric_matrix(
    df,
    predictors,
    target,
):

    columns = [
        c
        for c in predictors
        if c in df.columns
        and c != target
    ]

    if not columns:
        return None

    work = df[
        columns + [target]
    ].copy()

    # Encodage simple des catégorielles
    work = pd.get_dummies(
        work,
        columns=[
            c
            for c in columns
            if (
                pd.api.types.is_object_dtype(
                    work[c]
                )
                or pd.api.types.is_categorical_dtype(
                    work[c]
                )
                or pd.api.types.is_bool_dtype(
                    work[c]
                )
            )
        ],
        drop_first=True,
        dtype=float,
    )

    work = work.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).dropna()

    if work.empty:
        return None

    X_columns = [
        c
        for c in work.columns
        if c != target
    ]

    X = work[
        X_columns
    ].astype(float).values

    y = work[
        target
    ].astype(float).values

    return X, y, X_columns, work.index


def linear_regression_analysis(
    df,
    target,
    predictors,
):

    if not SCIPY_AVAILABLE:

        return {
            "status":
                "NON EXECUTABLE",

            "message":
                "SciPy est nécessaire pour l'analyse statistique complète.",
        }

    prepared = prepare_numeric_matrix(
        df,
        predictors,
        target,
    )

    if prepared is None:

        return {
            "status":
                "NON EXECUTABLE",

            "message":
                "Impossible de construire X et Y.",
        }

    X, y, columns, index = prepared

    n = len(y)

    if n < 10:

        return {
            "status":
                "NON EXECUTABLE",

            "message":
                "Au moins 10 observations complètes sont recommandées.",
        }

    # Intercept
    X_design = np.column_stack(
        [
            np.ones(n),
            X,
        ]
    )

    try:

        beta = np.linalg.lstsq(
            X_design,
            y,
            rcond=None,
        )[0]

        y_hat = (
            X_design @ beta
        )

        residuals = (
            y - y_hat
        )

        ss_res = np.sum(
            residuals ** 2
        )

        ss_tot = np.sum(
            (
                y
                - np.mean(y)
            ) ** 2
        )

        r2 = (
            1
            - ss_res / ss_tot
            if ss_tot > 0
            else np.nan
        )

        rmse = np.sqrt(
            np.mean(
                residuals ** 2
            )
        )

        mae = np.mean(
            np.abs(residuals)
        )

        coefficients = pd.DataFrame(
            {
                "Variable":
                    [
                        "Intercept"
                    ]
                    + columns,

                "Coefficient":
                    beta,
            }
        )

        return {
            "status":
                "OK",

            "n":
                n,

            "coefficients":
                coefficients,

            "predictions":
                pd.DataFrame(
                    {
                        "Observation":
                            index,

                        "Réel":
                            y,

                        "Prédit":
                            y_hat,

                        "Résidu":
                            residuals,
                    }
                ),

            "r2":
                float(r2),

            "rmse":
                float(rmse),

            "mae":
                float(mae),
        }

    except Exception as exc:

        return {
            "status":
                "NON EXECUTABLE",

            "message":
                str(exc),
        }


# ============================================================
# VALIDATION REGRESSION
# ============================================================

def validate_regression(
    regression_result,
):

    if (
        not regression_result
        or regression_result.get(
            "status"
        ) != "OK"
    ):

        return {
            "status":
                "NON DISPONIBLE"
        }

    residuals = (
        regression_result[
            "predictions"
        ]["Résidu"]
        .values
    )

    result = {
        "status":
            "OK",
    }

    if len(residuals) >= 8:

        if SCIPY_AVAILABLE:

            try:

                shapiro_stat, shapiro_p = (
                    stats.shapiro(
                        residuals
                    )
                )

                result[
                    "normalite_residus"
                ] = {
                    "statistique":
                        float(
                            shapiro_stat
                        ),
                    "p_value":
                        float(
                            shapiro_p
                        ),
                    "interpretation":
                        (
                            "Compatible avec la normalité "
                            "au seuil usuel."
                            if shapiro_p >= 0.05
                            else
                            "Écart à la normalité détecté."
                        ),
                }

            except Exception:
                pass

    return result


# ============================================================
# REGRESSION LOGISTIQUE
# ============================================================

def logistic_regression_analysis(
    df,
    target,
    predictors,
):

    if not SCIPY_AVAILABLE:

        return {
            "status":
                "NON EXECUTABLE",
            "message":
                "SciPy est nécessaire.",
        }

    prepared = prepare_numeric_matrix(
        df,
        predictors,
        target,
    )

    if prepared is None:

        return {
            "status":
                "NON EXECUTABLE",
            "message":
                "Impossible de construire X et Y.",
        }

    X, y, columns, index = prepared

    unique = np.unique(y)

    if len(unique) != 2:

        return {
            "status":
                "NON EXECUTABLE",
            "message":
                "La cible n'est pas binaire.",
        }

    # Encodage 0/1
    y = (
        y == unique[1]
    ).astype(float)

    n = len(y)

    if n < 20:

        return {
            "status":
                "NON EXECUTABLE",
            "message":
                "Échantillon trop limité pour une estimation robuste.",
        }

    # Ajout intercept
    Xd = np.column_stack(
        [
            np.ones(n),
            X,
        ]
    )

    def sigmoid(z):

        z = np.clip(
            z,
            -30,
            30,
        )

        return 1 / (
            1 + np.exp(-z)
        )

    def objective(beta):

        p = sigmoid(
            Xd @ beta
        )

        eps = 1e-12

        return -np.sum(
            y * np.log(
                p + eps
            )
            + (
                1 - y
            )
            * np.log(
                1 - p + eps
            )
        )

    try:

        initial = np.zeros(
            Xd.shape[1]
        )

        result = minimize(
            objective,
            initial,
            method="BFGS",
        )

        beta = result.x

        probabilities = sigmoid(
            Xd @ beta
        )

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        accuracy = np.mean(
            predictions == y
        )

        tp = np.sum(
            (predictions == 1)
            & (y == 1)
        )

        fp = np.sum(
            (predictions == 1)
            & (y == 0)
        )

        fn = np.sum(
            (predictions == 0)
            & (y == 1)
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0
        )

        coefficients = pd.DataFrame(
            {
                "Variable":
                    [
                        "Intercept"
                    ]
                    + columns,

                "Coefficient":
                    beta,
            }
        )

        return {
            "status":
                "OK",

            "coefficients":
                coefficients,

            "predictions":
                pd.DataFrame(
                    {
                        "Observation":
                            index,

                        "Réel":
                            y,

                        "Probabilité":
                            probabilities,

                        "Classe prédite":
                            predictions,
                    }
                ),

            "accuracy":
                float(accuracy),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "optimization_success":
                bool(
                    result.success
                ),
        }

    except Exception as exc:

        return {
            "status":
                "NON EXECUTABLE",
            "message":
                str(exc),
        }


# ============================================================
# KAPLAN-MEIER
# ============================================================

def kaplan_meier(
    duration,
    event,
):

    data = pd.DataFrame(
        {
            "duration":
                pd.to_numeric(
                    duration,
                    errors="coerce",
                ),

            "event":
                pd.to_numeric(
                    event,
                    errors="coerce",
                ),
        }
    ).dropna()

    data = data[
        data["duration"] >= 0
    ]

    data["event"] = (
        data["event"] > 0
    ).astype(int)

    if len(data) == 0:

        return {
            "status":
                "NON EXECUTABLE"
        }

    times = np.sort(
        data["duration"]
        .unique()
    )

    survival = 1.0

    rows = []

    for t in times:

        at_risk = int(
            (
                data["duration"]
                >= t
            ).sum()
        )

        events = int(
            (
                (
                    data["duration"]
                    == t
                )
                & (
                    data["event"]
                    == 1
                )
            ).sum()
        )

        if at_risk > 0:

            survival *= (
                1
                - events / at_risk
            )

        rows.append(
            {
                "Temps":
                    float(t),

                "À risque":
                    at_risk,

                "Événements":
                    events,

                "Survie":
                    float(survival),
            }
        )

    curve = pd.DataFrame(
        rows
    )

    median_survival = None

    reached = curve[
        curve["Survie"] <= 0.5
    ]

    if not reached.empty:

        median_survival = float(
            reached.iloc[0]["Temps"]
        )

    return {
        "status":
            "OK",

        "curve":
            curve,

        "median_survival":
            median_survival,

        "n":
            len(data),

        "events":
            int(
                data["event"].sum()
            ),

        "censored":
            int(
                (
                    data["event"]
                    == 0
                ).sum()
            ),
    }


# ============================================================
# WEIBULL
# ============================================================

def weibull_fit(
    duration,
    event,
):

    if not SCIPY_AVAILABLE:

        return {
            "status":
                "NON EXECUTABLE",
            "message":
                "SciPy est nécessaire.",
        }

    data = pd.DataFrame(
        {
            "duration":
                pd.to_numeric(
                    duration,
                    errors="coerce",
                ),

            "event":
                pd.to_numeric(
                    event,
                    errors="coerce",
                ),
        }
    ).dropna()

    data = data[
        data["duration"] > 0
    ]

    data["event"] = (
        data["event"] > 0
    ).astype(int)

    if len(data) < 5:

        return {
            "status":
                "NON EXECUTABLE",
            "message":
                "Trop peu d'observations.",
        }

    t = data[
        "duration"
    ].values

    d = data[
        "event"
    ].values

    failures = d.sum()

    if failures < 2:

        return {
            "status":
                "NON EXECUTABLE",
            "message":
                "Au moins deux événements sont nécessaires.",
        }

    # --------------------------------------------------------
    # Log-vraisemblance Weibull avec censure à droite
    #
    # f(t) = beta/eta * (t/eta)^(beta-1)
    #        * exp(-(t/eta)^beta)
    #
    # contribution censurée :
    # S(t) = exp(-(t/eta)^beta)
    # --------------------------------------------------------

    def neg_log_likelihood(params):

        log_beta, log_eta = params

        beta = np.exp(log_beta)
        eta = np.exp(log_eta)

        z = (
            t / eta
        ) ** beta

        log_hazard = (
            np.log(beta)
            - np.log(eta)
            + (
                beta - 1
            )
            * (
                np.log(t)
                - np.log(eta)
            )
        )

        ll = (
            d * log_hazard
            - z
        )

        return -np.sum(ll)

    initial = np.log(
        [
            1.0,
            np.median(t),
        ]
    )

    result = minimize(
        neg_log_likelihood,
        initial,
        method="Nelder-Mead",
    )

    if not result.success:

        return {
            "status":
                "NON EXECUTABLE",
            "message":
                "Échec de l'ajustement Weibull.",
        }

    beta = np.exp(
        result.x[0]
    )

    eta = np.exp(
        result.x[1]
    )

    # --------------------------------------------------------
    # AIC
    # --------------------------------------------------------

    log_likelihood = (
        -result.fun
    )

    aic = (
        2 * 2
        - 2 * log_likelihood
    )

    # --------------------------------------------------------
    # Courbe
    # --------------------------------------------------------

    grid = np.linspace(
        max(
            np.min(t),
            1e-12,
        ),
        np.max(t),
        200,
    )

    reliability = np.exp(
        -(
            grid / eta
        ) ** beta
    )

    hazard = (
        beta / eta
        * (
            grid / eta
        ) ** (
            beta - 1
        )
    )

    curve = pd.DataFrame(
        {
            "Temps":
                grid,

            "Fiabilité":
                reliability,

            "Taux de défaillance":
                hazard,
        }
    )

    if beta < 1:

        interpretation = (
            "β < 1 : tendance à un taux de défaillance décroissant."
        )

    elif np.isclose(
        beta,
        1,
        atol=0.05,
    ):

        interpretation = (
            "β ≈ 1 : taux de défaillance approximativement constant."
        )

    else:

        interpretation = (
            "β > 1 : tendance à un taux de défaillance croissant."
        )

    return {
        "status":
            "OK",

        "beta":
            float(beta),

        "eta":
            float(eta),

        "aic":
            float(aic),

        "log_likelihood":
            float(log_likelihood),

        "events":
            int(failures),

        "censored":
            int(
                len(d)
                - failures
            ),

        "curve":
            curve,

        "interpretation":
            interpretation,
    }


# ============================================================
# ANALYSE TEMPORELLE
# ============================================================

def time_series_analysis(
    df,
    time_variable,
    target,
):

    if (
        time_variable not in df.columns
        or target not in df.columns
    ):

        return {
            "status":
                "NON EXECUTABLE",
            "message":
                "Variables absentes.",
        }

    data = df[
        [
            time_variable,
            target,
        ]
    ].copy()

    data[
        time_variable
    ] = pd.to_datetime(
        data[
            time_variable
        ],
        errors="coerce",
    )

    data[
        target
    ] = pd.to_numeric(
        data[
            target
        ],
        errors="coerce",
    )

    data = data.dropna()

    if len(data) < 5:

        return {
            "status":
                "NON EXECUTABLE",
            "message":
                "Trop peu d'observations.",
        }

    data = data.sort_values(
        time_variable
    )

    data[
        "Moyenne_mobile"
    ] = (
        data[target]
        .rolling(
            window=min(
                5,
                len(data),
            ),
            min_periods=1,
        )
        .mean()
    )

    data[
        "Ecart_mobile"
    ] = (
        data[target]
        .rolling(
            window=min(
                5,
                len(data),
            ),
            min_periods=1,
        )
        .std()
    )

    first_mean = (
        data[target]
        .head(
            max(
                2,
                len(data) // 5,
            )
        )
        .mean()
    )

    last_mean = (
        data[target]
        .tail(
            max(
                2,
                len(data) // 5,
            )
        )
        .mean()
    )

    return {
        "status":
            "OK",

        "data":
            data,

        "first_mean":
            float(first_mean),

        "last_mean":
            float(last_mean),

        "variation":
            float(
                last_mean
                - first_mean
            ),
    }


# ============================================================
# DETECTION D'ANOMALIES
# ============================================================

def anomaly_detection(
    df,
    variables,
):

    variables = [
        col
        for col in variables
        if col in df.columns
        and pd.api.types.is_numeric_dtype(
            df[col]
        )
    ]

    if not variables:

        return {
            "status":
                "NON EXECUTABLE",
            "message":
                "Aucune variable quantitative disponible.",
        }

    data = df[
        variables
    ].copy()

    medians = data.median()

    mad = (
        data
        - medians
    ).abs().median()

    mad = mad.replace(
        0,
        np.nan,
    )

    robust_z = (
        0.6745
        * (
            data
            - medians
        )
        / mad
    )

    robust_z = robust_z.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    max_score = (
        robust_z.abs()
        .max(
            axis=1,
            skipna=True,
        )
        .fillna(0)
    )

    result = df.copy()

    result[
        "Anomaly_Score"
    ] = max_score

    result[
        "Anomaly_Flag"
    ] = (
        max_score >= 3.5
    )

    return {
        "status":
            "OK",

        "data":
            result,

        "n_anomalies":
            int(
                result[
                    "Anomaly_Flag"
                ].sum()
            ),
    }


# ============================================================
# VALIDATION PREDICTIVE
# ============================================================

def train_test_regression(
    df,
    target,
    predictors,
    test_fraction=0.2,
):

    prepared = prepare_numeric_matrix(
        df,
        predictors,
        target,
    )

    if prepared is None:

        return {
            "status":
                "NON EXECUTABLE"
        }

    X, y, columns, index = prepared

    n = len(y)

    if n < 20:

        return {
            "status":
                "NON EXECUTABLE",
            "message":
                "Au moins 20 observations complètes sont recommandées.",
        }

    split = int(
        n
        * (
            1
            - test_fraction
        )
    )

    if split < 5 or n - split < 5:

        return {
            "status":
                "NON EXECUTABLE",
        }

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    X_train_d = np.column_stack(
        [
            np.ones(
                len(X_train)
            ),
            X_train,
        ]
    )

    beta = np.linalg.lstsq(
        X_train_d,
        y_train,
        rcond=None,
    )[0]

    X_test_d = np.column_stack(
        [
            np.ones(
                len(X_test)
            ),
            X_test,
        ]
    )

    predictions = (
        X_test_d @ beta
    )

    mae = np.mean(
        np.abs(
            y_test
            - predictions
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                y_test
                - predictions
            ) ** 2
        )
    )

    ss_res = np.sum(
        (
            y_test
            - predictions
        ) ** 2
    )

    ss_tot = np.sum(
        (
            y_test
            - np.mean(y_test)
        ) ** 2
    )

    r2 = (
        1
        - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    return {
        "status":
            "OK",

        "train_n":
            len(y_train),

        "test_n":
            len(y_test),

        "mae":
            float(mae),

        "rmse":
            float(rmse),

        "r2":
            float(r2),

        "predictions":
            pd.DataFrame(
                {
                    "Observation":
                        index[split:],

                    "Réel":
                        y_test,

                    "Prédit":
                        predictions,
                }
            ),
    }


# ============================================================
# VALIDATION GLOBALE
# ============================================================

def validate_conditions(
    df,
    context,
    decision_table,
):

    checks = []

    n = len(df)

    # --------------------------------------------------------
    # TAILLE
    # --------------------------------------------------------

    if n < 10:

        checks.append(
            {
                "Contrôle":
                    "Taille d'échantillon",

                "Statut":
                    "CRITIQUE",

                "Commentaire":
                    f"{n} observation(s).",
            }
        )

    elif n < 30:

        checks.append(
            {
                "Contrôle":
                    "Taille d'échantillon",

                "Statut":
                    "LIMITÉ",

                "Commentaire":
                    f"{n} observation(s). Prudence pour les modèles complexes.",
            }
        )

    else:

        checks.append(
            {
                "Contrôle":
                    "Taille d'échantillon",

                "Statut":
                    "OK",

                "Commentaire":
                    f"{n} observation(s).",
            }
        )

    # --------------------------------------------------------
    # CIBLE
    # --------------------------------------------------------

    if context[
        "target"
    ]:

        checks.append(
            {
                "Contrôle":
                    "Définition de Y",

                "Statut":
                    "À CONFIRMER",

                "Commentaire":
                    f"Cible candidate : {context['target']}.",
            }
        )

    else:

        checks.append(
            {
                "Contrôle":
                    "Définition de Y",

                "Statut":
                    "CRITIQUE",

                "Commentaire":
                    "Aucune cible confirmée.",
            }
        )

    # --------------------------------------------------------
    # TEMPS
    # --------------------------------------------------------

    if context[
        "has_time"
    ]:

        checks.append(
            {
                "Contrôle":
                    "Chronologie",

                "Statut":
                    "À PRENDRE EN COMPTE",

                "Commentaire":
                    "Les dépendances temporelles doivent être conservées.",
            }
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if "Prédiction" in context[
        "questions"
    ]:

        checks.extend(
            [
                {
                    "Contrôle":
                        "Fuite d'information",

                    "Statut":
                        "OBLIGATOIRE",

                    "Commentaire":
                        "Les informations futures ne doivent pas servir à prédire le passé.",
                },

                {
                    "Contrôle":
                        "Validation hors échantillon",

                    "Statut":
                        "OBLIGATOIRE",

                    "Commentaire":
                        "La performance doit être mesurée sur des observations non utilisées pour l'ajustement.",
                },
            ]
        )

    # --------------------------------------------------------
    # FIABILITE
    # --------------------------------------------------------

    if "Fiabilité" in context[
        "domains"
    ]:

        if context[
            "time_to_event_confirmed"
        ]:

            status = "OK"
        else:

            status = "CRITIQUE"

        checks.append(
            {
                "Contrôle":
                    "Reconstruction temps-événement",

                "Statut":
                    status,

                "Commentaire":
                    "Une intervention ne constitue pas automatiquement un événement de défaillance.",
            }
        )

        checks.append(
            {
                "Contrôle":
                    "Censure",

                "Statut":
                    "OK"
                    if context[
                        "censoring_defined"
                    ]
                    else
                    "CRITIQUE",

                "Commentaire":
                    "Les observations sans événement doivent être distinguées.",
            }
        )

    # --------------------------------------------------------
    # DECISION METHODS
    # --------------------------------------------------------

    if not decision_table.empty:

        compatible = int(
            (
                decision_table[
                    "Statut"
                ]
                == "COMPATIBLE"
            ).sum()
        )

        conditional = int(
            (
                decision_table[
                    "Statut"
                ]
                == "CONDITIONNEL"
            ).sum()
        )

        blocked = int(
            (
                decision_table[
                    "Statut"
                ]
                == "BLOQUÉ"
            ).sum()
        )

        checks.append(
            {
                "Contrôle":
                    "Méthodes scientifiques",

                "Statut":
                    "INFO",

                "Commentaire":
                    f"{compatible} compatible(s), "
                    f"{conditional} conditionnelle(s), "
                    f"{blocked} bloquée(s).",
            }
        )

    return pd.DataFrame(
        checks
    )


# ============================================================
# EXECUTION DU PIPELINE
# ============================================================

def execute_selected_methods(
    df,
    context,
    decision_table,
):

    results = {}

    if decision_table.empty:
        return results

    compatible_methods = (
        decision_table[
            decision_table[
                "Statut"
            ]
            == "COMPATIBLE"
        ]["ID"]
        .tolist()
    )

    # ========================================================
    # DESCRIPTIF
    # ========================================================

    if (
        "descriptive_statistics"
        in compatible_methods
    ):

        numeric_stats, categorical_stats = (
            descriptive_statistics(
                df
            )
        )

        results[
            "descriptive_statistics"
        ] = {
            "status":
                "OK",

            "numeric":
                numeric_stats,

            "categorical":
                categorical_stats,
        }

    # ========================================================
    # CORRELATION
    # ========================================================

    if (
        "correlation"
        in compatible_methods
    ):

        results[
            "correlation"
        ] = correlation_analysis(
            df
        )

    # ========================================================
    # COMPARAISON
    # ========================================================

    if (
        "group_comparison"
        in compatible_methods
    ):

        target = context[
            "target"
        ]

        categorical = context[
            "categorical_columns"
        ]

        if target and categorical:

            group_var = categorical[0]

            results[
                "group_comparison"
            ] = group_comparison(
                df,
                target,
                group_var,
            )

    # ========================================================
    # REGRESSION LINEAIRE
    # ========================================================

    if (
        "linear_regression"
        in compatible_methods
    ):

        target = context[
            "target"
        ]

        predictors = [
            col
            for col in df.columns
            if col != target
            and pd.api.types.is_numeric_dtype(
                df[col]
            )
        ]

        if target and predictors:

            results[
                "linear_regression"
            ] = linear_regression_analysis(
                df,
                target,
                predictors,
            )

            results[
                "linear_regression_validation"
            ] = validate_regression(
                results[
                    "linear_regression"
                ]
            )

    # ========================================================
    # LOGISTIQUE
    # ========================================================

    if (
        "logistic_regression"
        in compatible_methods
    ):

        target = context[
            "target"
        ]

        predictors = [
            col
            for col in df.columns
            if col != target
        ]

        if target:

            results[
                "logistic_regression"
            ] = logistic_regression_analysis(
                df,
                target,
                predictors,
            )

    # ========================================================
    # KAPLAN-MEIER
    # ========================================================

    if (
        "kaplan_meier"
        in compatible_methods
    ):

        event_variable = context[
            "event_variable"
        ]

        time_variable = context[
            "time_variable"
        ]

        if (
            event_variable
            and time_variable
            and time_variable in df.columns
            and event_variable in df.columns
        ):

            results[
                "kaplan_meier"
            ] = kaplan_meier(
                df[
                    time_variable
                ],
                df[
                    event_variable
                ],
            )

    # ========================================================
    # WEIBULL
    # ========================================================

    if (
        "weibull"
        in compatible_methods
    ):

        event_variable = context[
            "event_variable"
        ]

        time_variable = context[
            "time_variable"
        ]

        if (
            event_variable
            and time_variable
            and time_variable in df.columns
            and event_variable in df.columns
        ):

            results[
                "weibull"
            ] = weibull_fit(
                df[
                    time_variable
                ],
                df[
                    event_variable
                ],
            )

    # ========================================================
    # TEMPS
    # ========================================================

    if (
        "time_series"
        in compatible_methods
    ):

        target = context[
            "target"
        ]

        time_variable = context[
            "time_variable"
        ]

        if (
            target
            and time_variable
        ):

            results[
                "time_series"
            ] = time_series_analysis(
                df,
                time_variable,
                target,
            )

    # ========================================================
    # ANOMALIES
    # ========================================================

    if (
        "anomaly_detection"
        in compatible_methods
    ):

        variables = context[
            "numeric_columns"
        ]

        results[
            "anomaly_detection"
        ] = anomaly_detection(
            df,
            variables,
        )

    # ========================================================
    # VALIDATION PREDICTIVE
    # ========================================================

    if (
        "linear_regression"
        in compatible_methods
        and "Prédiction"
        in context[
            "questions"
        ]
    ):

        target = context[
            "target"
        ]

        predictors = [
            col
            for col in context[
                "numeric_columns"
            ]
            if col != target
        ]

        if target and predictors:

            results[
                "out_of_sample_regression"
            ] = train_test_regression(
                df,
                target,
                predictors,
            )

    return results


# ============================================================
# INTERPRETATION SCIENTIFIQUE
# ============================================================

def build_interpretation(
    analysis,
):

    statements = []

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    critical_quality = [
        msg
        for level, msg
        in analysis[
            "quality_issues"
        ]
        if level == "CRITIQUE"
    ]

    if critical_quality:

        statements.append(
            "L'étude présente au moins une anomalie critique "
            "de données qui doit être résolue avant toute conclusion forte."
        )

    else:

        statements.append(
            "Les données sont exploitables pour une phase "
            "d'analyse sous réserve des contrôles méthodologiques."
        )

    # --------------------------------------------------------
    # CIBLE
    # --------------------------------------------------------

    if not analysis[
        "target"
    ]:

        statements.append(
            "Aucune variable cible n'est confirmée. "
            "Une modélisation supervisée ne doit pas être considérée "
            "comme établie."
        )

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    correlation_result = analysis[
        "execution"
    ].get(
        "correlation"
    )

    if correlation_result:

        pearson = correlation_result[
            "pearson"
        ]

        if not pearson.empty:

            pairs = []

            columns = list(
                pearson.columns
            )

            for i in range(
                len(columns)
            ):

                for j in range(
                    i + 1,
                    len(columns),
                ):

                    value = pearson.iloc[
                        i,
                        j
                    ]

                    if pd.notna(value):

                        pairs.append(
                            (
                                columns[i],
                                columns[j],
                                abs(value),
                            )
                        )

            pairs.sort(
                key=lambda x: x[2],
                reverse=True,
            )

            if pairs:

                a, b, strength = pairs[0]

                statements.append(
                    f"L'association linéaire absolue la plus forte "
                    f"observée dans la matrice de Pearson concerne "
                    f"{a} et {b} (|r| ≈ {strength:.2f}). "
                    f"Cette observation ne constitue pas une preuve de causalité."
                )

    # --------------------------------------------------------
    # WEIBULL
    # --------------------------------------------------------

    weibull = analysis[
        "execution"
    ].get(
        "weibull"
    )

    if (
        weibull
        and weibull.get(
            "status"
        ) == "OK"
    ):

        statements.append(
            "Weibull : "
            + weibull[
                "interpretation"
            ]
            + " Cette interprétation porte sur la forme "
            "du modèle ajusté et non sur une cause physique identifiée."
        )

    # --------------------------------------------------------
    # KM
    # --------------------------------------------------------

    km = analysis[
        "execution"
    ].get(
        "kaplan_meier"
    )

    if (
        km
        and km.get(
            "status"
        ) == "OK"
    ):

        median = km[
            "median_survival"
        ]

        if median is not None:

            statements.append(
                f"L'estimation Kaplan-Meier donne une médiane "
                f"de survie estimée à {median:.3g} dans l'unité "
                f"du temps fourni."
            )

        else:

            statements.append(
                "La courbe Kaplan-Meier n'atteint pas 50 % "
                "de survie dans la période observée ; la médiane "
                "n'est donc pas estimable directement."
            )

    # --------------------------------------------------------
    # REGRESSION
    # --------------------------------------------------------

    regression = analysis[
        "execution"
    ].get(
        "linear_regression"
    )

    if (
        regression
        and regression.get(
            "status"
        ) == "OK"
    ):

        statements.append(
            f"La régression linéaire ajustée présente un "
            f"R² apparent de {regression['r2']:.3f}, "
            f"avec RMSE = {regression['rmse']:.3g}. "
            f"Ces indicateurs décrivent l'ajustement aux données "
            f"et ne suffisent pas à établir une capacité prédictive générale."
        )

    # --------------------------------------------------------
    # OUT OF SAMPLE
    # --------------------------------------------------------

    oos = analysis[
        "execution"
    ].get(
        "out_of_sample_regression"
    )

    if (
        oos
        and oos.get(
            "status"
        ) == "OK"
    ):

        statements.append(
            f"La validation hors échantillon donne "
            f"RMSE = {oos['rmse']:.3g}, "
            f"MAE = {oos['mae']:.3g} et "
            f"R² = {oos['r2']:.3f} sur l'échantillon de test."
        )

    # --------------------------------------------------------
    # ANOMALIES
    # --------------------------------------------------------

    anomaly = analysis[
        "execution"
    ].get(
        "anomaly_detection"
    )

    if (
        anomaly
        and anomaly.get(
            "status"
        ) == "OK"
    ):

        statements.append(
            f"La méthode de détection robuste identifie "
            f"{anomaly['n_anomalies']} observation(s) "
            f"statistiquement atypique(s). "
            f"Une expertise métier est nécessaire avant de les qualifier "
            f"d'anomalies physiques."
        )

    # --------------------------------------------------------
    # PRINCIPE FINAL
    # --------------------------------------------------------

    statements.append(
        "Les résultats statistiques, leur interprétation "
        "scientifique et la décision industrielle sont trois niveaux "
        "distincts et doivent rester séparés."
    )

    return statements


# ============================================================
# PIPELINE
# ============================================================

def build_pipeline():

    return [
        "01 — Formalisation du problème",
        "02 — Classification scientifique",
        "03 — Compréhension des données",
        "04 — Contrôle de qualité",
        "05 — Reconstruction du phénomène",
        "06 — Identification de l'unité d'étude",
        "07 — Identification de Y",
        "08 — Identification de X",
        "09 — Formulation des hypothèses",
        "10 — Scientific Knowledge Base",
        "11 — Decision Engine",
        "12 — Validation des conditions",
        "13 — Exécution des méthodes",
        "14 — Validation des résultats",
        "15 — Interprétation scientifique",
        "16 — Traduction industrielle",
        "17 — Décision / action",
        "18 — Rapport scientifique",
    ]


# ============================================================
# MOTEUR PRINCIPAL
# ============================================================

def run_scientific_engine(
    problem,
    objective,
    df,
    confirmations,
):

    # --------------------------------------------------------
    # 1. STANDARDISATION
    # --------------------------------------------------------

    df = standardize_dataframe(
        df
    )

    # --------------------------------------------------------
    # 2. CLASSIFICATION
    # --------------------------------------------------------

    domains, questions = classify_problem(
        problem,
        objective,
    )

    # --------------------------------------------------------
    # 3. PROFIL
    # --------------------------------------------------------

    semantics = infer_variable_semantics(
        df
    )

    dataset_profile = characterize_dataset(
        df
    )

    # --------------------------------------------------------
    # 4. QUALITE
    # --------------------------------------------------------

    quality_issues = inspect_data_quality(
        df
    )

    # --------------------------------------------------------
    # 5. UNITE
    # --------------------------------------------------------

    study_unit_info = identify_study_unit(
        df,
        semantics,
    )

    study_unit = confirmations.get(
        "study_unit"
    )

    if not study_unit:

        study_unit = (
            study_unit_info[
                "candidate"
            ]
        )

    # --------------------------------------------------------
    # 6. CIBLES
    # --------------------------------------------------------

    targets = identify_candidate_targets(
        df,
        semantics,
    )

    target = confirmations.get(
        "target"
    )

    # IMPORTANT :
    # aucune cible n'est imposée si l'utilisateur
    # ne la confirme pas.
    if (
        target
        and target not in df.columns
    ):

        target = None

    # --------------------------------------------------------
    # 7. PREDICTEURS
    # --------------------------------------------------------

    predictors = identify_predictors(
        df,
        target,
    )

    # --------------------------------------------------------
    # 8. CONTEXTE
    # --------------------------------------------------------

    context = build_scientific_context(
        df=df,
        target=target,
        unit=study_unit,
        domains=domains,
        questions=questions,
        confirmations=confirmations,
    )

    # --------------------------------------------------------
    # 9. HYPOTHESES
    # --------------------------------------------------------

    hypotheses = generate_hypotheses(
        problem,
        objective,
        domains,
        questions,
        df,
        target,
    )

    # --------------------------------------------------------
    # 10. KNOWLEDGE BASE
    # --------------------------------------------------------

    decision_table = decision_engine(
        SCIENTIFIC_KNOWLEDGE_BASE,
        context,
    )

    # --------------------------------------------------------
    # 11. QUESTIONS MANQUANTES
    # --------------------------------------------------------

    required_questions = (
        build_required_questions(
            domains,
            questions,
            context,
        )
    )

    # --------------------------------------------------------
    # 12. RECONSTRUCTION
    # --------------------------------------------------------

    reconstruction = build_reconstruction(
        domains,
        questions,
        context,
    )

    # --------------------------------------------------------
    # 13. CONDITIONS
    # --------------------------------------------------------

    validation_checks = (
        validate_conditions(
            df,
            context,
            decision_table,
        )
    )

    # --------------------------------------------------------
    # 14. EXECUTION
    # --------------------------------------------------------

    execution = execute_selected_methods(
        df,
        context,
        decision_table,
    )

    # --------------------------------------------------------
    # 15. ANALYSE
    # --------------------------------------------------------

    analysis = {

        "engine":
            ENGINE_NAME,

        "version":
            ENGINE_VERSION,

        "problem":
            problem,

        "objective":
            objective,

        "domains":
            domains,

        "questions":
            questions,

        "dataset_profile":
            dataset_profile,

        "semantics":
            semantics,

        "quality_issues":
            quality_issues,

        "study_unit":
            study_unit,

        "study_unit_candidates":
            study_unit_info[
                "candidates"
            ],

        "targets":
            targets,

        "target":
            target,

        "predictors":
            predictors,

        "hypotheses":
            hypotheses,

        "decision_context":
            context,

        "decision_table":
            decision_table,

        "required_questions":
            required_questions,

        "reconstruction":
            reconstruction,

        "validation_checks":
            validation_checks,

        "execution":
            execution,

        "pipeline":
            build_pipeline(),

        "generated_at":
            datetime.now().isoformat(),
    }

    # --------------------------------------------------------
    # 16. INTERPRETATION
    # --------------------------------------------------------

    analysis[
        "interpretation"
    ] = build_interpretation(
        analysis
    )

    return analysis


# ============================================================
# PDF
# ============================================================

def generate_pdf(
    analysis
):

    buffer = io.BytesIO()

    styles = getSampleStyleSheet()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.3 * cm,
        leftMargin=1.3 * cm,
        topMargin=1.3 * cm,
        bottomMargin=1.3 * cm,
    )

    story = []

    # --------------------------------------------------------
    # TITRE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "DOSSIER SCIENTIFIQUE D'INGÉNIERIE",
            styles["Title"],
        )
    )

    story.append(
        Paragraph(
            f"{ENGINE_NAME} — v{ENGINE_VERSION}",
            styles["Heading2"],
        )
    )

    story.append(
        Paragraph(
            f"Généré le "
            f"{datetime.now():%d/%m/%Y à %H:%M}",
            styles["Normal"],
        )
    )

    story.append(
        Spacer(
            1,
            0.5 * cm,
        )
    )

    # --------------------------------------------------------
    # PROBLEME
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. Problématique",
            styles["Heading2"],
        )
    )

    story.append(
        Paragraph(
            str(
                analysis["problem"]
            ),
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph(
            "Objectif : "
            + str(
                analysis["objective"]
            ),
            styles["BodyText"],
        )
    )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. Classification",
            styles["Heading2"],
        )
    )

    story.append(
        Paragraph(
            "Domaines : "
            + ", ".join(
                analysis["domains"]
            ),
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph(
            "Questions : "
            + ", ".join(
                analysis["questions"]
            ),
            styles["BodyText"],
        )
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "3. Données",
            styles["Heading2"],
        )
    )

    profile = analysis[
        "dataset_profile"
    ]

    story.append(
        Paragraph(
            f"Observations : {profile['n_observations']}",
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph(
            f"Variables : {profile['n_variables']}",
            styles["BodyText"],
        )
    )

    # --------------------------------------------------------
    # UNITE / CIBLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "4. Formalisation",
            styles["Heading2"],
        )
    )

    story.append(
        Paragraph(
            f"Unité d'étude : "
            f"{analysis['study_unit'] or 'Non confirmée'}",
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph(
            f"Variable Y : "
            f"{analysis['target'] or 'Non confirmée'}",
            styles["BodyText"],
        )
    )

    # --------------------------------------------------------
    # HYPOTHESES
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "5. Hypothèses",
            styles["Heading2"],
        )
    )

    for _, row in analysis[
        "hypotheses"
    ].iterrows():

        story.append(
            Paragraph(
                f"{row['ID']} — "
                f"{row['Hypothèse']}",
                styles["BodyText"],
            )
        )

    # --------------------------------------------------------
    # DECISION ENGINE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "6. Décision méthodologique",
            styles["Heading2"],
        )
    )

    rows = [
        [
            "Méthode",
            "Famille",
            "Score",
            "Statut",
        ]
    ]

    for _, row in analysis[
        "decision_table"
    ].head(25).iterrows():

        rows.append(
            [
                row["Méthode"],
                row["Famille"],
                str(
                    row[
                        "Score de compatibilité"
                    ]
                ),
                row["Statut"],
            ]
        )

    if len(rows) > 1:

        table = Table(
            rows,
            colWidths=[
                6.5 * cm,
                3.5 * cm,
                2 * cm,
                4 * cm,
            ],
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey,
                    ),

                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey,
                    ),

                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP",
                    ),
                ]
            )
        )

        story.append(
            table
        )

    # --------------------------------------------------------
    # QUESTIONS MANQUANTES
    # --------------------------------------------------------

    if not analysis[
        "required_questions"
    ].empty:

        story.append(
            Paragraph(
                "7. Informations manquantes",
                styles["Heading2"],
            )
        )

        for _, row in analysis[
            "required_questions"
        ].iterrows():

            story.append(
                Paragraph(
                    f"<b>{row['Élément']}</b> : "
                    f"{row['Question']}",
                    styles["BodyText"],
                )
            )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "8. Conditions de validité",
            styles["Heading2"],
        )
    )

    rows = [
        [
            "Contrôle",
            "Statut",
            "Commentaire",
        ]
    ]

    for _, row in analysis[
        "validation_checks"
    ].iterrows():

        rows.append(
            [
                row["Contrôle"],
                row["Statut"],
                row["Commentaire"],
            ]
        )

    table = Table(
        rows,
        colWidths=[
            4.5 * cm,
            3 * cm,
            9.5 * cm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
            ]
        )
    )

    story.append(
        table
    )

    # --------------------------------------------------------
    # RESULTATS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "9. Résultats",
            styles["Heading2"],
        )
    )

    execution = analysis[
        "execution"
    ]

    for method_id, result in execution.items():

        if not isinstance(
            result,
            dict,
        ):
            continue

        if result.get(
            "status"
        ) != "OK":
            continue

        method_name = (
            SCIENTIFIC_KNOWLEDGE_BASE
            .get(
                method_id,
                {}
            )
            .get(
                "name",
                method_id,
            )
        )

        story.append(
            Paragraph(
                method_name,
                styles["Heading3"],
            )
        )

        # Résultats simples
        for key in [
            "n",
            "events",
            "censored",
            "beta",
            "eta",
            "aic",
            "r2",
            "rmse",
            "mae",
            "accuracy",
            "precision",
            "recall",
            "n_anomalies",
            "median_survival",
        ]:

            if key in result:

                story.append(
                    Paragraph(
                        f"{key} : "
                        f"{result[key]}",
                        styles["BodyText"],
                    )
                )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "10. Interprétation scientifique",
            styles["Heading2"],
        )
    )

    for item in analysis[
        "interpretation"
    ]:

        story.append(
            Paragraph(
                "• " + item,
                styles["BodyText"],
            )
        )

    # --------------------------------------------------------
    # LIMITES
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "11. Limites",
            styles["Heading2"],
        )
    )

    story.append(
        Paragraph(
            "Le moteur distingue les observations statistiques, "
            "les résultats de modèles et les interprétations. "
            "Une association ne constitue pas une preuve de causalité. "
            "Une performance d'ajustement ne constitue pas une preuve "
            "de généralisation. Les conclusions restent limitées par "
            "la qualité, la représentativité et la structure des données.",
            styles["BodyText"],
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# EXPORT JSON
# ============================================================

def build_json_export(
    analysis
):

    return {
        "meta": {
            "engine":
                ENGINE_NAME,

            "version":
                ENGINE_VERSION,

            "generated_at":
                analysis[
                    "generated_at"
                ],
        },

        "problem": {
            "description":
                analysis[
                    "problem"
                ],

            "objective":
                analysis[
                    "objective"
                ],
        },

        "classification": {
            "domains":
                analysis[
                    "domains"
                ],

            "questions":
                analysis[
                    "questions"
                ],
        },

        "dataset": {
            "profile":
                json_safe(
                    analysis[
                        "dataset_profile"
                    ]
                ),

            "semantics":
                dataframe_to_json_records(
                    analysis[
                        "semantics"
                    ]
                ),
        },

        "formalization": {
            "study_unit":
                analysis[
                    "study_unit"
                ],

            "target":
                analysis[
                    "target"
                ],

            "predictors":
                dataframe_to_json_records(
                    analysis[
                        "predictors"
                    ]
                ),
        },

        "hypotheses":
            dataframe_to_json_records(
                analysis[
                    "hypotheses"
                ]
            ),

        "decision_engine":
            dataframe_to_json_records(
                analysis[
                    "decision_table"
                ]
            ),

        "required_information":
            dataframe_to_json_records(
                analysis[
                    "required_questions"
                ]
            ),

        "reconstruction":
            dataframe_to_json_records(
                analysis[
                    "reconstruction"
                ]
            ),

        "validation":
            dataframe_to_json_records(
                analysis[
                    "validation_checks"
                ]
            ),

        "interpretation":
            analysis[
                "interpretation"
            ],

        "pipeline":
            analysis[
                "pipeline"
            ],

        "execution":
            json_safe(
                analysis[
                    "execution"
                ]
            ),
    }


# ============================================================
# INTERFACE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '⚙️ Scientific Engineering Engine'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Méta-moteur générique de résolution scientifique '
    'de problèmes industriels.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "1 — Problème"
    )

    problem_desc = st.text_area(
        "Description du problème",
        height=180,
        placeholder=(
            "Décris le phénomène réel, "
            "ce que tu observes, "
            "ce que tu cherches à comprendre, "
            "expliquer, comparer, prédire, détecter "
            "ou optimiser."
        ),
    )

    objective = st.text_input(
        "Objectif",
        placeholder=(
            "Décrire / expliquer / comparer / "
            "prédire / détecter / optimiser..."
        ),
    )

    st.header(
        "2 — Données"
    )

    uploaded_file = st.file_uploader(
        "Importer un CSV ou Excel",
        type=[
            "csv",
            "xlsx",
        ],
    )

    st.caption(
        "Aucune donnée de démonstration n'est intégrée "
        "dans le moteur."
    )


# ============================================================
# CHARGEMENT
# ============================================================

if uploaded_file is None:

    st.info(
        "Importe un fichier CSV ou Excel pour commencer."
    )

    st.stop()


try:

    df = load_uploaded_data(
        uploaded_file
    )

    df = standardize_dataframe(
        df
    )

    st.session_state.df = df

except Exception as exc:

    st.error(
        f"Erreur de lecture : {exc}"
    )

    st.stop()


# ============================================================
# APERCU
# ============================================================

st.header(
    "3 — Données actives"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Observations",
    f"{len(df):,}",
)

col2.metric(
    "Variables",
    f"{len(df.columns):,}",
)

col3.metric(
    "Valeurs manquantes",
    f"{int(df.isna().sum().sum()):,}",
)

col4.metric(
    "Doublons",
    f"{int(df.duplicated().sum()):,}",
)

with st.expander(
    "Afficher les données",
    expanded=False,
):

    st.dataframe(
        df.head(100),
        use_container_width=True,
    )


# ============================================================
# PRE-ANALYSE SEMANTIQUE
# ============================================================

semantics_preview = infer_variable_semantics(
    df
)

candidate_targets = identify_candidate_targets(
    df,
    semantics_preview,
)

candidate_unit = identify_study_unit(
    df,
    semantics_preview,
)


# ============================================================
# FORMALISATION UTILISATEUR
# ============================================================

st.markdown(
    "---"
)

st.header(
    "4 — Formalisation scientifique"
)

st.info(
    "Le moteur peut proposer des candidats, "
    "mais les éléments scientifiques ambigus doivent "
    "être confirmés avant l'analyse."
)


# ============================================================
# CIBLE
# ============================================================

target_options = [
    "— Aucune cible confirmée —"
] + list(
    df.columns
)

target_default = 0

if (
    st.session_state.confirmations.get(
        "target"
    )
    in df.columns
):

    target_default = (
        target_options.index(
            st.session_state.confirmations[
                "target"
            ]
        )
    )

target_choice = st.selectbox(
    "Variable cible Y",
    target_options,
    index=target_default,
    help=(
        "Y est la grandeur ou l'événement que "
        "l'étude cherche à expliquer, comparer ou prédire."
    ),
)


# ============================================================
# UNITE D'ETUDE
# ============================================================

unit_options = [
    "— À confirmer —"
] + list(
    df.columns
)

unit_default = 0

if (
    st.session_state.confirmations.get(
        "study_unit"
    )
    in df.columns
):

    unit_default = (
        unit_options.index(
            st.session_state.confirmations[
                "study_unit"
            ]
        )
    )

unit_choice = st.selectbox(
    "Unité d'étude / identifiant",
    unit_options,
    index=unit_default,
    help=(
        "Variable permettant d'identifier l'entité "
        "sur laquelle porte réellement l'étude."
    ),
)


# ============================================================
# VARIABLES TEMPORELLES
# ============================================================

datetime_candidates = [
    col
    for col in df.columns
    if (
        pd.api.types.is_datetime64_any_dtype(
            df[col]
        )
        or (
            contains_any(
                col,
                [
                    "date",
                    "time",
                    "timestamp",
                    "heure",
                ],
            )
            and pd.to_datetime(
                df[col],
                errors="coerce",
            ).notna().mean()
            >= 0.70
        )
    )
]

time_options = [
    "— Aucune —"
] + datetime_candidates

time_default = 0

if (
    st.session_state.confirmations.get(
        "time_variable"
    )
    in datetime_candidates
):

    time_default = time_options.index(
        st.session_state.confirmations[
            "time_variable"
        ]
    )

time_choice = st.selectbox(
    "Variable temporelle",
    time_options,
    index=time_default,
)


# ============================================================
# CLASSIFICATION PREVIEW
# ============================================================

domains_preview, questions_preview = classify_problem(
    problem_desc,
    objective,
)

col1, col2 = st.columns(2)

with col1:

    st.write(
        "**Domaines détectés**"
    )

    for domain in domains_preview:

        st.info(domain)

with col2:

    st.write(
        "**Types de questions détectés**"
    )

    for question in questions_preview:

        st.info(question)


# ============================================================
# FIABILITE
# ============================================================

if "Fiabilité" in domains_preview:

    st.markdown(
        "---"
    )

    st.header(
        "Formalisation fiabilité / survie"
    )

    event_options = [
        "— Aucun événement confirmé —"
    ] + list(
        df.columns
    )

    previous_event = (
        st.session_state.confirmations.get(
            "event_variable"
        )
    )

    event_default = 0

    if previous_event in df.columns:

        event_default = (
            event_options.index(
                previous_event
            )
        )

    event_choice = st.selectbox(
        "Variable définissant l'événement",
        event_options,
        index=event_default,
        help=(
            "Cette variable doit représenter explicitement "
            "l'événement étudié. Une intervention ne doit "
            "pas être assimilée automatiquement à une défaillance."
        ),
    )

    time_to_event_confirmed = st.checkbox(
        "Le temps/exposition jusqu'à l'événement est correctement reconstruit",
        value=bool(
            st.session_state.confirmations.get(
                "time_to_event_confirmed",
                False,
            )
        ),
    )

    censoring_defined = st.checkbox(
        "La censure est explicitement définie",
        value=bool(
            st.session_state.confirmations.get(
                "censoring_defined",
                False,
            )
        ),
    )

else:

    event_choice = "— Aucun événement confirmé —"
    time_to_event_confirmed = False
    censoring_defined = False


# ============================================================
# OPTIMISATION
# ============================================================

if "Optimisation" in questions_preview:

    st.markdown(
        "---"
    )

    st.header(
        "Formalisation de l'optimisation"
    )

    controllable_candidates = [
        col
        for col in df.select_dtypes(
            include=np.number
        ).columns
    ]

    previous_controls = (
        st.session_state.confirmations.get(
            "controllable_variables",
            [],
        )
    )

    selected_controls = st.multiselect(
        "Variables réellement contrôlables",
        controllable_candidates,
        default=[
            c
            for c in previous_controls
            if c in controllable_candidates
        ],
    )

    objective_function = st.text_area(
        "Fonction objectif",
        value=(
            st.session_state.confirmations.get(
                "objective_function",
                "",
            )
        ),
        placeholder=(
            "Exemple générique : minimiser Y "
            "sous contraintes sur X1, X2..."
        ),
    )

    constraints = st.text_area(
        "Contraintes",
        value=(
            st.session_state.confirmations.get(
                "constraints",
                "",
            )
        ),
        placeholder=(
            "Limites physiques, économiques, "
            "opérationnelles, réglementaires..."
        ),
    )

else:

    selected_controls = []
    objective_function = ""
    constraints = ""


# ============================================================
# DETECTION
# ============================================================

normal_behavior_defined = False

if "Détection" in questions_preview:

    st.markdown(
        "---"
    )

    st.header(
        "Définition du comportement nominal"
    )

    normal_behavior_defined = st.checkbox(
        "Un comportement nominal / une référence peut être défini",
        value=bool(
            st.session_state.confirmations.get(
                "normal_behavior_defined",
                False,
            )
        ),
    )


# ============================================================
# ENREGISTRER FORMALISATION
# ============================================================

confirm_button = st.button(
    "✅ Confirmer la formalisation",
    type="primary",
    use_container_width=True,
)


if confirm_button:

    st.session_state.confirmations = {

        "target":
            None
            if target_choice
            == "— Aucune cible confirmée —"
            else
            target_choice,

        "study_unit":
            None
            if unit_choice
            == "— À confirmer —"
            else
            unit_choice,

        "time_variable":
            None
            if time_choice
            == "— Aucune —"
            else
            time_choice,

        "event_variable":
            None
            if event_choice
            == "— Aucun événement confirmé —"
            else
            event_choice,

        "time_to_event_confirmed":
            time_to_event_confirmed,

        "censoring_defined":
            censoring_defined,

        "controllable_variables":
            selected_controls,

        "objective_function":
            objective_function.strip(),

        "constraints":
            constraints.strip(),

        "normal_behavior_defined":
            normal_behavior_defined,
    }

    st.success(
        "Formalisation enregistrée."
    )


# ============================================================
# LANCEMENT
# ============================================================

st.markdown(
    "---"
)

launch = st.button(
    "🚀 Exécuter le moteur scientifique",
    type="primary",
    use_container_width=True,
)


if launch:

    if not problem_desc.strip():

        st.warning(
            "Décris d'abord le problème."
        )

        st.stop()

    confirmations = (
        st.session_state.confirmations
    )

    with st.spinner(
        "Le moteur construit et vérifie la démarche scientifique..."
    ):

        analysis = run_scientific_engine(
            problem=
                problem_desc,

            objective=
                objective,

            df=
                df,

            confirmations=
                confirmations,
        )

        st.session_state.analysis = (
            analysis
        )


# ============================================================
# RESULTATS
# ============================================================

analysis = (
    st.session_state.analysis
)


if analysis is None:

    st.info(
        "Confirme la formalisation puis lance le moteur scientifique."
    )

    st.stop()


# ============================================================
# 5 — FORMALISATION
# ============================================================

st.markdown(
    "---"
)

st.header(
    "5 — Formalisation retenue"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.write(
        "**Unité d'étude**"
    )

    st.info(
        analysis[
            "study_unit"
        ]
        or
        "Non confirmée"
    )

with col2:

    st.write(
        "**Variable Y**"
    )

    st.info(
        analysis[
            "target"
        ]
        or
        "Non confirmée"
    )

with col3:

    st.write(
        "**Variable temporelle**"
    )

    st.info(
        analysis[
            "decision_context"
        ].get(
            "time_variable"
        )
        or
        "Aucune"
    )


# ============================================================
# 6 — CLASSIFICATION
# ============================================================

st.header(
    "6 — Classification scientifique"
)

col1, col2 = st.columns(2)

with col1:

    st.subheader(
        "Domaines"
    )

    for item in analysis[
        "domains"
    ]:

        st.info(item)

with col2:

    st.subheader(
        "Questions"
    )

    for item in analysis[
        "questions"
    ]:

        st.info(item)


# ============================================================
# 7 — DATASET
# ============================================================

st.header(
    "7 — Compréhension des données"
)

profile = analysis[
    "dataset_profile"
]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Observations",
    profile[
        "n_observations"
    ],
)

col2.metric(
    "Variables",
    profile[
        "n_variables"
    ],
)

col3.metric(
    "Quantitatives",
    profile[
        "numeric_count"
    ],
)

col4.metric(
    "Catégorielles",
    profile[
        "categorical_count"
    ],
)

st.dataframe(
    analysis[
        "semantics"
    ],
    use_container_width=True,
)


# ============================================================
# 8 — DATA QUALITY
# ============================================================

st.header(
    "8 — Data Quality Engine"
)

if not analysis[
    "quality_issues"
]:

    st.success(
        "Aucune anomalie élémentaire détectée."
    )

else:

    for level, message in analysis[
        "quality_issues"
    ]:

        if level == "CRITIQUE":

            st.error(
                f"{level} — {message}"
            )

        elif level in [
            "IMPORTANT",
            "AVERTISSEMENT",
        ]:

            st.warning(
                f"{level} — {message}"
            )

        else:

            st.info(
                f"{level} — {message}"
            )


# ============================================================
# 9 — RECONSTRUCTION
# ============================================================

st.header(
    "9 — Reconstruction du phénomène"
)

st.dataframe(
    analysis[
        "reconstruction"
    ],
    use_container_width=True,
)


# ============================================================
# 10 — Y / X
# ============================================================

st.header(
    "10 — Y et X"
)

st.subheader(
    "Variable cible Y"
)

if analysis[
    "target"
]:

    st.success(
        f"Y = **{analysis['target']}**"
    )

else:

    st.warning(
        "Aucune cible confirmée."
    )

st.subheader(
    "Variables X candidates"
)

st.dataframe(
    analysis[
        "predictors"
    ],
    use_container_width=True,
)


# ============================================================
# 11 — HYPOTHESES
# ============================================================

st.header(
    "11 — Hypothèses scientifiques"
)

st.dataframe(
    analysis[
        "hypotheses"
    ],
    use_container_width=True,
)


# ============================================================
# 12 — QUESTIONS MANQUANTES
# ============================================================

st.header(
    "12 — Informations scientifiques manquantes"
)

required_questions = analysis[
    "required_questions"
]

if required_questions.empty:

    st.success(
        "Aucune information scientifique obligatoire "
        "supplémentaire détectée."
    )

else:

    st.dataframe(
        required_questions,
        use_container_width=True,
    )


# ============================================================
# 13 — DECISION ENGINE
# ============================================================

st.markdown(
    "---"
)

st.header(
    "13 — Scientific Knowledge Base + Decision Engine"
)

st.caption(
    "Le score présenté ici mesure uniquement la compatibilité "
    "structurelle avec les exigences déclarées de la méthode. "
    "Ce n'est pas un score de qualité scientifique."
)

decision_table = analysis[
    "decision_table"
]

st.dataframe(
    decision_table,
    use_container_width=True,
)


# ============================================================
# DETAILS DECISION
# ============================================================

for _, row in decision_table.iterrows():

    status = row[
        "Statut"
    ]

    title = (
        f"{row['Méthode']} — "
        f"{status}"
    )

    if status == "COMPATIBLE":

        with st.expander(
            "🟢 " + title,
            expanded=False,
        ):

            st.write(
                "**Conditions satisfaites :**",
                row[
                    "Conditions satisfaites"
                ],
            )

            st.write(
                "**Sorties :**",
                row[
                    "Sorties"
                ],
            )

            st.write(
                "**Limites :**",
                row[
                    "Limites"
                ],
            )

    elif status == "CONDITIONNEL":

        with st.expander(
            "🟠 " + title,
            expanded=False,
        ):

            st.write(
                "**Conditions satisfaites :**",
                row[
                    "Conditions satisfaites"
                ],
            )

            st.write(
                "**Informations manquantes :**",
                row[
                    "Informations manquantes"
                ],
            )

    else:

        with st.expander(
            "🔴 " + title,
            expanded=False,
        ):

            st.write(
                "**Informations manquantes :**",
                row[
                    "Informations manquantes"
                ],
            )


# ============================================================
# 14 — CONDITIONS
# ============================================================

st.header(
    "14 — Validation des conditions"
)

st.dataframe(
    analysis[
        "validation_checks"
    ],
    use_container_width=True,
)


# ============================================================
# 15 — EXECUTION DES METHODES
# ============================================================

st.header(
    "15 — Analyse scientifique exécutée"
)

execution = analysis[
    "execution"
]

if not execution:

    st.warning(
        "Aucune méthode compatible n'a pu être exécutée "
        "avec les informations actuellement disponibles."
    )


# ============================================================
# STATISTIQUES
# ============================================================

if "descriptive_statistics" in execution:

    result = execution[
        "descriptive_statistics"
    ]

    if not result[
        "numeric"
    ].empty:

        st.subheader(
            "Statistiques quantitatives"
        )

        st.dataframe(
            result[
                "numeric"
            ],
            use_container_width=True,
        )

    if not result[
        "categorical"
    ].empty:

        st.subheader(
            "Statistiques catégorielles"
        )

        st.dataframe(
            result[
                "categorical"
            ],
            use_container_width=True,
        )


# ============================================================
# CORRELATION
# ============================================================

if "correlation" in execution:

    st.subheader(
        "Corrélation de Pearson"
    )

    st.dataframe(
        execution[
            "correlation"
        ][
            "pearson"
        ],
        use_container_width=True,
    )

    st.subheader(
        "Corrélation de Spearman"
    )

    st.dataframe(
        execution[
            "correlation"
        ][
            "spearman"
        ],
        use_container_width=True,
    )


# ============================================================
# COMPARAISON
# ============================================================

if "group_comparison" in execution:

    result = execution[
        "group_comparison"
    ]

    if result.get(
        "status"
    ) == "OK":

        st.subheader(
            "Comparaison de groupes"
        )

        st.dataframe(
            result[
                "table"
            ],
            use_container_width=True,
        )

        if result.get(
            "test"
        ):

            st.json(
                result[
                    "test"
                ]
            )

    else:

        st.warning(
            result.get(
                "message",
                "Comparaison non exécutée.",
            )
        )


# ============================================================
# REGRESSION LINEAIRE
# ============================================================

if "linear_regression" in execution:

    result = execution[
        "linear_regression"
    ]

    if result.get(
        "status"
    ) == "OK":

        st.subheader(
            "Régression linéaire"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "R²",
            f"{result['r2']:.3f}",
        )

        col2.metric(
            "RMSE",
            f"{result['rmse']:.4g}",
        )

        col3.metric(
            "MAE",
            f"{result['mae']:.4g}",
        )

        st.dataframe(
            result[
                "coefficients"
            ],
            use_container_width=True,
        )

        validation = execution.get(
            "linear_regression_validation"
        )

        if validation:

            st.json(
                json_safe(
                    validation
                )
            )

    else:

        st.warning(
            result.get(
                "message",
                "Régression non exécutée.",
            )
        )


# ============================================================
# VALIDATION HORS ECHANTILLON
# ============================================================

if "out_of_sample_regression" in execution:

    result = execution[
        "out_of_sample_regression"
    ]

    st.subheader(
        "Validation hors échantillon"
    )

    if result.get(
        "status"
    ) == "OK":

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "R² test",
            f"{result['r2']:.3f}",
        )

        col2.metric(
            "RMSE test",
            f"{result['rmse']:.4g}",
        )

        col3.metric(
            "MAE test",
            f"{result['mae']:.4g}",
        )

        st.dataframe(
            result[
                "predictions"
            ],
            use_container_width=True,
        )

    else:

        st.warning(
            result.get(
                "message",
                "Validation hors échantillon non disponible.",
            )
        )


# ============================================================
# LOGISTIQUE
# ============================================================

if "logistic_regression" in execution:

    result = execution[
        "logistic_regression"
    ]

    st.subheader(
        "Régression logistique"
    )

    if result.get(
        "status"
    ) == "OK":

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{result['accuracy']:.3f}",
        )

        col2.metric(
            "Precision",
            f"{result['precision']:.3f}",
        )

        col3.metric(
            "Recall",
            f"{result['recall']:.3f}",
        )

        st.dataframe(
            result[
                "coefficients"
            ],
            use_container_width=True,
        )

    else:

        st.warning(
            result.get(
                "message",
                "Régression logistique non exécutée.",
            )
        )


# ============================================================
# KAPLAN-MEIER
# ============================================================

if "kaplan_meier" in execution:

    result = execution[
        "kaplan_meier"
    ]

    st.subheader(
        "Kaplan-Meier"
    )

    if result.get(
        "status"
    ) == "OK":

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Observations",
            result[
                "n"
            ],
        )

        col2.metric(
            "Événements",
            result[
                "events"
            ],
        )

        col3.metric(
            "Censurées",
            result[
                "censored"
            ],
        )

        if result[
            "median_survival"
        ] is not None:

            st.info(
                "Médiane de survie estimée : "
                f"{result['median_survival']:.4g}"
            )

        curve = result[
            "curve"
        ]

        fig, ax = plt.subplots(
            figsize=(9, 4)
        )

        ax.step(
            curve["Temps"],
            curve["Survie"],
            where="post",
        )

        ax.set_xlabel(
            "Temps / exposition"
        )

        ax.set_ylabel(
            "Probabilité de survie"
        )

        ax.set_title(
            "Courbe de Kaplan-Meier"
        )

        ax.set_ylim(
            0,
            1.05,
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True,
        )

    else:

        st.warning(
            result.get(
                "message",
                "Kaplan-Meier non exécuté.",
            )
        )


# ============================================================
# WEIBULL
# ============================================================

if "weibull" in execution:

    result = execution[
        "weibull"
    ]

    st.subheader(
        "Modèle de Weibull"
    )

    if result.get(
        "status"
    ) == "OK":

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "β — forme",
            f"{result['beta']:.4g}",
        )

        col2.metric(
            "η — échelle",
            f"{result['eta']:.4g}",
        )

        col3.metric(
            "AIC",
            f"{result['aic']:.4g}",
        )

        st.info(
            result[
                "interpretation"
            ]
        )

        curve = result[
            "curve"
        ]

        fig, ax = plt.subplots(
            figsize=(9, 4)
        )

        ax.plot(
            curve["Temps"],
            curve["Fiabilité"],
        )

        ax.set_xlabel(
            "Temps / exposition"
        )

        ax.set_ylabel(
            "Fiabilité R(t)"
        )

        ax.set_title(
            "Fonction de fiabilité Weibull"
        )

        ax.set_ylim(
            0,
            1.05,
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True,
        )

    else:

        st.warning(
            result.get(
                "message",
                "Weibull non exécuté.",
            )
        )


# ============================================================
# SERIE TEMPORELLE
# ============================================================

if "time_series" in execution:

    result = execution[
        "time_series"
    ]

    st.subheader(
        "Analyse temporelle"
    )

    if result.get(
        "status"
    ) == "OK":

        data = result[
            "data"
        ]

        st.write(
            f"Moyenne première période : "
            f"{result['first_mean']:.4g}"
        )

        st.write(
            f"Moyenne dernière période : "
            f"{result['last_mean']:.4g}"
        )

        fig, ax = plt.subplots(
            figsize=(9, 4)
        )

        target = analysis[
            "target"
        ]

        time_variable = analysis[
            "decision_context"
        ][
            "time_variable"
        ]

        ax.plot(
            data[
                time_variable
            ],
            data[
                target
            ],
        )

        ax.plot(
            data[
                time_variable
            ],
            data[
                "Moyenne_mobile"
            ],
        )

        ax.set_title(
            "Évolution temporelle"
        )

        ax.set_xlabel(
            "Temps"
        )

        ax.set_ylabel(
            target
        )

        fig.autofmt_xdate()

        fig.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True,
        )

    else:

        st.warning(
            result.get(
                "message",
                "Analyse temporelle non exécutée.",
            )
        )


# ============================================================
# ANOMALIES
# ============================================================

if "anomaly_detection" in execution:

    result = execution[
        "anomaly_detection"
    ]

    st.subheader(
        "Détection d'anomalies"
    )

    if result.get(
        "status"
    ) == "OK":

        st.metric(
            "Observations atypiques",
            result[
                "n_anomalies"
            ],
        )

        st.dataframe(
            result[
                "data"
            ],
            use_container_width=True,
        )

        st.caption(
            "Une observation statistiquement atypique "
            "ne constitue pas automatiquement une anomalie physique."
        )

    else:

        st.warning(
            result.get(
                "message",
                "Détection non exécutée.",
            )
        )


# ============================================================
# 16 — INTERPRETATION
# ============================================================

st.markdown(
    "---"
)

st.header(
    "16 — Interprétation scientifique"
)

for statement in analysis[
    "interpretation"
]:

    st.write(
        "• " + statement
    )


# ============================================================
# 17 — DECISION
# ============================================================

st.header(
    "17 — Décision / action"
)

critical_count = sum(
    1
    for level, _
    in analysis[
        "quality_issues"
    ]
    if level == "CRITIQUE"
)

blocked_methods = 0

if not decision_table.empty:

    blocked_methods = int(
        (
            decision_table[
                "Statut"
            ]
            == "BLOQUÉ"
        ).sum()
    )

conditional_methods = 0

if not decision_table.empty:

    conditional_methods = int(
        (
            decision_table[
                "Statut"
            ]
            == "CONDITIONNEL"
        ).sum()
    )


if critical_count:

    st.error(
        "DÉCISION BLOQUÉE : "
        "des problèmes critiques de données doivent être résolus."
    )

elif not required_questions.empty:

    st.warning(
        "DÉCISION CONDITIONNELLE : "
        "des informations scientifiques doivent encore être confirmées."
    )

elif conditional_methods:

    st.warning(
        "Certaines méthodes restent conditionnelles."
    )

else:

    st.success(
        "La structure scientifique est compatible "
        "avec au moins une méthode exécutable."
    )

st.caption(
    "La décision industrielle finale doit être prise à partir "
    "des résultats validés, du contexte physique et des contraintes "
    "réelles du système."
)


# ============================================================
# 18 — STATUT GLOBAL
# ============================================================

st.header(
    "18 — Statut global de l'étude"
)

if critical_count > 0:

    st.error(
        "🔴 BLOQUÉ"
    )

elif not required_questions.empty:

    st.warning(
        "🟠 INFORMATIONS REQUISES"
    )

elif blocked_methods > 0:

    st.warning(
        "🟠 MÉTHODES PARTIELLEMENT BLOQUÉES"
    )

else:

    st.success(
        "🟢 STRUCTURE SCIENTIFIQUE COMPATIBLE"
    )


# ============================================================
# 19 — PIPELINE
# ============================================================

st.header(
    "19 — Pipeline scientifique exécuté"
)

st.code(
    "\n↓\n".join(
        analysis[
            "pipeline"
        ]
    ),
    language="text",
)


# ============================================================
# 20 — PDF
# ============================================================

st.markdown(
    "---"
)

st.header(
    "20 — Dossier scientifique"
)

pdf = generate_pdf(
    analysis
)

st.download_button(
    label=
        "📥 Télécharger le rapport scientifique PDF",

    data=
        pdf,

    file_name=
        (
            "Scientific_Engineering_Report_"
            f"{datetime.now():%Y%m%d_%H%M}.pdf"
        ),

    mime=
        "application/pdf",

    use_container_width=
        True,
)


# ============================================================
# 21 — JSON
# ============================================================

st.header(
    "21 — Protocole scientifique JSON"
)

export_data = build_json_export(
    analysis
)

json_string = json.dumps(
    export_data,
    ensure_ascii=False,
    indent=2,
    default=str,
)

st.download_button(
    label=
        "🧠 Exporter le protocole scientifique JSON",

    data=
        json_string,

    file_name=
        "scientific_protocol.json",

    mime=
        "application/json",

    use_container_width=
        True,
)


# ============================================================
# 22 — KNOWLEDGE BASE
# ============================================================

with st.expander(
    "🧠 Voir la base de connaissances scientifique",
    expanded=False,
):

    kb_rows = []

    for method_id, method in (
        SCIENTIFIC_KNOWLEDGE_BASE.items()
    ):

        kb_rows.append(
            {
                "ID":
                    method_id,

                "Méthode":
                    method[
                        "name"
                    ],

                "Famille":
                    method[
                        "family"
                    ],

                "Objectifs":
                    ", ".join(
                        method[
                            "objectives"
                        ]
                    ),

                "Exigences":
                    ", ".join(
                        method.get(
                            "requires",
                            [],
                        )
                    ),
            }
        )

    st.dataframe(
        pd.DataFrame(
            kb_rows
        ),
        use_container_width=True,
    )


# ============================================================
# 23 — LIMITES SCIENTIFIQUES
# ============================================================

with st.expander(
    "⚠️ Principes et limites scientifiques",
    expanded=False,
):

    st.markdown(
        """
        ### Le moteur applique les principes suivants

        **1. Association ≠ causalité**

        Une corrélation statistique ne permet pas à elle seule
        d'affirmer qu'une variable provoque une autre.

        **2. Intervention ≠ événement scientifique**

        Une ligne de données n'est pas automatiquement une
        défaillance, un remplacement, une panne ou une fin de cycle.

        **3. Ajustement ≠ prédiction**

        Une bonne performance sur les données utilisées pour
        construire un modèle ne prouve pas sa capacité à généraliser.

        **4. Prédiction → contrôle du leakage**

        Les informations qui ne seraient pas disponibles au moment
        réel de la prédiction ne doivent pas être utilisées comme
        prédicteurs.

        **5. Fiabilité → reconstruction des cycles**

        Kaplan-Meier et Weibull nécessitent une définition correcte
        du temps jusqu'à événement et de la censure.

        **6. Optimisation → variables contrôlables**

        Une variable observée n'est pas automatiquement une variable
        de décision.

        **7. Anomalie statistique ≠ anomalie physique**

        Une observation atypique doit être interprétée dans son
        contexte scientifique et industriel.

        **8. Les scores de compatibilité ne sont pas des scores
        de qualité scientifique.**

        Ils indiquent seulement si la structure actuellement connue
        correspond aux exigences formelles de la méthode.
        """
    )


# ============================================================
# FIN
# ============================================================
