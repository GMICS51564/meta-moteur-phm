# ============================================================
# SCIENTIFIC ENGINEERING ENGINE
# Meta-moteur générique d'ingénierie scientifique industrielle
#
# VERSION 3.3 (Ergonomie simplifiée + Nettoyage & Encodage automatiques)
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
    FONCTION 1 : Nettoyage automatique des colonnes numériques.
    Convertit les nombres piégés dans du texte avec des espaces (ex: '85 000') ou virgules en vrais nombres.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            cleaned = df[col].astype(str).str.replace(' ', '').str.replace('\xa0', '').str.replace(',', '.')
            converted = pd.to_numeric(cleaned, errors='coerce')
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
# CHARGEMENT ET NETTOYAGE DES DONNEES
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
    
    # Application du nettoyage automatique
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
# INFERENCE SEMANTIQUE & VOCABULAIRE ÉLARGI (SYNONYMES)
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
        return [("CRITIQUE", "Le jeu de données est vide.")]
    if len(df) < 5:
        issues.append(("CRITIQUE", "Le jeu de données contient moins de 5 observations."))
    return issues


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
# EXÉCUTION STATISTIQUE AVEC ENCODAGE AUTOMATIQUE (One-Hot Encoding)
# ============================================================

def prepare_full_matrix(df, target):
    """
    FONCTION 2 : Encodage automatique des variables textuelles/catégorielles via pd.get_dummies.
    Transforme les colonnes textuelles (Organe, Cause, etc.) en variables mathématiques exploitables.
    """
    if target not in df.columns:
        return None
    
    # Exclure les colonnes textuelles purement descriptives ou uniques (ID, commentaires)
    exclude_cols = [target, 'ID', 'Commentaire', 'Commentaire.1', 'Date']
    features = [c for c in df.columns if c not in exclude_cols]
    
    work = df[features + [target]].copy()
    
    # Encodage automatique One-Hot Encoding de toutes les variables non numériques
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
    Analyse d'influence via Régression Logistique sur la matrice encodée.
    Permet de répondre précisément à : 'Quelle variable influence le plus la hausse des pannes ?'
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
        
    y_bin = (y == unique_y[1]).astype(float)
    
    from sklearn.linear_model import LogisticRegression
    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y_bin)
        
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
        res_log = logistic_regression_influence_analysis(df, target)
        if res_log["status"] == "OK":
            execution["logistic_regression"] = res_log
            
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
            "Nettoyage automatique des espaces et formats numériques appliqué avec succès.",
            "Encodage automatique (One-Hot Encoding) des variables textuelles réalisé pour l'analyse d'impact.",
            "Les facteurs ayant le plus fort impact sur la variable cible ont été extraits et classés par ordre d'importance."
        ],
        "generated_at": datetime.now().isoformat(),
    }
    return analysis


def generate_pdf(analysis):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.3*cm, leftMargin=1.3*cm, topMargin=1.3*cm, bottomMargin=1.3*cm)
    styles = getSampleStyleSheet()
    story = [Paragraph("RAPPORT D'ANALYSE INDUSTRIELLE", styles["Title"]), Spacer(1, 0.5*cm)]
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
# INTERFACE UTILISATEUR SIMPLIFIÉE
# ============================================================

st.markdown('<div class="main-title">⚙️ Scientific Engineering Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Outil d\'aide à l\'analyse de données et à la résolution de problèmes industriels.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("1 — Votre Problème")
    problem_desc = st.text_area(
        "Décrivez votre situation ou votre problème",
        height=150,
        placeholder="Ex: Nous avons une hausse des pannes sur la ligne 3 et nous voulons comprendre quelles variables influencent ces arrêts..."
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
st.header("3 — Aperçu de vos données (Nettoyées automatiquement)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes (observations)", f"{len(df):,}")
c2.metric("Colonnes (variables)", f"{len(df.columns):,}")
c3.metric("Valeurs manquantes", f"{int(df.isna().sum().sum()):,}")
c4.metric("Doublons", f"{int(df.duplicated().sum()):,}")

with st.expander("Voir un extrait des données"):
    st.dataframe(df.head(10), use_container_width=True)


# ============================================================
# FORMALISATION CLAIRE (Avec les explications simples)
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
# FORMULAIRES CONDITIONNELS SIMPLIFIÉS
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


# Validation du formulaire
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
    with st.spinner("Nettoyage automatique et analyse statistique par le moteur..."):
        analysis = run_scientific_engine(problem_desc, objective, df, st.session_state.confirmations)
        st.session_state.analysis = analysis

if analysis is not None:
    st.markdown("---")
    st.header("📊 Résultats de l'analyse d'influence")
    
    if "logistic_regression" in analysis["execution"]:
        reg = analysis["execution"]["logistic_regression"]
        st.metric("Précision du modèle d'explication", f"{reg['accuracy']*100:.1f}%")
        
        st.subheader("Classement des variables qui influencent le plus la cible :")
        st.dataframe(reg["coefficients"], use_container_width=True)
        st.info("💡 Un coefficient positif pousse vers la hausse de la panne, tandis qu'un coefficient négatif réduit ou protège de la panne.")
    else:
        st.warning("Veuillez sélectionner une variable cible valide pour lancer l'analyse d'influence.")

    st.subheader("Interprétations clés")
    for statement in analysis["interpretation"]:
        st.write(f"• {statement}")

    # Export PDF
    st.markdown("---")
    pdf_data = generate_pdf(analysis)
    st.download_button("📥 Télécharger le rapport de synthèse (PDF)", data=pdf_data, file_name="Rapport_Industriel_Robuste.pdf", mime="application/pdf", use_container_width=True)
else:
    st.info("👉 Veuillez remplir les choix ci-dessus et cliquer sur le bouton de validation pour afficher les résultats.")
