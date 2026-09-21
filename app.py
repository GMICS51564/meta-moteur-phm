
# KIMPA 3.4 -> 4.0 : BLOCS CORRIGÉS
# Remplacer les fonctions correspondantes dans app.py.
# Les blocs ci-dessous corrigent :
# 1) cible Oui/Non envoyée directement vers astype(float)
# 2) encodage robuste des catégories
# 3) valeurs manquantes
# 4) identifiants et dates utilisés à tort comme prédicteurs
# 5) leakage post-événement
# 6) accuracy calculée sur le jeu d'entraînement
# 7) validation train/test
# 8) séparation temporelle si une date existe
# 9) coefficients présentés comme causalité
# 10) SciPy inutile pour la régression logistique

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        f1_score, precision_score, recall_score, roc_auc_score,
        confusion_matrix,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


POSITIVE_LABELS = {
    "1", "oui", "yes", "true", "vrai", "failure", "fail", "failed",
    "panne", "defaillance", "defaut", "echec", "casse", "rupture",
    "rebut", "non_conforme", "non_conformite", "anomalie", "alerte",
}

NEGATIVE_LABELS = {
    "0", "non", "no", "false", "faux", "normal", "ok",
    "sans_panne", "sans_defaillance", "conforme", "success",
    "succes", "aucun",
}


def encode_binary_target(series):
    """Encode une cible binaire numérique, booléenne ou textuelle en 0/1."""
    valid = series.dropna()

    if valid.empty:
        return None, None, "La cible est entièrement vide."

    unique = pd.unique(valid)

    if len(unique) != 2:
        return (
            None,
            None,
            f"La cible doit être binaire : {len(unique)} modalités détectées."
        )

    if pd.api.types.is_bool_dtype(valid):
        mapping = {False: 0, True: 1}
        return series.map(mapping), mapping, None

    if pd.api.types.is_numeric_dtype(valid):
        vals = sorted(pd.to_numeric(unique).tolist())
        if set(vals) == {0, 1}:
            mapping = {0: 0, 1: 1}
        else:
            mapping = {vals[0]: 0, vals[1]: 1}
        return series.map(mapping), mapping, None

    normalized = {v: normalize_name(v) for v in unique}
    positive = [
        original for original, norm in normalized.items()
        if norm in POSITIVE_LABELS or contains_any(norm, POSITIVE_LABELS)
    ]
    negative = [
        original for original, norm in normalized.items()
        if norm in NEGATIVE_LABELS or contains_any(norm, NEGATIVE_LABELS)
    ]

    if len(positive) == 1 and len(negative) == 1:
        mapping = {negative[0]: 0, positive[0]: 1}
    else:
        # Fallback déterministe : il est signalé dans le résultat.
        ordered = sorted(unique, key=lambda x: str(x))
        mapping = {ordered[0]: 0, ordered[1]: 1}

    return series.map(mapping), mapping, None


def identify_identifier_columns(df, study_unit=None):
    excluded = set()

    for col in df.columns:
        n = normalize_name(col)

        if col == study_unit:
            excluded.add(col)
            continue

        # Ne pas considérer "machine", "equipment", "ligne", etc.
        # comme ID automatiquement : ce sont souvent des facteurs explicatifs.
        if (
            n in {
                "id", "identifiant", "asset_id", "equipment_id",
                "machine_id", "serial", "serial_number",
                "numero", "numero_serie", "reference", "ref"
            }
            or n.endswith("_id")
            or n.startswith("id_")
        ):
            excluded.add(col)

    return excluded


def identify_post_event_columns(df, target, confirmations):
    """Détecte les variables susceptibles de décrire l'événement après coup."""
    leakage = set()
    event_var = confirmations.get("event_variable")

    post_event_keywords = [
        "type_defaillance",
        "cause",
        "nature_intervention",
        "intervention",
        "reparation",
        "remplacement",
        "reset",
        "duree_arret",
        "temps_arret",
        "solution",
        "action_corrective",
        "diagnostic",
        "commentaire",
        "description_panne",
    ]

    for col in df.columns:
        if col == target:
            continue

        if col == event_var:
            leakage.add(col)
            continue

        n = normalize_name(col)
        if any(k in n for k in post_event_keywords):
            leakage.add(col)

    return leakage


def select_predictor_columns(df, target, confirmations):
    excluded = {target}

    excluded |= identify_identifier_columns(
        df,
        confirmations.get("study_unit")
    )

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            excluded.add(col)
        elif contains_any(
            col,
            ["date", "datetime", "timestamp", "horodatage"]
        ):
            excluded.add(col)

        if contains_any(
            col,
            ["commentaire", "description", "notes", "observation"]
        ):
            excluded.add(col)

    leakage = identify_post_event_columns(
        df,
        target,
        confirmations
    )
    excluded |= leakage

    predictors = [
        col for col in df.columns
        if col not in excluded
    ]

    return predictors, sorted(excluded), sorted(leakage)


def make_one_hot_encoder():
    """Compatibilité sklearn récent + anciennes versions."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse=False,
        )


def build_preprocessor(X):
    numeric = list(X.select_dtypes(include=np.number).columns)
    categorical = list(
        X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns
    )

    transformers = []

    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )

    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical,
            )
        )

    if not transformers:
        return None

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def prepare_full_matrix(df, target, confirmations):
    if not SKLEARN_AVAILABLE:
        return {
            "status": "NON EXECUTABLE",
            "message": "scikit-learn est requis.",
        }

    if target not in df.columns:
        return {
            "status": "NON EXECUTABLE",
            "message": "La cible sélectionnée n'existe pas.",
        }

    y, mapping, error = encode_binary_target(df[target])

    if error:
        return {
            "status": "NON EXECUTABLE",
            "message": error,
        }

    predictors, excluded, leakage = select_predictor_columns(
        df,
        target,
        confirmations,
    )

    if not predictors:
        return {
            "status": "NON EXECUTABLE",
            "message": "Aucun prédicteur exploitable après filtrage.",
            "excluded_columns": excluded,
            "leakage_columns": leakage,
        }

    X = df[predictors].copy()

    # Les colonnes entièrement vides ne peuvent rien apporter.
    empty = [c for c in X.columns if X[c].isna().all()]
    if empty:
        X = X.drop(columns=empty)
        predictors = [c for c in predictors if c not in empty]

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)

    if y.nunique() != 2:
        return {
            "status": "NON EXECUTABLE",
            "message": "La cible ne possède plus deux classes après nettoyage.",
        }

    preprocessor = build_preprocessor(X)

    if preprocessor is None:
        return {
            "status": "NON EXECUTABLE",
            "message": "Aucun prédicteur numérique/catégoriel exploitable.",
        }

    return {
        "status": "OK",
        "X_raw": X,
        "y": y,
        "preprocessor": preprocessor,
        "predictors": predictors,
        "excluded_columns": excluded,
        "leakage_columns": leakage,
        "target_mapping": mapping,
    }


def temporal_train_test_split(X, y, df, time_variable):
    """80/20 temporel si possible, sinon split stratifié déterministe."""

    if time_variable and time_variable in df.columns:
        dates = pd.to_datetime(
            df.loc[X.index, time_variable],
            errors="coerce",
            dayfirst=True,
        )

        valid = dates.notna()

        if valid.sum() >= 10:
            ordered = dates[valid].sort_values().index
            split = int(len(ordered) * 0.80)

            if 0 < split < len(ordered):
                train_idx = ordered[:split]
                test_idx = ordered[split:]

                return (
                    X.loc[train_idx],
                    X.loc[test_idx],
                    y.loc[train_idx],
                    y.loc[test_idx],
                    "Séparation temporelle 80/20",
                )

    if len(X) < 10:
        return None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    except ValueError:
        # Dataset trop petit / classe trop rare.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        "Séparation train/test 80/20",
    )


def logistic_regression_analysis(df, target, confirmations):
    """
    Analyse binaire robuste.

    Important :
    - pas de astype(float) sur une cible textuelle ;
    - preprocessing appris uniquement sur train ;
    - score calculé sur test ;
    - variables post-événement exclues ;
    - coefficients = associations du modèle, pas causalité.
    """
    prepared = prepare_full_matrix(
        df,
        target,
        confirmations,
    )

    if prepared["status"] != "OK":
        return prepared

    X = prepared["X_raw"]
    y = prepared["y"]

    if len(X) < 10:
        return {
            "status": "NON EXECUTABLE",
            "message": (
                f"{len(X)} observations exploitables : "
                "jeu trop petit pour une validation train/test raisonnable."
            ),
            "excluded_columns": prepared["excluded_columns"],
            "leakage_columns": prepared["leakage_columns"],
        }

    split = temporal_train_test_split(
        X,
        y,
        df,
        confirmations.get("time_variable"),
    )

    if split is None:
        return {
            "status": "NON EXECUTABLE",
            "message": "Impossible de créer les jeux train/test.",
        }

    X_train, X_test, y_train, y_test, split_method = split

    if y_train.nunique() < 2:
        return {
            "status": "NON EXECUTABLE",
            "message": "Le train ne contient pas les deux classes.",
        }

    model = Pipeline(
        steps=[
            ("preprocessor", prepared["preprocessor"]),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    try:
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(
                accuracy_score(y_test, y_pred)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, y_pred)
            ),
            "precision": float(
                precision_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                )
            ),
            "roc_auc": (
                float(roc_auc_score(y_test, y_prob))
                if y_test.nunique() == 2
                else None
            ),
        }

        fitted_preprocessor = model.named_steps["preprocessor"]
        fitted_model = model.named_steps["model"]

        try:
            names = fitted_preprocessor.get_feature_names_out().tolist()
        except Exception:
            names = [
                f"Variable_{i+1}"
                for i in range(len(fitted_model.coef_[0]))
            ]

        coef = fitted_model.coef_[0]

        coef_df = pd.DataFrame(
            {
                "Variable transformée": names,
                "Coefficient standardisé": coef,
                "Valeur absolue": np.abs(coef),
                "Association": [
                    "positive" if c > 0
                    else "négative" if c < 0
                    else "neutre"
                    for c in coef
                ],
            }
        ).sort_values(
            "Valeur absolue",
            ascending=False,
        ).reset_index(drop=True)

        return {
            "status": "OK",
            "metrics": metrics,
            "confusion_matrix": confusion_matrix(
                y_test,
                y_pred,
            ).tolist(),
            "coefficients": coef_df,
            "split_method": split_method,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "target_mapping": prepared["target_mapping"],
            "predictors": prepared["predictors"],
            "excluded_columns": prepared["excluded_columns"],
            "leakage_columns": prepared["leakage_columns"],
        }

    except Exception as exc:
        return {
            "status": "NON EXECUTABLE",
            "message": f"Erreur pendant la régression logistique : {exc}",
            "excluded_columns": prepared["excluded_columns"],
            "leakage_columns": prepared["leakage_columns"],
        }


# ============================================================
# MODIFICATIONS A FAIRE DANS run_scientific_engine()
# ============================================================
#
# Remplacer :
#
#     res_log = logistic_regression_influence_analysis(df, target)
#
# par :
#
#     res_log = logistic_regression_analysis(
#         df,
#         target,
#         confirmations,
#     )
#
# Et remplacer l'interprétation :
#
#     "Les facteurs ayant le plus fort impact..."
#
# par :
#
#     "Les coefficients indiquent des associations conditionnelles
#      au modèle. Ils ne démontrent pas une causalité."
#
#
# ============================================================
# MODIFICATIONS A FAIRE DANS L'AFFICHAGE
# ============================================================
#
# Remplacer :
#
#     reg["accuracy"]
#
# par :
#
#     reg["metrics"]["accuracy"]
#
# et afficher aussi :
#
#     reg["metrics"]["balanced_accuracy"]
#     reg["metrics"]["precision"]
#     reg["metrics"]["recall"]
#     reg["metrics"]["f1"]
#     reg["metrics"]["roc_auc"]
#
# Ne plus appeler l'accuracy "précision du modèle".
# "Accuracy" = proportion de classifications correctes.
# "Precision" = précision au sens ML.
#
# Ne plus écrire :
#     "variables qui influencent"
# ou :
#     "impact"
#
# Préférer :
#     "variables associées à la cible"
#     "coefficient du modèle"
#
#
# ============================================================
# CORRECTION IMPORTANTE POUR LE PDF
# ============================================================
#
# Dans generate_pdf(), ne plus faire :
#
#     reg['accuracy']
#
# mais :
#
#     metrics = reg["metrics"]
#     metrics["accuracy"]
#
# Et afficher explicitement :
#     Accuracy test
#     Balanced accuracy
#     Precision
#     Recall
#     F1
#     ROC-AUC
#
#
# ============================================================
# ERREUR SCIENTIFIQUE A NE PAS REINTRODUIRE
# ============================================================
#
# Une colonne "Nature_Intervention", "Cause", "Type_Defaillance",
# "Duree_Arret", "Remplacement", "Reset_Organe", etc. ne doit pas
# automatiquement être utilisée pour prédire une panne future si ces
# informations ne sont connues QU'APRES la panne.
#
# Pour une vraie prédiction PHM :
#
#     X(t) --------------------> Y(t+h)
#       |
#       +-- informations disponibles AVANT t
#
# Il faut interdire :
#
#     X(t+h) / informations post-événement ---> Y(t+h)
#
# ============================================================
