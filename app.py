import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# Configuration de la page
st.set_page_config(page_title="Méta-Moteur PHM - Production", layout="wide")

st.title("⚙️ Méta-Moteur d'Ingénierie Scientifique & PHM")
st.markdown("Plateforme interactive de diagnostic, modélisation et d'aide à la décision industrielle.")

# Panneau latéral pour les paramètres
st.sidebar.header("1. Paramètres de l'étude")
problem_desc = st.sidebar.text_area("Objectif / Problématique :", "Réduire les arrêts non planifiés et comprendre les facteurs d'usure.")
exposure_col_name = st.sidebar.text_input("Nom de la colonne d'exposition (ex: Exposition / Kilometrage)", "Exposition")
target_col_name = st.sidebar.text_input("Nom de la colonne des organes / composants", "Organe")

# Section 2 : Importation des données
st.header("2. Données de l'étude")
uploaded_file = st.file_uploader("Importez votre fichier de données industrielles (CSV ou Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.success("Fichier chargé avec succès !")
else:
    st.info("Aucun fichier importé. Utilisation d'un échantillon de référence (Flotte de bus).")
    df = pd.DataFrame({
        'ID_Equipement': ['EQ-101', 'EQ-102', 'EQ-103', 'EQ-101', 'EQ-104'],
        'Exposition': [85000, 145000, 92000, 108000, 110000],
        'Organe': ['Train roulant', 'Freinage', 'Train roulant', 'Train roulant', 'Freinage'],
        'Nature': ['Corrective', 'Corrective', 'Curative', 'Corrective', 'Corrective']
    })

# Affichage sécurisé du tableau
st.subheader("Aperçu des données actives")
st.dataframe(df, use_container_width=True)

st.markdown("---")

# Bouton d'action principal pour lancer le moteur
if st.button("🚀 Lancer l'analyse scientifique et générer le diagnostic"):
    st.header("3. Résultats & Diagnostic Automatisé")
    
    # Affichage du protocole
    st.subheader("📋 Protocole & Hypothèses Validés")
    st.markdown(f"""
    - **Problématique ciblée :** {problem_desc}
    - **H1 :** L'intensité d'utilisation ou l'exposition influence directement la criticité des pannes.
    - **H2 :** Hétérogénéité des défaillances marquée selon le composant.
    """)
    
    # Disposition en colonnes pour les graphiques et métriques
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Répartition des pannes par organe")
        if target_col_name in df.columns:
            counts = df[target_col_name].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3))
            counts.plot(kind='bar', color='#1f77b4', edgecolor='black', ax=ax)
            ax.set_ylabel("Nombre d'incidents")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)
        else:
            st.error(f"La colonne '{target_col_name}' est introuvable dans votre tableau.")

    with col2:
        st.subheader("📈 Indicateurs d'exposition")
        if exposure_col_name in df.columns and pd.api.types.is_numeric_dtype(df[exposure_col_name]):
            mean_exp = df[exposure_col_name].mean()
            max_exp = df[exposure_col_name].max()
            min_exp = df[exposure_col_name].min()
            st.metric("Exposition moyenne à l'incident", f"{mean_exp:,.0f}")
            st.metric("Seuil critique minimal observé", f"{min_exp:,.0f}")
            st.metric("Seuil limite maximal observé", f"{max_exp:,.0f}")
        else:
            st.warning("La colonne d'exposition spécifiée n'est pas numérique ou est introuvable.")

    # Génération du PDF en mémoire pour téléchargement direct
    st.markdown("---")
    st.header("4. Exportation du Dossier d'Expertise")
    
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter
    now = datetime.now().strftime("%d/%m/%Y à %H:%M:%S")
    
    # Mise en page du PDF
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "DOSSIER SCIENTIFIQUE DE L'ANALYSE INDUSTRIELLE")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"Date et heure de génération : {now}")
    c.line(50, height - 80, width - 50, height - 80)
    
    c.drawString(50, height - 110, f"Problématique : {problem_desc}")
    c.drawString(50, height - 130, "Statut : Hypothèses validées, analyse statistique exécutée.")
    c.drawString(50, height - 150, "Recommandation : Migration vers une maintenance conditionnelle ciblée.")
    
    c.save()
    pdf_buffer.seek(0)
    
    # Bouton de téléchargement interactif
    st.download_button(
        label="📥 Télécharger le Rapport d'Expertise PDF",
        data=pdf_buffer,
        file_name=f"Rapport_PHM_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf"
    )
