"""
Application Streamlit — Prédiction de l'état d'une voiture

Lancement en local :  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st

COLONNES_FEATURES = ["Marque", "Année", "Transmission", "Quartier", "Prix"]

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de l'état d'une voiture",
    page_icon="🚘",
    layout="centered",
)

DESCRIPTION = (
    "Ce modèle de machine learning permet de prédire l'état d'une voiture en partant "
    "de la marque, de l'année, de la transmission, du quartier de vente et du prix."
)


# Chargement des artefacts (mis en cache : chargés une seule fois)

@st.cache_resource
def load_artifacts():
    encoders = jb.load("encoder.joblib")   # encodeurs 
    scaler = jb.load("scaler.joblib")       # normaliseur
    xgb = jb.load("xgb_model.joblib")       # modèle
    return encoders, scaler, xgb


encoders, scaler, xgb = load_artifacts()

marque = st.selectbox(
    "Marque",
    options=encoders["Marque"].classes_
)

transmission = st.selectbox(
    "Transmission",
    options=encoders["Transmission"].classes_
)

quartier = st.selectbox(
    "Quartier",
    options=encoders["Quartier"].classes_
)
class_names = ["D'Occasion", "Venant"]
 

# Fonction de prédiction simple

def Pred_func(marque, annee, transmission, quartier, prix):

    entree = pd.DataFrame([{
        "Marque": marque,
        "Année": annee,
        "Transmission": transmission,
        "Quartier": quartier,
        "Prix": prix
    }])

    for col in ["Marque", "Transmission", "Quartier"]:
        entree[col] = encoders[col].transform(entree[col])

    x_new = scaler.transform(entree)

    y_pred = xgb.predict(x_new)

    return class_names[y_pred[0]]
 



# Fonction de prédiction multiple

def Pred_func_csv(file):
    # Lire le fichier csv
    df = pd.read_csv(file)
    predictions = []
    # Boucle sur les lignes du dataframe
    for row in df.iloc[:, :].values:
        # prédiction simple
        y_pred = Pred_func(row[0], row[1], row[2], row[3], row[4])
        predictions.append(y_pred)
    df["etat"] = predictions
    return df



# Interface

st.title("🚘 Prédiction de l'état d'une voiture")
 
onglet1, onglet2 = st.tabs(["Prédiction simple", "Prédiction multiple"])
 
# ----------------------------- Onglet 1 -------------------------------
with onglet1:
    st.subheader("Prédire l'état d'une voiture à partir d'une entrée")
    st.write(DESCRIPTION)
 
    with st.form("formulaire_simple"):
        col1, col2 = st.columns(2)
        with col1:
            marque = st.selectbox("Marque", options=sorted(encoders["Marque"].classes_))
            annee = st.number_input("Année", min_value=1990, max_value=2026, value=2015, step=1)
        with col2:
            transmission = st.selectbox("Transmission", options=sorted(encoders["Transmission"].classes_))
            quartier = st.selectbox("Quartier", options=sorted(encoders["Quartier"].classes_))
 
        prix = st.number_input("Prix (FCfa)", min_value=0, value=5_000_000, step=100_000)
 
        soumettre = st.form_submit_button("Prédire", type="primary")
 
    if soumettre:
        try:
            resultat = Pred_func(marque, annee, transmission, quartier, prix)
            st.success(f"**État prédit de la voiture :** {resultat}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
 
# ----------------------------- Onglet 2 -------------------------------
with onglet2:
    st.subheader("Prédire l'état de plusieurs voitures à partir d'un fichier CSV")
    st.write(DESCRIPTION)
    st.caption(
        "Le fichier CSV doit contenir, avec ces noms de colonnes exacts : "
        "Marque, Année, Transmission, Quartier, Prix."
    )
 
    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])
 
    if fichier is not None:
        try:
            with st.spinner("Prédictions en cours…"):
                df_resultat = Pred_func_csv(fichier)
 
            st.success(f"{len(df_resultat)} prédiction(s) effectuée(s).")
            st.dataframe(df_resultat, use_container_width=True)
 
            st.download_button(
                label="⬇️ Télécharger le fichier CSV",
                data=df_resultat.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")
 
