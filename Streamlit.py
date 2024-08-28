import pandas as pd
import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import pickle
import shap
import numpy as np


def main():
    # 3. Récupération du modèle
    def modele():
        with open("mlflow_model/model.pkl", "rb") as mod_pickle:
            return pickle.load(mod_pickle)

        # 2. Module de prédiction
    def predict_proba(id_client):
        data_client = echantillon_clients.loc[[id_client]]
        perfect_model = modele()
        # Assurez-vous que data_client est un DataFrame à une seule ligne
        return perfect_model.predict_proba(data_client)[:, 1][0]  # Renvoie la probabilité pour la classe positive

    def info_client(ID):
        """Isole la ligne du client voulu"""
        data_client = echantillon_clients.loc[echantillon_clients.index == ID]
        print("Data Client:", data_client)  # Debugging line
        return data_client

    def prediction_1(client_id):
      pred = predict_proba (client_id)
      #pred = request_prediction(URI, client_id)
      print("Predictions:", pred)  # Debugging line
      return pred

    echantillon_clients = pd.read_csv("Data_test/echantillon_clients.csv", index_col="SK_ID_CURR")
    trainset = pd.read_csv("Data_test/trainset.csv")
    trainset_0 = trainset.loc[trainset["TARGET"] == 0].drop(columns=["TARGET"])
    trainset_1 = trainset.loc[trainset["TARGET"] == 1].drop(columns=["TARGET"])
    seuil = echantillon_clients.iloc[0]["threshold"]
    echantillon_clients = echantillon_clients.drop(columns=["threshold"])

    URI = "https://nadat-project7.onrender.com/predict"
                
    client_choice = st.sidebar.selectbox(
        "Quel client souhaitez-vous évaluer ?", echantillon_clients.index
    )

    # Saisie client
    data_client = info_client(client_choice)

    # SIDEBAR
    st.sidebar.write("Client :blue[{}]".format(client_choice))
    st.sidebar.write(
        "Âge : :orange[{}] ans".format(int(-data_client["DAYS_BIRTH"].values[0] / 365))
    )
    st.sidebar.write(
        "Nombre d'enfant(s) : :orange[{}]".format(
            int(data_client["CNT_CHILDREN"].values[0])
        )
    )
    st.sidebar.write(
        "Revenu total : :orange[{}] $".format(
            int(data_client["AMT_INCOME_TOTAL"].values[0])
        )
    )
    st.sidebar.write(
        "Ancienneté dans l'emploi : :orange[{}] an(s)".format(
            int(-data_client["DAYS_EMPLOYED"].values[0] / 365)
        )
    )
    st.sidebar.write(
        "Crédit sollicité : :orange[{}] $".format(
            int(data_client["AMT_CREDIT"].values[0])
        )
    )
    st.sidebar.write(
        "Annuité du prêt : :orange[{}] $".format(
            int(data_client["AMT_ANNUITY"].values[0])
        )
    )
    st.sidebar.write(
        "Prix du bien : :orange[{}] $".format(
            int(data_client["AMT_GOODS_PRICE"].values[0])
        )
    )

    # PAGE PRINCIPALE
    st.title("Dashboard Scoring Crédit")

    prediction_value = prediction_1(client_choice)

    if prediction_value < seuil:
        st.header("Client :blue[{}] : Crédit :green[accepté]".format(client_choice))
        st.subheader(
            "Risque de défaut = :green[{:.1f} %] - Seuil de décision = :orange[{:.1f} %]".format(
                prediction_value * 100, seuil * 100
            )
        )
    else:
        st.header("Client :blue[{}] : Crédit :red[refusé]".format(client_choice))
        st.subheader(
            "Risque de défaut = :red[{:.1f} %] - Seuil de décision = :orange[{:.1f} %]".format(
                prediction_value * 100, seuil * 100
            )
        )

    # Configuration de la figure en format demi-cercle
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': 'polar'})

    # Limiter les angles pour créer un demi-cercle
    ax.set_thetamax(180)  # Limite à l'angle 0 (12h)
    ax.set_thetamin(0)  # Limite à l'angle -180° (6h)

    # Masquer les graduations et les étiquettes radiales
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    # Créer les angles pour le demi-cercle
    angles = np.linspace(0, np.pi, 100)  # Demi-cercle de gauche (9h) à droite (3h)

    # Remplir la partie rouge à droite du seuil
    seuil_angle = np.pi - seuil * np.pi
    ax.fill_between(angles, 0, 1, where=(angles  <= seuil_angle), color='red')

    # Remplir la partie verte jusqu'au seuil
    ax.fill_between(angles, 0, 1, where=(angles >  seuil_angle), color='green')

    # Ligne indiquant la position de la prédiction
    prediction_angle = np.pi-prediction_value * np.pi
    ax.annotate('', xy=(prediction_angle, 1), xytext=(0, 0),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10))
                
    # Ajouter le texte pour le seuil
    ax.text(seuil_angle, 1.1, f'Seuil: {seuil*100:.0f}%', horizontalalignment='center', fontsize=12, color='orange')

    # Ajouter le texte pour la valeur prédite
    ax.text(prediction_angle, 1.1, f'Prédiction: {prediction_value*100:.0f}%', horizontalalignment='center', fontsize=12, color='blue')


    # Affichage dans Streamlit
    st.pyplot(fig)


    # Partie pour afficher les caractéristiques locales
    if st.checkbox(
        "Visualiser les principales caractéristiques pour le score du client :blue[{}]".format(
            client_choice
        )
    ):
        perfect_model = modele()
        f = lambda x: perfect_model.predict_proba(x)[:, 1]
        med = echantillon_clients.median().values.reshape(
            (1, echantillon_clients.shape[1])
        )
        explainer = shap.Explainer(f, med)
        shap_values = explainer(data_client, max_evals=1517)
        fig, ax = plt.subplots(figsize=(6, 6))
        shap.plots.waterfall(shap_values[0], max_display=10)
        st.pyplot(fig)

    # Partie pour afficher les caractéristiques globales
    if st.checkbox("Visualiser les principales caractéristiques globales pour le score"):
        perfect_model = modele()
        f = lambda x: perfect_model.predict_proba(x)[:, 1]
        med = echantillon_clients.median().values.reshape((1, echantillon_clients.shape[1]))
        explainer = shap.Explainer(f, med)
        shap_values = explainer(echantillon_clients, max_evals=1517)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        shap.summary_plot(shap_values, echantillon_clients, max_display=10, show=False)
        
        ax = plt.gca()  # Obtenir l'axe actuel
        if 'right' in ax.spines:
            ax.spines['right'].set_visible(False)
        if 'top' in ax.spines:
            ax.spines['top'].set_visible(False)
        
        st.pyplot(fig)


    # Distribution des features selon les classes avec positionnement du client
    if st.checkbox(
        "Situer le client :blue[{}] dans la distribution des caractéristiques".format(
            client_choice
        )
    ):
        caracteristique = st.selectbox(
            "Quelle caractéristique souhaitez-vous observer ?", trainset_0.columns
        )

        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(121)
        ax.hist(trainset_0[caracteristique], color="green", bins=20)
        ax.axvline(
            data_client[caracteristique].values[0],
            color="black",
            linewidth=4,
            linestyle="--",
        )
        ax.set(title="Crédits acceptés")
        ax = fig.add_subplot(122)
        ax.hist(trainset_1[caracteristique], color="red", bins=20)
        ax.axvline(
            data_client[caracteristique].values[0],
            color="black",
            linewidth=4,
            linestyle="--",
        )
        ax.set(title="Crédits refusés")
        st.pyplot(fig)

        st.write(
            "Valeur de la caractéristique pour le client :blue[{}] = :orange[{}]".format(
                client_choice, data_client[caracteristique].values[0]
            )
        )

if __name__ == "__main__":
    main()
