# Modèle de Scoring de Crédit

## Le contexte
Je suis Data Scientist au sein d'une société financière, nommée "Prêt à dépenser", qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

"Prêt à dépenser", souhaite développer un outil de "scoring crédit" pour évaluer la probabilité de remboursement des crédits et classer les demandes en crédits accordés ou refusés. L'entreprise veut également répondre à la demande croissante de transparence de la part des clients. Pour ce faire, elle prévoit de créer un tableau de bord interactif permettant aux chargés de relation client d'expliquer les décisions d'octroi de crédit de manière transparente et de fournir aux clients un accès facile à leurs informations personnelles.

Les [données](https://www.kaggle.com/competitions/home-credit-default-risk/data ) nécessaires pour ce projet proviennent de la compétition Kaggle "Home Credit Default Risk", comprenant 10 fichiers avec 346 colonnes. Ces fichiers sont liés par des clés.
Pour simplifier le processus d'analyse, de préparation des données et le feature engineering, nous nous appuyons sur [un kernel](https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script) Kaggle déjà existant.

## La mission:
* Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
* Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.
* Mettre en production le modèle de scoring de prédiction à l’aide d’une API, ainsi que le dashboard interactif qui appelle l’API pour les prédictions.

## Organisation des Branches et des Dossiers

Pour le déploiement de l'API et du tableau de bord, nous avons choisi Heroku, qui nous permet de déployer des applications à partir de différentes branches. Pour simplifier la gestion, nous avons déployé l'API à partir de la branche principale (main branch) et le tableau de bord à partir de la branche spécifique (dashboard branch).

Vous trouverez les dossiers suivants dans

**la branche 'main' :**

- Les Jupyter Notebooks, la Note Méthodologique, la Présentation, le Data Drift Report, ainsi que les fichiers relatifs à l'API et au tableau de bord.

- .github/workflows : Contient le fichier 'python-app.yml' pour les actions GitHub.

- api :
  - 'api.py' : Fichier contenant le code de l'API.
  - Fichiers '.pkl' : Fichiers pour le modèle et le pipeline de prétraitement.

- dashboard : Les fichiers de la branche dashboard.

- jupyter_notebooks : Les Jupyter Notebooks pour les analyses exploratoires, la modélisation et les fonctions.

- mlruns : Les fichiers de Mlflow.

- tests : Le fichier pour le test de l'API.

- .gitignore : Utilisé pour ignorer certains fichiers lors de l'opération 'git push'.

- Procfile : Utilisé pour le déploiement de l'API sur Heroku.

- requirements : Les libraries nécessaires pour l'API sont listées dans ce fichier.

**la branche 'dashboard' :**

- data : Les données pour le dashboard 
- image : Les images pour le dasboard
- dashboard.py : Fichier contenant le code du dashboard
- Procfile et setup.sh :  Utilisés pour le déploiement du dashboard sur Heroku
- requirements : Pour ce projet, nous avons utilisé Python en version 3.9.17. Les libraries nécessaires sont listées dans ce fichier.

## Liens
* Le dossier Github : https://github.com/githubzey/p7_Home_Credit
* Api : https://apihomecredit-861d00eaed91.herokuapp.com/ 
* Dashboard : https://dashboardhomecredit-1913c1e69feb.herokuapp.com/
  
