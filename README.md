# Rapport de Projet Machine Learning
---
## En-tête

- 🎓 Institut : **ISPM (Institut Supérieur Polytechnique de Madagascar)**
- 🌐 Site officiel : https://ispm-edu.com/ 

---
## Nom du Groupe

**Nom du groupe :** NSDuo  

---
## Membres du Groupe

- RAKOTONDRAZAKA Nameno Fanantenana IMTICIA 4 numéro 09
- RAJAONARIVONY Steve Marino ISAIA 4 numéro 01 
---
## Description du Projet

**Titre du projet :** Prédiction du résultat d’un jeu de morpion avec Machine Learning

**Objectif :**  
Ce projet vise à développer un modèle de machine learning capable de prédire le résultat d’une partie de morpion (victoire X, victoire O, ou match nul) à partir de l’état du plateau.

**Contexte :**  
Le morpion est un jeu simple permettant de tester différents modèles de classification.

---
## Structure du Repository

└── ressources
    ├── dataset.csv
    ├── model_draw.pkl
    ├── model_xwins.pkl
├── Etape1_a_Etape3.ipynb
├── game.py
├── generator.py
├── README.md
└── requirements.txt

---
## Résultats Machine Learning

| Model             | Target   |   Accuracy |   F1 Score |
|:------------------|:---------|-----------:|-----------:|
| Logistic          | x_wins   |   0.754639 |  0.855055  |
| Decision Tree     | x_wins   |   0.243299 |  0.0316623 |
| Random Forest     | x_wins   |   0.859794 |  0.912821  |
| Gradient Boosting | x_wins   |   0.82268  |  0.890306  |
| XGBoost           | x_wins   |   0.913402 |  0.943243  |
| MLP               | x_wins   |   0.91134  |  0.940853  |
| Logistic          | is_draw  |   0.583505 |  0.344156  |
| Decision Tree     | is_draw  |   0.804124 |  0.201681  |
| Random Forest     | is_draw  |   0.876289 |  0.583333  |
| Gradient Boosting | is_draw  |   0.837113 |  0.431655  |
| XGBoost           | is_draw  |   0.886598 |  0.736842  |
| MLP               | is_draw  |   0.907216 |  0.759358  |


---
## Lien vers la Vidéo de Présentation
https://www.youtube.com/watch?v=jfyPEV3DTrI
---
## Analyse des modèles

### Q1 — Analyse des coefficients

La case centrale est la plus importante.  
Par exemple, son coefficient est d’environ **592**, ce qui est le plus élevé.

Les coins ont des coefficients proches, autour de **569**.  
Les côtés sont plus faibles, autour de **521**.

Quand X joue au centre, cela augmente fortement la probabilité de victoire.  
Quand O occupe ces cases, cela diminue les chances de X.

La raison est simple :  
Le centre participe à **4 combinaisons gagnantes**.  
Les coins à **2 combinaisons**.  
Les côtés à **1 seule**.

Cela correspond à la stratégie humaine :  
jouer au centre en premier, puis dans les coins.

---

### Q2 — Déséquilibre des classes

Le dataset est déséquilibré.

Pour `x_wins` :  
- 1 → **75.5%**  
- 0 → **24.5%**

Pour `is_draw` :  
- 1 → **18.2%**  
- 0 → **81.8%**

Cela signifie que les matchs nuls sont rares.

Un modèle peut obtenir plus de **80% d’accuracy** en ignorant les matchs nuls, ce qui est trompeur.

Les métriques à utiliser sont :  
- **F1-score** (meilleur compromis)  
- **Recall** (important pour les classes rares)  
- **AUC** (si probabilités)

Conclusion : il faut privilégier le F1-score dans ce cas.

---

### Q3 — Comparaison des modèles

#### Résultats pour `x_wins`
- XGBoost → Accuracy ≈ **0.91**, F1 ≈ **0.94**  
- MLP → Accuracy ≈ **0.90**, F1 ≈ **0.93**  
- Random Forest → Accuracy ≈ **0.88**

#### Résultats pour `is_draw`
- MLP → Accuracy ≈ **0.90**, F1 ≈ **0.75**  
- XGBoost → F1 ≈ **0.72**  
- Random Forest → F1 ≈ **0.70**

Le problème `is_draw` est plus difficile.

La raison :  
- Classe rare (**18%**)  
- Nécessite une vision globale du plateau  
- Pas de pattern simple

#### Exemple d’erreur (`x_wins`)
Matrice de confusion :
    [[ 15 109 ]
    [ 10 351 ]]


Le modèle fait **109 faux positifs**, donc il prédit trop souvent une victoire de X.

Conclusion : plus une classe est rare, plus elle est difficile à prédire.

---

### Q4 — Mode hybride

Le mode IA-ML pur utilise uniquement les données.

Le mode hybride combine ML + règles (minimax).

Résultat :  
- Le mode hybride réduit les erreurs  
- Il bloque mieux les coups adverses  
- Il évite les pièges comme les forks  

Par exemple, on observe moins de mauvaises décisions en fin de partie.

Le modèle seul peut faire des erreurs dans des positions complexes.

Conclusion : le mode hybride est plus performant.

---