x# 🎮 Projet Apprentissage par Renforcement : DQN vs Monte Carlo

Ce dépôt implémente deux méthodes d’apprentissage par renforcement appliquées au jeu **Three In Row**.  
L’objectif de l’agent est d’aligner ses **3 pions bleus** pour gagner la partie.

---

## 🕹️ Structure du projet

```
manette/
│── env.py            # Définition de l’environnement ThreeInRowEnv
│── run_all.py        # Menu interactif pour lancer DQN ou Monte Carlo

dqn_project/
│── env.py   
│── train_dqn.py      # Entraînement avec Deep Q-Network (DQN)
│── play.py           # Launcher Tkinter pour jouer avec le modèle DQN
│── dqn_model.pth     # Modèle sauvegardé après apprentissage

monte_carlo_project/
│── env.py   
│── mc_control.py     # Entraînement avec Monte Carlo Control
│── mc_play.py        # Launcher Tkinter pour jouer avec la politique MC
│── mc_policy.pkl     # Politique sauvegardée après apprentissage
```

---

## 📖 Règles du jeu (Three In Row)

1. Plateau **3x3**.  
2. Chaque joueur dispose de **3 pions** en meme temps 
3. **Phase de placement** :  
   - L’agent place ses 3 pions un par un.  
   - L’adversaire place aussi 3 pions, automatiquement et aléatoirement.  
4. **Phase de mouvement** :  
   - L’agent peut déplacer un pion vers une case voisine libre (y compris en diagonale).  
   - L’adversaire rouge est fixe dans la version actuelle (pions immobiles).  
5. La partie se termine si :  
   - L’agent aligne 3 pions → **Victoire**.  
   - Aucun coup valide → **Match nul / défaite**.  

---

## 📚 Partie Théorique

### 🔹 Deep Q-Learning (DQN)
- Approximation de la fonction Q(s,a) avec un réseau de neurones.  
- Utilise **Experience Replay** + **réseau cible** pour stabiliser l’apprentissage.  
- Politique **epsilon-greedy** pour équilibrer exploration/exploitation.  

**Avantages :**
- Adapté aux grands espaces d’états.  
- Capable d’apprendre des stratégies complexes.  
- Bonne généralisation.  

**Inconvénients :**
- Entraînement lent (beaucoup d’épisodes nécessaires).  
- Sensible aux hyperparamètres.  
- Moins interprétable (boîte noire).  

---

### 🔹 Monte Carlo Control (MC)
- Apprentissage basé sur les **retours d’épisodes complets**.  
- Politique epsilon-greedy également.  

**Avantages :**
- Simple à comprendre et implémenter.  
- Théoriquement converge avec assez d’épisodes.  
- Pas besoin de modèle complexe.  

**Inconvénients :**
- Beaucoup plus lent que DQN pour les grands espaces.  
- Variance élevée → instable sur peu d’épisodes.  
- Peu adapté aux environnements continus.  

---

## 🛠️ Partie Pratique

### 💻 Pré-requis
- Python 3.10+  
- Install requirements.txt : pip install -r requirements.txt
- IDE conseillé : **Visual Studio Code (VS Code)** avec terminal intégré.
  
### 🚀 Lancer le menu principal
Depuis **Visual Studio Code (VS Code)**, ouvrez un terminal intégré et exécutez :
```bash
cd manette
python run_all.py
```

Vous aurez accès au menu :

```
=== Three In Row - Menu ===
1. Entraîner DQN
2. Jouer avec DQN
3. Entraîner Monte Carlo
4. Jouer avec Monte Carlo
0. Quitter
```

### 🔹 Entraîner un agent
- **DQN** → choix 1  
- **Monte Carlo** → choix 3  

### 🔹 Jouer avec un agent
- **DQN** → choix 2  
- **Monte Carlo** → choix 4  

Une interface Tkinter s’ouvre avec :  
- Plateau 3x3  
- Pions rouges fixes (adversaire)  
- Pions bleus (contrôlés par l’agent)  
- Boutons `Jouer un coup` et `Nouvelle partie`.  

---

## ⚖️ Comparaison DQN vs Monte Carlo

|                   | DQN ✅ | Monte Carlo ⚪ |
|--------------------------|--------|----------------|
| Vitesse d’apprentissage | Rapide après réglages | Lent |
| Complexité implémentation| Plus complexe (réseaux, replay buffer) | Simple |
| Adapté aux grands espaces | Oui | Non |
| Stabilité | Bonne avec réseau cible | Variable |

---

## 📝 Conclusion
- **Monte Carlo** → idéal pour débuter, pédagogie et environnements petits.  
- **DQN** → puissant, généralisable, adapté aux environnements plus complexes.  
 

