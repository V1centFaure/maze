# Guide d'Installation et Résolution des Problèmes d'Importation

## Problème Courant : "ModuleNotFoundError: No module named 'maze'"

Si vous rencontrez cette erreur après avoir installé le package depuis GitHub, voici les causes possibles et leurs solutions.

## 🔍 Diagnostic du Problème

### 1. Vérification de l'Installation

Après installation, vérifiez que le package est bien installé :

```bash
pip list | grep maze
```

Vous devriez voir quelque chose comme :
```
maze-rl    1.0.0
```

### 2. Vérification de l'Environnement Python

Assurez-vous d'utiliser le bon environnement Python :

```bash
# Vérifiez quel Python vous utilisez
which python
python --version

# Vérifiez où pip installe les packages
pip show maze-rl
```

## ✅ Solutions

### Solution 1 : Importation Correcte

**❌ Incorrect (ne fonctionne qu'en développement) :**
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from maze import Test_maze
from agent_epsilon_greedy import Agent_epsilon_greedy
```

**✅ Correct (après installation pip) :**
```python
from maze import Test_maze, Agent_epsilon_greedy
import maze.config as config
```

### Solution 2 : Réinstallation Propre

Si le problème persiste, réinstallez le package :

```bash
# Désinstaller
pip uninstall maze-rl -y

# Réinstaller depuis GitHub
pip install git+https://github.com/V1centFaure/maze.git

# Ou installation en mode développement
git clone https://github.com/V1centFaure/maze.git
cd maze
pip install -e .
```

### Solution 3 : Vérification des Environnements Virtuels

Si vous utilisez des environnements virtuels :

```bash
# Activez votre environnement virtuel
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Puis installez
pip install git+https://github.com/V1centFaure/maze.git
```

## 📝 Exemple Complet Fonctionnel

Voici un exemple complet qui fonctionne après installation via pip :

```python
"""
Exemple d'utilisation après installation pip
"""

# Importations correctes après installation
from maze import Test_maze, Agent_epsilon_greedy
import maze.config as config
import numpy as np

def main():
    print("Test d'importation du package maze...")
    
    # Créer un environnement
    env = Test_maze()
    print(f"✅ Environnement créé : {env.width}x{env.height}")
    
    # Créer un agent
    agent = Agent_epsilon_greedy(
        height=env.height,
        width=env.width,
        epsilon=0.1
    )
    print(f"✅ Agent créé avec epsilon={agent.epsilon}")
    
    # Test rapide d'entraînement
    env.reset()
    state = env.agent_pos
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    
    if next_state is not None:
        agent.update(state, action, reward, next_state, done)
        print("✅ Mise à jour de l'agent réussie")
    
    print("✅ Tous les tests d'importation ont réussi !")

if __name__ == "__main__":
    main()
```

## 🐛 Problèmes Spécifiques et Solutions

### Problème : "ImportError: cannot import name 'config'"

**Solution :**
```python
# ❌ Incorrect
import config

# ✅ Correct
import maze.config as config
```

### Problème : "ModuleNotFoundError: No module named 'agent_epsilon_greedy'"

**Solution :**
```python
# ❌ Incorrect
from agent_epsilon_greedy import Agent_epsilon_greedy

# ✅ Correct
from maze import Agent_epsilon_greedy
```

### Problème : Point d'entrée console ne fonctionne pas

Si la commande `maze-demo` ne fonctionne pas, c'est probablement dû à un problème de configuration dans `setup.py`. Utilisez plutôt :

```python
# Créez votre propre script de démonstration
from maze import Test_maze, Agent_epsilon_greedy

env = Test_maze()
agent = Agent_epsilon_greedy(env.height, env.width)
# ... votre code ...
```

## 🔧 Développement vs Production

### Mode Développement (avec sys.path)
Utilisé uniquement lors du développement du package :
```python
import sys
import os
sys.path.insert(0, 'src')
from maze import Test_maze
```

### Mode Production (après installation pip)
Utilisé après installation du package :
```python
from maze import Test_maze, Agent_epsilon_greedy
import maze.config as config
```

## 📋 Checklist de Vérification

- [ ] Le package `maze-rl` est installé (`pip list | grep maze`)
- [ ] Vous utilisez les bonnes importations (`from maze import ...`)
- [ ] Vous êtes dans le bon environnement Python
- [ ] Vous n'utilisez pas `sys.path.insert()` après installation
- [ ] Vous importez `config` comme `maze.config`

## 🆘 Support

Si vous rencontrez toujours des problèmes :

1. Vérifiez la version de Python (>= 3.8 requis)
2. Vérifiez que toutes les dépendances sont installées
3. Essayez dans un nouvel environnement virtuel
4. Ouvrez une issue sur GitHub avec :
   - Votre version de Python
   - La sortie de `pip list`
   - Le message d'erreur complet
   - Votre système d'exploitation

---

**Note :** Ce guide résout les problèmes d'importation les plus courants. Pour des problèmes spécifiques, consultez la documentation complète dans le README.md.
