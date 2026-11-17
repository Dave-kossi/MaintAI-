# MaintAI — Adaptation légère d'un petit LLM à la maintenance industrielle

Comparaison d'un petit modèle de langage **avant et après adaptation légère (LoRA, QLoRA)** pour
extraire des informations structurées de textes de maintenance industrielle.

> Entrée : `replace blown air conditioner motor`
> Sortie : `{"equipements": ["air conditioner motor"], "symptomes": ["blown"], "actions": ["replace"], "proprietes": []}`

## Lancer le projet sur Kaggle

1. **Dataset :** *Datasets → New Dataset*, uploader `maintai_kaggle.zip`, le nommer `maintai`.
2. **Notebook :** *Code → New Notebook → File → Import Notebook*, choisir `maintai_kaggle.ipynb`.
3. **Réglages (panneau de droite) :** Accelerator **GPU T4 x2** · Internet **On** · *Add Input* → `maintai`.
4. **Test express :** dans la cellule « Réglages », `RAPIDE = True`, puis *Run All* (≈ 10 min).
   Tout doit aller jusqu'au bout sans erreur.
5. **Vraies expériences :** `RAPIDE = False`, puis **Save Version → Save & Run All**
   (≈ 1 h 30 à 2 h, tourne même navigateur fermé).
6. **Récupérer :** onglet *Output* de la version → `maintai_sorties.zip`.

Si une session interactive coupe, relancez *Run All* : chaque étape déjà terminée est sautée.

## Configurations comparées

| Nom | Modèle | Méthode |
|---|---|---|
| `base05_zs`, `base05_fs` | Qwen2.5-0.5B-Instruct | sans adaptation, zero-shot / 5-shot |
| `lora05` | Qwen2.5-0.5B-Instruct | LoRA |
| `base15_zs`, `base15_fs` | Qwen2.5-1.5B-Instruct | sans adaptation, zero-shot / 5-shot |
| `lora15` | Qwen2.5-1.5B-Instruct | LoRA |
| `qlora15` | Qwen2.5-1.5B-Instruct | QLoRA (base en 4 bits) |

Réglages communs : LoRA rang 16, alpha 32, dropout 0,05, sur les projections d'attention et MLP ;
taux d'apprentissage 2e-4 (cosinus, 5 % de chauffe), batch 8, 3 epochs, meilleur epoch choisi sur `val` ;
décodage déterministe ; graine 42. Même consigne et même jeu de test pour toutes les configurations.

## Fichiers

| Fichier | Rôle |
|---|---|
| `data/train.jsonl`, `val.jsonl`, `test.jsonl` | 860 / 107 / 108 textes, découpage figé (ne jamais modifier le test) |
| `convert_maintie.py` | conversion du corpus MaintIE brut vers ce format (déjà fait) |
| `train_lora.py` | entraînement LoRA ou QLoRA (`--qlora`) |
| `infer.py` | génération des réponses (modèle de base ou adapté), latence, mémoire |
| `evaluate.py` | JSON valide, P/R/F1 par champ et global, IC à 95 % (bootstrap), inventions |
| `summarize.py` | tableau comparatif de toutes les configurations |
| `compare.py` | l'écart entre deux configurations est-il significatif ? (bootstrap apparié) |
| `error_analysis.py` | classement automatique des erreurs (oubli, invention, mauvais champ…) |
| `make_figures.py` | les 5 figures du rapport |
| `maintai_kaggle.ipynb` | le notebook qui enchaîne tout |

## Où trouver chaque élément du rapport

| Section du rapport | Fichier à utiliser |
|---|---|
| Données | statistiques affichées par `convert_maintie.py` (860 / 107 / 108 textes) |
| Méthode | réglages ci-dessus ; `runs/*/train_stats.json` (paramètres entraînés) |
| Résultats avant / après | `results/tableau.csv`, `figures/fig1_f1_global.png`, `figures/fig2_f1_par_champ.png` |
| Significativité | `results/comparaisons.csv` (différence de F1, IC à 95 %, p-valeur) |
| Coûts LoRA vs QLoRA | `figures/fig3_couts.png`, colonnes temps / mémoire / latence de `tableau.csv` |
| Apprentissage | `figures/fig4_pertes.png` (surapprentissage si la validation remonte) |
| Analyse des erreurs | `results/erreurs_resume.csv`, `figures/fig5_erreurs.png`, exemples dans `results/erreurs_<config>.csv` |

## Comment lire les métriques

- **JSON valide :** le modèle respecte-t-il le format ? Souvent le premier gain visible de LoRA.
- **F1 global (micro) :** compromis entre éléments justes trouvés (rappel) et éléments prédits justes (précision).
  Correspondance exacte après mise en minuscules : « pump » ne compte pas pour « water pump ».
- **IC à 95 % :** avec 108 textes de test, un écart de quelques points peut être dû au hasard ;
  `compare.py` dit si l'écart est significatif.
- **Taux d'inventions :** part des éléments prédits absents du texte (hallucinations).

## Limites connues

- Textes très courts (≈ 6 mots), en anglais, surtout issus du secteur minier.
- Jeu de test de 108 textes : intervalles de confiance larges ; le champ `proprietes` n'a que 8 éléments.
- Les champs « résultat » et « cause » ne sont pas annotés dans MaintIE et ne sont pas évalués.
- Correspondance exacte : une extraction partiellement juste compte comme une erreur
  (catégorie « frontière » dans l'analyse des erreurs).

## Source des données

MaintIE, licence MIT — <https://github.com/nlp-tlp/maintie>.
Bikaun, T., French, T., Stewart, M., Liu, W., Hodkiewicz, M. (2024). *MaintIE: A Fine-Grained
Annotation Schema and Benchmark for Information Extraction from Maintenance Short Texts.*
LREC-COLING 2024. <https://aclanthology.org/2024.lrec-main.954>
