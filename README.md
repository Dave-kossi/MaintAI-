# MaintAI — Adaptation légère d'un petit LLM à la maintenance industrielle

Projet du module « Adaptation légère d'un petit modèle de langage » — M2 Ingénierie Mathématique et
Science des Données, Université de Haute-Alsace (novembre 2026).

## 1. Description et objectifs

Les ordres de travail de maintenance sont des textes courts saisis par les techniciens
(« replace blown air conditioner motor »). Ils s'accumulent par milliers mais sont difficiles à exploiter.
Nous adaptons un **petit modèle de langage** pour les transformer en données structurées :

```
Entrée : replace blown air conditioner motor
Sortie : {"equipements": ["air conditioner motor"], "symptomes": ["blown"], "actions": ["replace"], "proprietes": []}
```

Objectifs, conformément au sujet :

1. adapter un modèle préentraîné à cette tâche **sans réentraîner tous ses paramètres** (LoRA, QLoRA) ;
2. comparer ses performances **avant et après adaptation** sur un jeu de test indépendant ;
3. analyser les **avantages et les limites** de la méthode (qualité, coûts, erreurs).

## 2. Données : provenance et préparation

- **Source :** corpus **MaintIE**, sous-ensemble expert (`gold_release.json`) — 1 076 ordres de travail réels,
  annotés en double par des experts. Licence MIT. <https://github.com/nlp-tlp/maintie>
  Référence : Bikaun et al. (2024), *MaintIE*, LREC-COLING 2024. <https://aclanthology.org/2024.lrec-main.954>
- **Préparation** (`convert_maintie.py`) :
  1. projection des classes MaintIE sur 4 champs : PhysicalObject → `equipements`,
     State + Process → `symptomes`, Activity → `actions`, Property → `proprietes` ;
  2. entités imbriquées d'un même champ : seule la plus longue est gardée ;
  3. suppression d'un texte en double (1 075 textes) ;
  4. découpage aléatoire 80 / 10 / 10 avec la graine 42.
- **Résultat** (déjà fourni dans `data/`) :

| Fichier | Textes | Éléments à extraire |
|---|---|---|
| `data/train.jsonl` | 860 | 2 408 |
| `data/val.jsonl` | 107 | 319 |
| `data/test.jsonl` | 108 | 300 (169 équipements, 54 symptômes, 69 actions, 8 propriétés) |

Pour régénérer les données à partir du corpus brut :

```bash
git clone https://github.com/nlp-tlp/maintie.git
python convert_maintie.py --input maintie/data/gold_release.json --outdir data/ --seed 42
```

## 3. Bibliothèques et dépendances

Python ≥ 3.10 et un GPU NVIDIA (testé sur T4 16 Go). Dépendances dans `requirements.txt` :
`torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes` (QLoRA), `pandas`, `matplotlib`.
Modèles téléchargés automatiquement depuis Hugging Face : `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`.

## 4. Installation et exécution

### Sur Kaggle (méthode utilisée, recommandée)

1. *Datasets → New Dataset* : importer `maintai_kaggle.zip` (données + scripts), nommé `maintai`.
2. *Code → New Notebook → File → Import Notebook* : `maintai_kaggle.ipynb`.
3. Réglages : Accelerator **GPU T4 x2**, Internet **On**, *Add Input* → `maintai`.
4. Test rapide : `RAPIDE = True`, *Run All* (≈ 10 min). Expériences complètes : `RAPIDE = False`,
   *Save Version → Save & Run All* (≈ 1 h 30). Sorties dans l'onglet *Output* (`maintai_sorties.zip`).

Le notebook installe `peft` et `bitsandbytes` et désinstalle `torchao` (version préinstallée sur Kaggle,
incompatible avec PEFT).

### En local (GPU NVIDIA)

```bash
pip install -r requirements.txt
pip uninstall -y torchao            # seulement si une version < 0.16 est installée

M=Qwen/Qwen2.5-0.5B-Instruct
python infer.py --model $M --shots 0 --out results/base05_zs.jsonl          # baseline zero-shot
python infer.py --model $M --shots 5 --out results/base05_fs.jsonl          # baseline 5-shot
python train_lora.py --model $M --out runs/lora05                           # adaptation LoRA
python infer.py --model $M --adapter runs/lora05 --out results/lora05.jsonl # modèle adapté
# QLoRA : train_lora.py ... --qlora, puis infer.py ... --load-4bit

for c in base05_zs base05_fs lora05; do
  python evaluate.py --gold data/test.jsonl --pred results/$c.jsonl --out results/${c}_metrics.json
done
python summarize.py          # results/tableau.csv
python compare.py            # results/comparaisons.csv (tests de significativité)
python error_analysis.py     # results/erreurs_*.csv
python make_figures.py       # figures/*.png
```

### Interface de démonstration (Gradio)

`app.py` ouvre une interface web à trois onglets :
- **Analyser un texte :** JSON extrait, texte surligné par champ, temps de réponse, alerte si un élément est absent du texte ;
- **Comparer :** même texte, modèle de base (5-shot) contre modèle adapté par LoRA ;
- **Lot de textes :** un texte par ligne, résultats en tableau, export CSV.

Sur Kaggle : importer `maintai_interface.ipynb`, mêmes réglages (GPU, Internet, dataset `maintai`), *Run All*,
puis ouvrir le lien public `*.gradio.live`. L'adaptateur `runs/lora05` est réutilisé s'il est fourni en entrée,
sinon il est réentraîné (≈ 3 min). En local :

```bash
python app.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/lora05          # http://127.0.0.1:7860
```

## 5. Méthode

| Étape | Script | Contenu |
|---|---|---|
| Données | `convert_maintie.py` | conversion de MaintIE, découpage figé |
| Adaptation | `train_lora.py` | LoRA r = 16, α = 32, dropout 0,05 sur q, k, v, o, gate, up, down ; perte sur la réponse JSON seulement ; AdamW 2e-4 (cosinus, 5 % de chauffe), batch 8, 3 epochs, fp16 ; meilleur epoch choisi sur la validation. QLoRA : base en NF4 + double quantification |
| Génération | `infer.py` | même consigne système pour toutes les configurations ; décodage glouton, 160 jetons max |
| Évaluation | `evaluate.py` | JSON valide, P / R / F1 micro par champ et global (correspondance exacte après minuscules), inventions, exact match, IC 95 % par bootstrap (1 000 tirages) |
| Comparaison | `compare.py` | bootstrap apparié (2 000 tirages) : écart de F1, IC 95 %, p-valeur |
| Analyse | `error_analysis.py`, `make_figures.py` | classement automatique des erreurs, figures |

Sept configurations : Qwen2.5-0.5B et Qwen2.5-1.5B en **zero-shot**, en **few-shot** (5 exemples fixes)
et avec **LoRA** ; Qwen2.5-1.5B avec **QLoRA**. Graine 42 partout ; un entraînement par configuration.

## 6. Résultats et interprétation

Test : 108 textes, 300 éléments. GPU T4.

| Configuration | F1 global [IC 95 %] | JSON valide | Inventions | Exact match | Temps d'entraînement | Mémoire inférence |
|---|---|---|---|---|---|---|
| 0.5B zero-shot | 0,01 [0,00 ; 0,02] | 2 % | 75 % | 0 % | — | 1,0 Go |
| 0.5B few-shot | 0,53 [0,48 ; 0,59] | 99 % | 11 % | 12 % | — | 1,0 Go |
| **0.5B LoRA** | **0,90 [0,85 ; 0,94]** | 100 % | 0 % | 79 % | 3,2 min | 1,0 Go |
| 1.5B zero-shot | 0,27 [0,19 ; 0,34] | 34 % | 25 % | 6 % | — | 3,1 Go |
| 1.5B few-shot | 0,55 [0,50 ; 0,61] | 100 % | 13 % | 22 % | — | 3,1 Go |
| **1.5B LoRA** | **0,90 [0,86 ; 0,94]** | 100 % | 0 % | 77 % | 7,5 min | 3,2 Go |
| **1.5B QLoRA** | **0,91 [0,87 ; 0,94]** | 100 % | 0 % | 77 % | 9,8 min | 1,3 Go |

**Interprétation :**

1. **LoRA apporte +0,35 de F1** par rapport au few-shot, pour les deux tailles (p < 0,001), en n'entraînant
   que 1,75 % (0.5B) ou 1,18 % (1.5B) des paramètres. Format toujours respecté, plus aucune invention.
2. **Après adaptation, la taille ne compte plus :** 0.5B LoRA et 1.5B LoRA ne diffèrent pas
   significativement (+0,006, p = 0,73). Le 0.5B adapté bat le 1.5B non adapté de +0,34.
3. **QLoRA = même qualité que LoRA** (p = 0,77), avec 60 % de mémoire d'inférence en moins, mais +30 % de
   temps d'entraînement et +33 % de latence. L'économie de mémoire à l'entraînement est faible (−13 %).
4. **Erreurs restantes :** surtout des frontières d'expressions (« boom » + « foot » au lieu de « boom foot »).
   En comptant un recouvrement partiel comme juste, LoRA atteint 0,97 (contre 0,76 à 0,80 en few-shot).
5. **Zero-shot :** échec surtout de format (objets au lieu de chaînes) ; en lecture tolérante, il atteint
   0,08 (0.5B) et 0,52 (1.5B), toujours loin de LoRA.

**Limites :** textes courts en anglais (secteur minier), test de 108 textes (`proprietes` : 8 éléments),
un seul entraînement par configuration, réglages non optimisés, adaptateur non fusionné (latence doublée).

Détails : `results/tableau.csv`, `results/comparaisons.csv`, `results/erreurs_resume.csv`, `figures/`.

## 7. Contribution de chaque membre

| Membre | Tâches réalisées |
|---|---|
| Kossi | Coordination du projet ; évaluation (`evaluate.py`, `compare.py`), tableau de résultats et intervalles de confiance ; rédaction de l'introduction, des résultats et de la conclusion ; assemblage du rapport |
| Hans | Expériences sur Qwen2.5-0.5B (baselines zero-shot et few-shot, LoRA) ; rédaction de la méthode |
| Marieme | Expériences sur Qwen2.5-1.5B (baselines, LoRA, QLoRA) ; mesures et analyse des coûts (mémoire, temps, latence) |
| Rayhan | Préparation des données (MaintIE) ; état de l'art ; analyse qualitative des erreurs et limites |

## 8. Organisation du dépôt

```
data/                 train.jsonl, val.jsonl, test.jsonl
convert_maintie.py    préparation des données
train_lora.py         adaptation LoRA / QLoRA
infer.py              génération des réponses
evaluate.py           métriques et intervalles de confiance
compare.py            tests de significativité
error_analysis.py     classement des erreurs
summarize.py          tableau comparatif
make_figures.py       figures
maintai_kaggle.ipynb  notebook qui enchaîne tout (Kaggle)
app.py                interface Gradio de démonstration
maintai_interface.ipynb  notebook qui lance l'interface (Kaggle)
requirements.txt      dépendances
LICENSE_MaintIE.md    licence des données
```

## Références

- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314
- Bikaun et al. (2024). *MaintIE.* LREC-COLING 2024
- Qwen Team (2024). *Qwen2.5 Technical Report.* arXiv:2412.15115
- Hugging Face PEFT — <https://github.com/huggingface/peft>
