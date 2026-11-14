# =============================================================================
# ÉTAPE 2 — NETTOYAGE DES DONNÉES
# Projet : Prédiction de Valeur Immobilière
# Dataset : Ames Housing (Kaggle)
# =============================================================================
#
# OBJECTIF DE CETTE ÉTAPE :
# On fait 3 choses dans l'ordre :
#   1. Supprimer les outliers        → méthode IQR
#   2. Remplir les valeurs manquantes → médiane pour les chiffres
#   3. Appliquer log(1+x) sur SalePrice → réduire l'asymétrie
#
# FICHIER D'ENTRÉE  : ames_housing.csv     (créé par edafinal1.py)
# NOTE : sur le vrai dataset Kaggle, ces 12 colonnes n'ont aucune valeur
#        manquante → l'imputation ne remplacera rien (c'est normal).
# FICHIER DE SORTIE : ames_clean.csv       (utilisé par l'Étape 3)
#
# =============================================================================


# -----------------------------------------------------------------------------
# PARTIE 0 — IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

print(" Bibliothèques importées")


# -----------------------------------------------------------------------------
# PARTIE 1 — CHARGEMENT DU DATASET (créé à l'Étape 1)
# -----------------------------------------------------------------------------
#
# On repart exactement là où on s'était arrêté à l'Étape 1.
# Le fichier ames_housing.csv contient nos 12 colonnes déjà sélectionnées.

df = pd.read_csv('ames_housing.csv')

print(f" Dataset chargé : {df.shape[0]} maisons × {df.shape[1]} colonnes")
print(f"   Prix moyen AVANT nettoyage : ${df['SalePrice'].mean():,.0f}")


# -----------------------------------------------------------------------------
# PARTIE 2 — SUPPRESSION DES OUTLIERS (méthode IQR)
# -----------------------------------------------------------------------------
#
# Un outlier = une valeur aberrante très éloignée des autres.
# Par exemple une maison à 5 000 000$ au milieu de maisons à 200K$.
# Si on la garde, elle va "tirer" notre modèle vers elle et le biaiser.
#
# LA MÉTHODE IQR EN 4 ÉTAPES :
#
#   Q1  = le prix du 25ème percentile
#         → 25% des maisons coûtent MOINS que Q1
#
#   Q3  = le prix du 75ème percentile
#         → 75% des maisons coûtent MOINS que Q3
#
#   IQR = Q3 - Q1
#         → c'est la "zone centrale" où se trouvent 50% des maisons
#
#   Borne basse = Q1 - 1.5 × IQR → en dessous de ça = outlier
#   Borne haute = Q3 + 1.5 × IQR → au dessus de ça  = outlier
#
# Le coefficient 1.5 est une convention inventée par le statisticien John Tukey.
# C'est le standard utilisé dans tous les projets ML.

print("\n" + "="*60)
print("PARTIE 2 — SUPPRESSION DES OUTLIERS")
print("="*60)

# Calcul des quartiles et de l'IQR
Q1  = df['SalePrice'].quantile(0.25)  # 25ème percentile
Q3  = df['SalePrice'].quantile(0.75)  # 75ème percentile
IQR = Q3 - Q1                         # Écart Interquartile

# Calcul des bornes
borne_basse = Q1 - 1.5 * IQR
borne_haute = Q3 + 1.5 * IQR

print(f"\n   Q1  (25ème percentile) : ${Q1:,.0f}")
print(f"   Q3  (75ème percentile) : ${Q3:,.0f}")
print(f"   IQR (Q3 - Q1)          : ${IQR:,.0f}")
print(f"\n   Borne basse : ${borne_basse:,.0f}  (Q1 - 1.5 × IQR)")
print(f"   Borne haute : ${borne_haute:,.0f}  (Q3 + 1.5 × IQR)")

# Nombre de maisons AVANT la suppression
n_avant = len(df)

# On garde seulement les maisons ENTRE les deux bornes
# & = ET logique → la maison doit respecter les DEUX conditions
df = df[
    (df['SalePrice'] >= borne_basse) &
    (df['SalePrice'] <= borne_haute)
].copy()

# Nombre de maisons APRÈS la suppression
n_apres   = len(df)
n_enleves = n_avant - n_apres

print(f"\n   Maisons avant : {n_avant}")
print(f"   Maisons après : {n_apres}")
print(f"   Outliers supprimés : {n_enleves} ({n_enleves/n_avant*100:.1f}%)")
print(f"\n Outliers supprimés !")


# -----------------------------------------------------------------------------
# PARTIE 3 — REMPLACEMENT DES VALEURS MANQUANTES
# -----------------------------------------------------------------------------
#
# Après l'Étape 1, on sait que certaines colonnes ont des cases vides (NaN).
# On ne peut pas laisser des NaN — les modèles ML ne savent pas les gérer.
#
# RÈGLE SIMPLE :
#   → Pour les colonnes NUMÉRIQUES  : remplacer par la MÉDIANE
#   → Pour les colonnes TEXTE       : remplacer par le MODE (valeur la plus fréquente)
#
# Pourquoi la MÉDIANE et pas la MOYENNE ?
#   Exemple : [100, 110, 105, 108, 5000]
#   Moyenne  = 1084  ← tirée vers le haut par 5000
#   Médiane  = 108   ← valeur du milieu, pas influencée par 5000
#   → La médiane est plus représentative

print("\n" + "="*60)
print("PARTIE 3 — REMPLACEMENT DES VALEURS MANQUANTES")
print("="*60)

print(f"\n   Valeurs manquantes AVANT : {df.isnull().sum().sum()}")

# Identification des types de colonnes
cols_numeriques    = df.select_dtypes(include=[np.number]).columns.tolist()
cols_numeriques    = [c for c in cols_numeriques if c != 'SalePrice']
cols_texte         = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Remplacement pour les colonnes numériques → médiane
for col in cols_numeriques:
    nb_manquants = df[col].isnull().sum()
    if nb_manquants > 0:
        mediane = df[col].median()
        df[col] = df[col].fillna(mediane)
        print(f"   '{col}' : {nb_manquants} valeurs → remplacées par la médiane ({mediane:.0f})")

# Remplacement pour les colonnes texte → mode
for col in cols_texte:
    nb_manquants = df[col].isnull().sum()
    if nb_manquants > 0:
        mode = df[col].mode()[0]  # mode()[0] → la valeur la plus fréquente
        df[col] = df[col].fillna(mode)
        print(f"   '{col}' : {nb_manquants} valeurs → remplacées par le mode ('{mode}')")

print(f"\n   Valeurs manquantes APRÈS : {df.isnull().sum().sum()}")
print(" Valeurs manquantes traitées !")


# -----------------------------------------------------------------------------
# PARTIE 4 — TRANSFORMATION LOGARITHMIQUE DE SALEPRICE
# -----------------------------------------------------------------------------
#
# Le problème : SalePrice a une distribution asymétrique (queue à droite).
# Quelques maisons très chères tirent la distribution vers la droite.
# Les modèles ML fonctionnent mieux avec une distribution symétrique (en cloche).
#
# La solution : appliquer log(1+x) sur SalePrice
#
# Pourquoi log(1+x) et pas juste log(x) ?
#   → log(0) = -infini → erreur mathématique si le prix était 0
#   → log(1+x) évite ce problème (1+0 = 1, log(1) = 0)
#   → Dans notre cas le prix n'est jamais 0, mais c'est une bonne habitude
#
# La skewness mesure l'asymétrie :
#   → skewness = 0    → distribution parfaitement symétrique
#   → skewness > 0    → queue à droite (notre cas)
#   → skewness < 0    → queue à gauche

print("\n" + "="*60)
print("PARTIE 4 — TRANSFORMATION LOG(1+X)")
print("="*60)

skewness_avant = df['SalePrice'].skew()
print(f"\n   Skewness AVANT log : {skewness_avant:.4f}  (asymétrie à droite)")

# Application de la transformation
# np.log1p(x) = log(1+x)
# On crée une NOUVELLE colonne 'SalePrice_log'
# On garde l'originale 'SalePrice' pour pouvoir reconvertir à la fin
df['SalePrice_log'] = np.log1p(df['SalePrice'])

skewness_apres = df['SalePrice_log'].skew()
print(f"   Skewness APRÈS log : {skewness_apres:.4f}  (beaucoup plus symétrique )")
print(f"\n   Amélioration : {skewness_avant:.2f} → {skewness_apres:.2f}")
print(" Transformation log appliquée !")


# -----------------------------------------------------------------------------
# PARTIE 5 — VISUALISATIONS
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("PARTIE 5 — CRÉATION DES GRAPHIQUES")
print("="*60)

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#F8F9FA')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

BLEU   = '#2C7BB6'
VERT   = '#1A9850'
ORANGE = '#F46D43'
ROUGE  = '#D73027'

# ── Graphique 1 : Boxplot AVANT/APRÈS suppression outliers ────────────────────
#
# Un boxplot (boîte à moustaches) montre :
# - La boîte = les 50% du milieu (entre Q1 et Q3)
# - La ligne dans la boîte = la médiane
# - Les moustaches = les bornes IQR
# - Les points au-delà = les outliers

ax1 = fig.add_subplot(gs[0, :2])

# On recharge le dataset original pour le "avant"
df_original = pd.read_csv('ames_housing.csv')

data_boxplot = [df_original['SalePrice'].dropna(), df['SalePrice']]

bp = ax1.boxplot(data_boxplot, patch_artist=True, orientation='horizontal',
                 tick_labels=['Avant\nnettoyage', 'Après\nnettoyage'])

# Couleurs des boîtes
bp['boxes'][0].set(facecolor=ORANGE, alpha=0.7)
bp['boxes'][1].set(facecolor=VERT,   alpha=0.7)
for mediane in bp['medians']:
    mediane.set(color='black', linewidth=2)

ax1.set_title(f'Suppression des Outliers — Méthode IQR\n'
              f'{n_enleves} maisons supprimées ({n_enleves/n_avant*100:.1f}%)',
              fontsize=12, fontweight='bold')
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
ax1.set_xlabel('SalePrice ($)')

# ── Graphique 2 : Valeurs manquantes AVANT/APRÈS ──────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])

cols_verif   = ['TotalBsmtSF', 'GarageCars', 'LotArea']
avant_vals   = [df_original[c].isnull().sum() for c in cols_verif]
apres_vals   = [0, 0, 0]  # après remplacement → 0 manquant partout

x = np.arange(len(cols_verif))
w = 0.35

barres_avant = ax2.bar(x - w/2, avant_vals, w, color=ORANGE, alpha=0.85,
                        label='Avant', edgecolor='white')
barres_apres = ax2.bar(x + w/2, apres_vals, w, color=VERT,   alpha=0.85,
                        label='Après', edgecolor='white')

# Afficher les chiffres sur chaque barre
for barre, val in zip(barres_avant, avant_vals):
    if val > 0:
        ax2.text(barre.get_x() + barre.get_width()/2,
                 barre.get_height() + 0.5, str(val),
                 ha='center', fontsize=10, fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(cols_verif, rotation=15, fontsize=9)
ax2.set_title('Valeurs Manquantes\nAvant / Après remplacement', fontsize=12, fontweight='bold')
ax2.set_ylabel('Nombre de valeurs manquantes')
ax2.legend()
if sum(avant_vals) == 0:
    ax2.text(0.5, 0.5, 'Aucune valeur manquante\ndans nos 12 colonnes',
             ha='center', va='center', transform=ax2.transAxes,
             fontsize=11, color=VERT, fontweight='bold')

# ── Graphique 3 : Distribution SalePrice AVANT log ───────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(df['SalePrice'], bins=45, color=ORANGE, edgecolor='white', alpha=0.85)
skew_brut = df['SalePrice'].skew()
ax3.set_title(f'Prix Brut\nSkewness = {skew_brut:.2f} ', fontsize=12, fontweight='bold')
ax3.set_xlabel('SalePrice ($)')
ax3.set_ylabel('Fréquence')
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
ax3.text(0.60, 0.88, 'Queue à droite\n→ asymétrie',
         transform=ax3.transAxes, fontsize=9, color=ROUGE,
         bbox=dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.8))

# ── Graphique 4 : Distribution SalePrice APRÈS log ───────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(df['SalePrice_log'], bins=45, color=VERT, edgecolor='white', alpha=0.85)
skew_log = df['SalePrice_log'].skew()
ax4.set_title(f'Prix après log(1+x)\nSkewness = {skew_log:.2f} ', fontsize=12, fontweight='bold')
ax4.set_xlabel('log(1 + SalePrice)')
ax4.set_ylabel('Fréquence')
ax4.text(0.05, 0.88, 'Distribution\nsymétrique ',
         transform=ax4.transAxes, fontsize=9, color=VERT,
         bbox=dict(boxstyle='round', facecolor='#E5FFE5', alpha=0.8))

# ── Graphique 5 : Explication visuelle de log(1+x) ───────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
x_vals = np.linspace(0, 700000, 500)

ax5.plot(x_vals / 1000, x_vals / 1000,
         color=ORANGE, linewidth=2, label='Prix brut (y = x)')
ax5.plot(x_vals / 1000, np.log1p(x_vals),
         color=VERT, linewidth=2.5, label='log(1+x)')

ax5.set_title('Pourquoi log(1+x) ?\nIl compresse les grandes valeurs',
              fontsize=11, fontweight='bold')
ax5.set_xlabel('Prix (en milliers $)')
ax5.legend(fontsize=9)

ax5.annotate('Les prix élevés\nsont "rapprochés"',
             xy=(600, np.log1p(600000)),
             xytext=(350, 10),
             fontsize=9, color=VERT,
             arrowprops=dict(arrowstyle='->', color=VERT))

fig.suptitle(' Étape 2 — Nettoyage & Transformation des Données',
             fontsize=15, fontweight='bold', y=1.01)

plt.savefig('etape2_nettoyage.png', dpi=130, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()

print(" Graphique sauvegardé : etape2_nettoyage.png")


# -----------------------------------------------------------------------------
# PARTIE 6 — SAUVEGARDE DU DATASET PROPRE
# -----------------------------------------------------------------------------
#
# On sauvegarde le dataset nettoyé dans un nouveau fichier.
# Ce fichier sera utilisé par l'Étape 3 (encodage).

df.to_csv('ames_clean.csv', index=False)
print(" Dataset propre sauvegardé : ames_clean.csv")


# -----------------------------------------------------------------------------
# RÉSUMÉ FINAL
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print(" ÉTAPE 2 TERMINÉE — RÉSUMÉ")
print("="*60)
print(f"  Maisons avant nettoyage  : {n_avant}")
print(f"  Outliers supprimés       : {n_enleves}")
print(f"  Maisons après nettoyage  : {n_apres}")
print(f"  Valeurs manquantes       : 0 (toutes traitées)")
print(f"  Skewness SalePrice       : {skewness_avant:.2f} → {skewness_apres:.2f}")
print()
print("  Fichiers créés :")
print("    → ames_clean.csv        (dataset propre pour l'Étape 3)")
print("    → etape2_nettoyage.png  (graphiques de nettoyage)")
print()
print("  → Prochaine étape : encodage des variables texte (Étape 3)")
print("="*60)