# =============================================================================
# ÉTAPE 3 — ENCODAGE DES VARIABLES TEXTE
# Projet : Prédiction de Valeur Immobilière
# Dataset : Ames Housing (Kaggle)
# =============================================================================
#
# OBJECTIF :
# Les modèles ML ne lisent QUE des chiffres.
# On a 3 colonnes texte : Neighborhood, HouseStyle, ExterQual
# → On doit les transformer en chiffres avant de pouvoir modéliser.
#
# 2 méthodes selon le cas :
#   → Label Encoding   : pour les variables ORDONNÉES (Fa < TA < Gd < Ex)
#   → One-Hot Encoding : pour les variables SANS ORDRE (quartiers)
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
# PARTIE 1 — CHARGEMENT DU DATASET PROPRE (créé à l'Étape 2)
# -----------------------------------------------------------------------------

df = pd.read_csv('ames_clean.csv')

print(f" Dataset chargé : {df.shape[0]} maisons × {df.shape[1]} colonnes")

# Affichons les colonnes texte qu'on doit encoder
cols_texte = df.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"\n Colonnes texte à encoder : {cols_texte}")
for col in cols_texte:
    print(f"   '{col}' → valeurs uniques : {df[col].unique()}")


# -----------------------------------------------------------------------------
# PARTIE 2 — LABEL ENCODING sur ExterQual
# -----------------------------------------------------------------------------
#
# ExterQual = qualité extérieure de la maison
# Valeurs possibles : 'Fa' (Faible), 'TA' (Moyen), 'Gd' (Bien), 'Ex' (Excellent)
#
# Il y a un ORDRE LOGIQUE : Fa < TA < Gd < Ex
# → On peut utiliser Label Encoding : on remplace chaque mot par un chiffre
#   qui respecte cet ordre.
#
# Fa → 1  (le moins bon)
# TA → 2
# Gd → 3
# Ex → 4  (le meilleur)
#
# Pourquoi pas One-Hot Encoding ici ?
# Parce qu'il y a un ordre logique. Le modèle doit savoir que
# Ex (4) est "mieux" que Gd (3), qui est "mieux" que TA (2)...
# Si on faisait du One-Hot, on perdrait cette information d'ordre.

print("\n" + "="*60)
print("PARTIE 2 — LABEL ENCODING sur ExterQual")
print("="*60)

# On crée un dictionnaire qui fait la correspondance mot → chiffre
# {'clé': valeur, 'clé2': valeur2, ...}
ordre_qualite = {
    'Fa': 1,   # Faible
    'TA': 2,   # Typique/Average (Moyen)
    'Gd': 3,   # Good (Bien)
    'Ex': 4    # Excellent
}

# Avant l'encodage
print(f"\n   AVANT : {df['ExterQual'].value_counts().to_dict()}")

# df['colonne'].map(dictionnaire) → remplace chaque valeur par sa correspondance
# dans le dictionnaire
df['ExterQual'] = df['ExterQual'].map(ordre_qualite)

# Vérification : une valeur absente du dictionnaire deviendrait NaN
print(f"   Valeurs non reconnues (NaN) après map : {df['ExterQual'].isnull().sum()}")

# Après l'encodage
print(f"   APRÈS : {df['ExterQual'].value_counts().sort_index().to_dict()}")
print("\n   Fa→1, TA→2, Gd→3, Ex→4 ")


# -----------------------------------------------------------------------------
# PARTIE 3 — ONE-HOT ENCODING sur Neighborhood et HouseStyle
# -----------------------------------------------------------------------------
#
# Neighborhood = quartier de la maison (NAmes, CollgCr, OldTown...)
# HouseStyle   = style de la maison (1Story, 2Story, SLvl...)
#
# Ces variables N'ONT PAS D'ORDRE LOGIQUE.
# NAmes n'est pas "plus grand" que CollgCr — ils sont juste différents.
# Si on faisait Label Encoding (NAmes=1, CollgCr=2, OldTown=3...),
# le modèle penserait que OldTown (3) > CollgCr (2) > NAmes (1)
# ce qui est FAUX et introduirait un biais.
#
# LA SOLUTION : One-Hot Encoding
# On crée UNE COLONNE PAR CATÉGORIE.
# Chaque colonne vaut 1 si la maison appartient à cette catégorie, 0 sinon.
#
# Exemple pour Neighborhood :
#
# AVANT :              APRÈS :
# Neighborhood         Neighborhood_CollgCr  Neighborhood_Edwards  Neighborhood_NAmes ...
# NAmes           →    0                     0                     1
# CollgCr         →    1                     0                     0
# OldTown         →    0                     0                     0
# NAmes           →    0                     0                     1
#
# drop_first=True → on supprime la première colonne de chaque groupe
# Pourquoi ? Si on a 10 quartiers et qu'une maison n'est dans aucune des
# 9 colonnes → on sait qu'elle est dans le 10ème. Garder les 10 serait
# redondant (problème de multicolinéarité).

print("\n" + "="*60)
print("PARTIE 3 — ONE-HOT ENCODING sur Neighborhood et HouseStyle")
print("="*60)

n_cols_avant = df.shape[1]
print(f"\n   Colonnes AVANT One-Hot : {n_cols_avant}")
print(f"   Valeurs uniques Neighborhood : {df['Neighborhood'].nunique()}")
print(f"   Valeurs uniques HouseStyle   : {df['HouseStyle'].nunique()}")

# pd.get_dummies() → applique le One-Hot Encoding automatiquement
# columns=['col1', 'col2'] → colonnes à encoder
# drop_first=True          → supprime la première colonne de chaque groupe
# dtype=int                → les nouvelles colonnes contiennent 0 et 1 (entiers)
df = pd.get_dummies(
    df,
    columns=['Neighborhood', 'HouseStyle'],
    drop_first=True,
    dtype=int
)

n_cols_apres = df.shape[1]
print(f"   Colonnes APRÈS One-Hot  : {n_cols_apres}")
print(f"   Nouvelles colonnes créées : {n_cols_apres - n_cols_avant}")
print(f"\n   Exemple de nouvelles colonnes :")

# Afficher les nouvelles colonnes créées
nouvelles_cols = [c for c in df.columns if c.startswith('Neighborhood_') or c.startswith('HouseStyle_')]
for col in nouvelles_cols[:6]:  # on affiche seulement les 6 premières
    print(f"   → {col}")
print(f"   ... (et {len(nouvelles_cols)-6} autres)")

print(f"\n One-Hot Encoding terminé !")


# -----------------------------------------------------------------------------
# PARTIE 4 — VÉRIFICATION FINALE
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("PARTIE 4 — VÉRIFICATION FINALE")
print("="*60)

# Vérifier qu'il ne reste plus aucune colonne texte
cols_texte_restantes = df.select_dtypes(exclude=[np.number]).columns.tolist()

if len(cols_texte_restantes) == 0:
    print("\n Aucune colonne texte restante — tout est numérique !")
else:
    print(f"\n  Colonnes texte restantes : {cols_texte_restantes}")

# Vérifier les valeurs manquantes
total_nan = df.isnull().sum().sum()
print(f" Valeurs manquantes : {total_nan}")

# Afficher le résultat final
print(f"\n Dataset encodé : {df.shape[0]} maisons × {df.shape[1]} colonnes")
print(f"   (on est passé de {n_cols_avant} colonnes à {df.shape[1]} après encodage)")
print(f"\n   Les 5 premières colonnes : {df.columns[:5].tolist()}")
print(f"   Les 5 dernières colonnes  : {df.columns[-5:].tolist()}")


# -----------------------------------------------------------------------------
# PARTIE 5 — VISUALISATIONS
# -----------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#F8F9FA')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.40)

BLEU   = '#2C7BB6'
VERT   = '#1A9850'
ORANGE = '#F46D43'
VIOLET = '#7B3294'
BLANC  = '#FFFFFF'

# ── Graphique 1 : Schéma Label Encoding ───────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 1)
ax1.axis('off')
ax1.set_facecolor('#F8F9FA')
ax1.set_title('① Label Encoding — Variables ORDONNÉES (ExterQual)',
              fontsize=13, fontweight='bold', pad=12)

categories = [('Fa\n(Faible)', ORANGE), ('TA\n(Moyen)', BLEU),
              ('Gd\n(Bien)', VERT), ('Ex\n(Excellent)', VIOLET)]

# Flèche du milieu
ax1.annotate('', xy=(5.2, 0.5), xytext=(4.2, 0.5),
             arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
ax1.text(4.7, 0.72, 'ENCODAGE', ha='center', fontsize=10, fontweight='bold')

# Boîtes AVANT (mots)
for i, (label, color) in enumerate(categories):
    x = 0.3 + i * 0.95
    rect = plt.Rectangle((x, 0.22), 0.75, 0.52, facecolor=color,
                          alpha=0.25, edgecolor=color, linewidth=2)
    ax1.add_patch(rect)
    ax1.text(x + 0.375, 0.48, label, ha='center', va='center',
             fontsize=10, fontweight='bold')

# Boîtes APRÈS (chiffres)
nums = [('1', ORANGE), ('2', BLEU), ('3', VERT), ('4', VIOLET)]
for i, (num, color) in enumerate(nums):
    x = 5.5 + i * 0.95
    rect = plt.Rectangle((x, 0.22), 0.75, 0.52, facecolor=color,
                          alpha=0.25, edgecolor=color, linewidth=2)
    ax1.add_patch(rect)
    ax1.text(x + 0.375, 0.48, num, ha='center', va='center',
             fontsize=16, fontweight='bold', color=color)

ax1.text(2.1, 0.06, 'AVANT : mots (non lisibles par le modèle)',
         ha='center', fontsize=10, color='gray', style='italic')
ax1.text(7.5, 0.06, 'APRÈS : chiffres avec ordre logique ✅',
         ha='center', fontsize=10, color='gray', style='italic')

# ── Graphique 2 : Prix moyen par ExterQual ────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
qual_price = df.groupby('ExterQual')['SalePrice'].mean()
colors_qual = [ORANGE, BLEU, VERT, VIOLET]
bars = ax2.bar(qual_price.index, qual_price.values,
               color=colors_qual[:len(qual_price)], alpha=0.85, edgecolor='white')
ax2.set_title('Prix moyen par\nQualité Extérieure', fontsize=12, fontweight='bold')
ax2.set_xlabel('ExterQual encodé (1=Fa, 2=TA, 3=Gd, 4=Ex)')
ax2.set_ylabel('Prix moyen ($)')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
for bar, val in zip(bars, qual_price.values):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1000, f'${val/1000:.0f}K',
             ha='center', fontsize=9, fontweight='bold')
ax2.text(0.5, 0.02, '→ L\'ordre est respecté : 1 < 2 < 3 < 4 ',
         ha='center', transform=ax2.transAxes, fontsize=9,
         color=VERT, style='italic')

# ── Graphique 3 : Colonnes avant/après encodage ───────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
bars3 = ax3.bar(['Avant\nencodage', 'Après\nencodage'],
                [n_cols_avant, df.shape[1]],
                color=[ORANGE, VERT], alpha=0.85, edgecolor='white', width=0.5)
ax3.set_title('Nombre de colonnes\nAvant / Après encodage', fontsize=12, fontweight='bold')
ax3.set_ylabel('Nombre de colonnes')
ax3.set_ylim(0, df.shape[1] + 5)
for bar, val in zip(bars3, [n_cols_avant, df.shape[1]]):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3, str(val),
             ha='center', fontsize=14, fontweight='bold')

# ── Graphique 4 : Prix moyen par quartier (top 10) ────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
# Pour ce graphique on relit le fichier original (avant encodage)
df_orig = pd.read_csv('ames_clean.csv')
neigh_price = df_orig.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=True)
ax4.barh(neigh_price.index, neigh_price.values, color=VIOLET, alpha=0.75, edgecolor='white')
ax4.set_title('Prix moyen par Quartier\n(pourquoi One-Hot ?)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Prix moyen ($)')
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
ax4.text(0.5, -0.18,
         '→ Pas d\'ordre entre quartiers → One-Hot Encoding',
         ha='center', transform=ax4.transAxes, fontsize=9,
         color=VIOLET, style='italic')

fig.suptitle('🔢 Étape 3 — Encodage des Variables Texte → Chiffres',
             fontsize=15, fontweight='bold', y=1.01)

plt.savefig('etape3_encodage.png', dpi=130, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("\n Graphique sauvegardé : etape3_encodage.png")


# -----------------------------------------------------------------------------
# PARTIE 6 — SAUVEGARDE
# -----------------------------------------------------------------------------

df.to_csv('ames_encoded.csv', index=False)
print(" Dataset encodé sauvegardé : ames_encoded.csv")


# -----------------------------------------------------------------------------
# RÉSUMÉ FINAL
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print(" ÉTAPE 3 TERMINÉE — RÉSUMÉ")
print("="*60)
print(f"  ExterQual    → Label Encoding  (Fa=1, TA=2, Gd=3, Ex=4)")
print(f"  Neighborhood → One-Hot Encoding ({df_orig['Neighborhood'].nunique()} quartiers)")
print(f"  HouseStyle   → One-Hot Encoding ({df_orig['HouseStyle'].nunique()} styles)")
print(f"  Colonnes : {n_cols_avant} → {df.shape[1]}")
print(f"  Colonnes texte restantes : 0 ")
print(f"  Valeurs manquantes : 0 ")
print()
print("  Fichiers créés :")
print("    → ames_encoded.csv     (dataset prêt pour la modélisation)")
print("    → etape3_encodage.png  (graphiques d'encodage)")
print()
print("  → Prochaine étape : Régression Linéaire (Étape 4)")
print("="*60)