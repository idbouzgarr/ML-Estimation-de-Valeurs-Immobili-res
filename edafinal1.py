# =============================================================================
# ÉTAPE 1 — EXPLORATION DES DONNÉES (EDA)
# Projet : Prédiction de Valeur Immobilière
# Dataset : Ames Housing RÉEL (Kaggle)
# =============================================================================
# -----------------------------------------------------------------------------
# PARTIE 0 — IMPORTATION DES BIBLIOTHÈQUES
# -----------------------------------------------------------------------------
#

import numpy as np               # calculs mathématiques (moyennes, tableaux...)
import pandas as pd              # manipulation de tableaux (comme Excel en Python)
import matplotlib.pyplot as plt  # créer des graphiques
import matplotlib.gridspec as gridspec  # organiser plusieurs graphiques
import warnings
warnings.filterwarnings('ignore')  # cache les avertissements non critiques

print("✅ Bibliothèques importées avec succès")


# -----------------------------------------------------------------------------
# PARTIE 1 — CHARGEMENT DU VRAI DATASET KAGGLE
# -----------------------------------------------------------------------------
#
# pd.read_csv() lit un fichier CSV et crée un tableau pandas (DataFrame)
#

df_complet = pd.read_csv(r"C:\Users\idris\Downloads\projetML\house-prices-advanced-regression-techniques\train.csv")

print(f"✅ Dataset chargé !")
print(f"   {df_complet.shape[0]} maisons × {df_complet.shape[1]} colonnes au total")
print(f"   (Le vrai dataset Kaggle a 81 colonnes — on va en garder 12)")


COLONNES_CHOISIES = [
    'GrLivArea',     # surface habitable en pieds carrés (1 ft² = 0.09 m²)
    'OverallQual',   # qualité générale de la construction (note de 1 à 10)
    'YearBuilt',     # année de construction
    'TotalBsmtSF',   # surface du sous-sol en pieds carrés
    'GarageCars',    # nombre de voitures dans le garage
    'FullBath',      # nombre de salles de bain complètes
    'BedroomAbvGr',  # nombre de chambres (au-dessus du sous-sol)
    'LotArea',       # surface du terrain en pieds carrés
    'Neighborhood',  # nom du quartier
    'HouseStyle',    # style de la maison (1 étage, 2 étages...)
    'ExterQual',     # qualité extérieure (Fa=Faible / TA=Moyen / Gd=Bien / Ex=Excellent)
    'SalePrice'      # ← VARIABLE CIBLE : prix de vente (ce qu'on veut prédire)
]

# On garde seulement ces 12 colonnes
# .copy() → crée une copie indépendante (bonne pratique pour éviter les erreurs)
df = df_complet[COLONNES_CHOISIES].copy()

print(f"\n✅ Sélection : {df.shape[1]} colonnes gardées sur {df_complet.shape[1]}")

# Sauvegarde dans le même dossier que ce fichier .py
# → Ce fichier sera réutilisé dans les étapes suivantes
df.to_csv('ames_housing.csv', index=False)
# index=False → ne pas sauvegarder les numéros de lignes (0, 1, 2...)
print("✅ Copie sauvegardée : ames_housing.csv")


# -----------------------------------------------------------------------------
# PARTIE 2 — EXPLORATION : QU'EST-CE QU'ON A ?
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("PARTIE 2 — EXPLORATION DES DONNÉES")
print("="*60)

# df.shape → retourne (nb_lignes, nb_colonnes)
print(f"\n📐 Taille : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"   → {df.shape[0]} maisons, {df.shape[1]} caractéristiques par maison")

# df.head(5) → affiche les 5 premières lignes
print("\n📋 Les 5 premières lignes :")
print(df.head(5).to_string())

# df.dtypes → type de chaque colonne
# int64   = entier (ex: 3 chambres)
# float64 = décimal (ex: 856.0 ft²)
# object  = texte  (ex: "NAmes")
print("\n🔍 Type de chaque colonne :")
print(df.dtypes)

# df.describe() → statistiques pour chaque colonne numérique :
# count = nb de valeurs non-manquantes
# mean  = moyenne
# std   = écart-type (dispersion des valeurs)
# min   = valeur minimale
# 50%   = médiane (valeur du milieu quand on trie)
# max   = valeur maximale
print("\n📊 Statistiques descriptives :")
print(df.describe().round(0).to_string())


# -----------------------------------------------------------------------------
# PARTIE 3 — VALEURS MANQUANTES
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("PARTIE 3 — VALEURS MANQUANTES")
print("="*60)

# df.isnull()  → tableau True/False (True = case vide dans le CSV)
# .sum()       → compte les True par colonne (True=1, False=0)
missing = df.isnull().sum()
missing = missing[missing > 0]  # garde seulement les colonnes avec des manquants

if len(missing) == 0:
    print("\n✅ Aucune valeur manquante dans les colonnes sélectionnées !")
else:
    missing_pct = (missing / len(df) * 100).round(2)
    print("\n📛 Colonnes avec valeurs manquantes :")
    for col in missing.index:
        print(f"   {col:15s} → {missing[col]:4d} manquants ({missing_pct[col]:.1f}%)")
    print("\n💡 On les traitera à l'Étape 2 (remplacement par la médiane)")

print(f"\n   Total valeurs manquantes : {df.isnull().sum().sum()}")


# -----------------------------------------------------------------------------
# PARTIE 4 — CORRÉLATION AVEC LE PRIX
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("PARTIE 4 — CORRÉLATION AVEC LE PRIX (SalePrice)")
print("="*60)

# On sélectionne seulement les colonnes numériques
# (les colonnes texte comme Neighborhood ne peuvent pas être corrélées)
cols_numeriques = df.select_dtypes(include=[np.number]).columns

# .corr()      → coefficient de Pearson entre toutes les colonnes
# ['SalePrice'] → on garde seulement la ligne SalePrice
# .abs()       → valeur absolue (force de la relation, pas le sens)
# .sort_values → du plus fort au plus faible
corr_avec_prix = df[cols_numeriques].corr()['SalePrice'].abs().sort_values(ascending=False)

print("\n🔗 Corrélation de chaque variable avec SalePrice :")
print("   r proche de 1 → forte influence  |  r proche de 0 → faible influence\n")

for col, r in corr_avec_prix.items():
    if col == 'SalePrice':
        continue  # on ignore SalePrice avec lui-même (toujours r=1)
    barre = '█' * int(r * 20)  # barre visuelle proportionnelle à r
    print(f"   {col:15s}  r={r:.2f}  {barre}")

top2 = corr_avec_prix.drop('SalePrice').head(2)
print(f"\n💡 Meilleur prédicteur : {top2.index[0]} (r={top2.iloc[0]:.2f})")
print(f"   2ème prédicteur    : {top2.index[1]} (r={top2.iloc[1]:.2f})")


# -----------------------------------------------------------------------------
# PARTIE 5 — VISUALISATIONS
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("PARTIE 5 — CRÉATION DES GRAPHIQUES")
print("="*60)

# Grande figure avec 9 sous-graphiques en 3 lignes × 3 colonnes
fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor('#F8F9FA')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

BLEU   = '#2C7BB6'
VERT   = '#1A9850'
ORANGE = '#F46D43'

# ── Graphique 1 : Distribution de SalePrice ───────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(df['SalePrice'], bins=50, color=BLEU, edgecolor='white', alpha=0.85)
ax1.set_title('Distribution du Prix de Vente (SalePrice)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Prix ($)')
ax1.set_ylabel('Nombre de maisons')
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))

# Ligne rouge = moyenne des prix
moyenne_prix = df['SalePrice'].mean()
ax1.axvline(moyenne_prix, color='red', linestyle='--', linewidth=2,
            label=f'Moyenne : ${moyenne_prix:,.0f}')
ax1.legend()

# ── Graphique 2 : Valeurs manquantes ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
missing_plot = df.isnull().sum()
missing_plot = missing_plot[missing_plot > 0].sort_values(ascending=True)

if len(missing_plot) > 0:
    ax2.barh(missing_plot.index, missing_plot.values, color=ORANGE, alpha=0.85, edgecolor='white')
    for i, v in enumerate(missing_plot.values):
        pct = v / len(df) * 100
        ax2.text(v + 0.5, i, f'{v} ({pct:.1f}%)', va='center', fontsize=9)
else:
    ax2.text(0.5, 0.5, '✅ Aucune valeur\nmanquante !',
             ha='center', va='center', fontsize=13, color=VERT,
             transform=ax2.transAxes)

ax2.set_title('Valeurs Manquantes', fontsize=13, fontweight='bold')
ax2.set_xlabel('Nombre')

# ── Graphique 3 : Distribution de OverallQual ────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
qual_counts = df['OverallQual'].value_counts().sort_index()
ax3.bar(qual_counts.index, qual_counts.values, color=VERT, alpha=0.85, edgecolor='white')
ax3.set_title('Qualité Générale\n(OverallQual)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Score (1=mauvais → 10=excellent)')
ax3.set_ylabel('Nombre de maisons')

# ── Graphique 4 : Distribution de GrLivArea ──────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(df['GrLivArea'], bins=40, color=BLEU, edgecolor='white', alpha=0.85)
ax4.set_title('Surface Habitable\n(GrLivArea, en pieds²)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Surface (ft²)')
ax4.set_ylabel('Nombre de maisons')

# ── Graphique 5 : Nombre de maisons par quartier ─────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
neigh_counts = df['Neighborhood'].value_counts()
ax5.barh(neigh_counts.index, neigh_counts.values, color='#7B3294', alpha=0.8, edgecolor='white')
ax5.set_title('Répartition par Quartier\n(Neighborhood)', fontsize=12, fontweight='bold')
ax5.set_xlabel('Nombre de maisons')
ax5.invert_yaxis()

# ── Graphiques 6, 7, 8 : Nuages de points — relations avec SalePrice ─────────
#
# Chaque point = une maison
# Axe X = variable explicative  |  Axe Y = prix
# Ligne rouge pointillée = tendance générale (droite de régression)

paires = [
    ('GrLivArea',   'Surface habitable (ft²)'),
    ('OverallQual', 'Qualité générale (1-10)'),
    ('YearBuilt',   'Année de construction')
]
couleurs_nuage = [BLEU, VERT, ORANGE]

for i, (col, label) in enumerate(paires):
    ax = fig.add_subplot(gs[2, i])

    # On supprime les lignes avec des NaN avant de tracer
    data_propre = df[[col, 'SalePrice']].dropna()

    # scatter → nuage de points
    # alpha=0.25 → transparence pour voir les points superposés
    # s=8        → petite taille des points
    ax.scatter(data_propre[col], data_propre['SalePrice'],
               alpha=0.25, s=8, color=couleurs_nuage[i])

    # Droite de tendance : y = a*x + b
    # np.polyfit calcule a et b automatiquement
    z = np.polyfit(data_propre[col], data_propre['SalePrice'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(data_propre[col].min(), data_propre[col].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=1.5)

    # Corrélation r
    r = data_propre.corr().iloc[0, 1]
    ax.set_title(f'{label}\n(r = {r:.2f})', fontsize=11, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('SalePrice ($)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))

fig.suptitle(
    '📊 Étape 1 — Exploration des Données (EDA)\nDataset : Ames Housing (Kaggle) — données réelles',
    fontsize=15, fontweight='bold', y=1.01
)

# Sauvegarde dans le même dossier que ce fichier .py
plt.savefig('etape1_eda.png', dpi=130, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()

print("✅ Graphique sauvegardé : etape1_eda.png")


# -----------------------------------------------------------------------------
# RÉSUMÉ FINAL
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("✅ ÉTAPE 1 TERMINÉE — RÉSUMÉ")
print("="*60)
print(f"  Dataset         : {df.shape[0]} maisons × {df.shape[1]} colonnes")
print(f"  Variable cible  : SalePrice")
print(f"  Prix moyen      : ${df['SalePrice'].mean():,.0f}")
print(f"  Prix min        : ${df['SalePrice'].min():,.0f}")
print(f"  Prix max        : ${df['SalePrice'].max():,.0f}")
print(f"  Valeurs manq.   : {df.isnull().sum().sum()} au total")
print(f"  Meilleur préd.  : {top2.index[0]} (r={top2.iloc[0]:.2f})")
print(f"  2ème préd.      : {top2.index[1]} (r={top2.iloc[1]:.2f})")
print()
print("  → Prochaine étape : nettoyage des données (Étape 2)")
print("="*60)