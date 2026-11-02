import pandas as pd


# Challenge 2 — Pandas : manipuler et résumer un dataset
# OBJECTIF : Inspecter, filtrer, enrichir et agréger un DataFrame.
# 6.Charger un CSV et inspecter les lignes, dimensions, colonnes et types.
# 7.Utiliser info() et describe().
# 8.Filtrer des lignes avec au moins deux conditions.
# 9.Créer une colonne dérivée.
# 10.Trier et calculer des fréquences avec value_counts().
# 11.Réaliser un groupby() ; si deux fichiers sont fournis, utiliser merge().

df = pd.read_csv('data.csv')
print(df.head())
print(df.shape)
print(df.columns)
print(df.dtypes)


print(df(df.info()))
print(df(df.describe()))


df[(df['price'] > 100) & (df['qte'] < 250)]
df['total'] = df['price'] * df['qte']

df.sort_values('total', ascending=False)
df['category'].value_counts()
df['category'].value_counts(normalize=True)

df.groupby('category')['price'].agg(['mean', 'count', 'sum'])

