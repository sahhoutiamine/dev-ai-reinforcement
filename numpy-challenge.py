import numpy as np

# Challenge 1 — NumPy : manipuler des données numériques
# OBJECTIFComprendre la représentation des données sous forme de tableaux, comme dans les algorithmes ML/DL.
# 1.Créer un tableau 2D contenant des valeurs numériques pour plusieurs individus et caractéristiques.
# 2.Afficher shape, ndim et dtype ; modifier la forme avec reshape.
# 3.Calculer moyenne, médiane, minimum, maximum et écart-type par colonne.
# 4.Sélectionner des valeurs avec un masque booléen.
# 5.Normaliser une colonne et gérer le cas où l’écart-type est nul.



a = np.array([[12, 45, 2, 7, 34], [32, 4, 2, 6, 10]])
print(a.shape)
print(a.ndim)
print(a.dtype)
print(a.reshape(5, 2))

print(a.mean(axis=0))
print(np.median(a, axis=0))
print(a.min(axis=0))
print(a.max(axis=0))
print(a.std(axis=0))
print(a[a > 10])

def normalize_column(col) :
    return (col - col.mean()) / col.std()
col = a[:, 0].astype(float)

print(normalize_column(col))

