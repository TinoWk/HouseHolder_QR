
'''             CARTOUCHE 
 Titre : LA METHODE DE HOUSEHOLDER ASSOCIE A QR
 Auteurs : WOWUI K. Martin & WONGUE Banlé.
'''
import numpy as np

# fonction de la décomposition de Householder
def householder_reflection(A):

    m, n = len(A), len(A[0])
    Q = np.eye(m)
    R = np.array(A, dtype=float).copy()
    for k in range(min(m, n)):
        x = R[k:, k]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)

        # Éviter la division par zéro si x est un vecteur nul
        if np.linalg.norm(x) == 0: # si vrai, il n'exécute plus les lignes suivantes , il remonte itérer k 
            continue
        
        u = x + np.sign(x[0]) * e
        v = u / np.linalg.norm(u)
        
        H_k = np.eye(m)
        H_k[k:, k:] =  H_k[k:, k:]- 2.0 * np.outer(v, v)
        R = H_k @ R # produit matriciel
        Q = Q @ H_k.T # la transposée est égale à l'inverse (H)
    np.set_printoptions(precision=10, suppress=True) 
    print(f" \nLa matrice orthogonale Q est :\n {Q} \n \n La matrice triangulaire supérieure R est:\n{R}")
    return Q, R