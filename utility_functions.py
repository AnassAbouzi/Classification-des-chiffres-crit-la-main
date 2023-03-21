import numpy as np

def distance_euclidienne(arr1, arr2) :
    """
    Calculer la distance euclidienne entre deux matrices/vecteurs
    Entree :
        arr1, arr2 : numpy arrays qu'on veut calculer la distance qui les separe
    sortie :
        distance : float representant la distance entre arr1 et arr2
    """
    distance = np.sqrt(np.sum(np.square(arr1 - arr2, dtype=np.float64)))
    return distance

def get_distance_mat(descriptor_mat1, descriptor_mat2) :
    """
    Calculer la matrice de distance a partir de deux matrices des descripteurs.
    Entree :
        descriptor_mat1, descriptor_mat2 : numpy array contenant des descripteurs.
    Sortie :
        distance_mat : numpy array contenant les distances
    """
    nl, nc = descriptor_mat1.shape[0], descriptor_mat2.shape[0]
    distance_mat = np.zeros((nl, nc))
    for i in range(nl) :
        for j in range(nc) :
            distance_mat[i][j] = distance_euclidienne(descriptor_mat1[i], descriptor_mat2[j])
    return distance_mat

def trier_distance(mat) :
    """mat is a distance matrix that gets sorted then each of the distances are replaced by their original index"""
    l, c = mat.shape
    sorted_mat = np.zeros((l, c))
    for i in range(l) :
        arr = [n for n in mat[i] ]
        line = sorted(mat[i])
        for j in range(c) :
            index = np.where(arr == line[j])[0][0]
            sorted_mat[i, j] = index
            arr[index] = float("inf")
    return sorted_mat

def get_binary_mat(arr, n_training_per_class, n_test_per_class) :
    import math
    l, c = arr.shape
    binary_arr = np.zeros((l, c))
    for i in range(l) :
        k = math.floor(i/(n_training_per_class))
        for j in range(c) :
            if arr[i, j] >= k * n_test_per_class and arr[i, j] < (k + 1) * n_test_per_class :
                binary_arr[i, j] = 1
    return binary_arr

def get_mat_precision(arr) :
    l, c = arr.shape
    precision = np.zeros((l, c))
    for i in range(l) :
        for j in range(1, c + 1) :
            precision[i][j - 1] = np.sum(arr[i][:j]) / j
    return precision

def get_mean_precision_mat(arr1, arr2) :
    """arr1 is an index matrix to track is true positives while arr2 is a precision matrix used to calculate the mean precision"""
    l,c = arr2.shape
    mean = np.zeros(l)
    for i in range(l) :
        arr = []
        for j in range(c):
            if arr1[i][j] == 1 :
                arr.append(arr2[i][j])
        mean[i] = sum(arr) / len(arr)
    return mean