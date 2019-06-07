import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import Lasso
import random

CST_MISSING = -100

def normalize_data(im, zero_one=True):
    '''
        transforme les valeurs de chaque canal couleur de l'image
        en des valeurs dans [0,1]

        paramètres
        ----------
        im : np.array, shape (n_height, n_width, 3)
             tenseur correspondant à l'image
        zero_one : boolean
        renvoie
        -------
        im_norm : np.array, shape (n_height, n_width, 3)
                  tenseur correspondant à l'image et dont les
                  valeurs sont normalisées dans [-1,1]

    '''
    height, width, _ = im.shape
    im_norm = im.copy().astype(float)

    
    min_0 = np.nanmin(im_norm[:,:,0])
    max_0 = np.nanmax(im_norm[:,:,0])

    min_1 = np.nanmin(im_norm[:,:,1])
    max_1 = np.nanmax(im_norm[:,:,1])

    min_2 = np.nanmin(im_norm[:,:,2])
    max_2 = np.nanmax(im_norm[:,:,2])

    if zero_one:
        im_norm[:,:,0] = (im_norm[:,:,0] - min_0) / (max_0 - min_0)
        im_norm[:,:,1] = (im_norm[:,:,1] - min_1) / (max_1 - min_1)
        im_norm[:,:,2] = (im_norm[:,:,2] - min_2) / (max_2 - min_2)
    else:
        im_norm[:,:,0] = 2 * ((im_norm[:,:,0] - min_0) / (max_0 - min_0)) - 1
        im_norm[:,:,1] = 2 * ((im_norm[:,:,1] - min_1) / (max_1 - min_1)) - 1
        im_norm[:,:,2] = 2 * ((im_norm[:,:,2] - min_2) / (max_2 - min_2)) -1
    

    return im_norm

def read_im(fn):
    '''
        permet de lire une image

        paramètres
        ----------
        fn : string
             nom de l'image
        renvoie
        -------
        imdata : np.array, shape (n_height, n_width, 3)
                 tenseur correspondant à l'image convertie
                 en hsv et dont les valeurs sont normalisées
                 dans [-1,1]
    '''
    imdata = plt.imread(fn).copy()
    # pour pouvoir convertir en hsv
    #imdata = normalize_data(imdata)
    # rgb -> hsv
    #imdata = matplotlib.colors.rgb_to_hsv(imdata)
    # -> [-1,1]
    imdata = normalize_data(imdata, zero_one=False)

    return imdata

def show_im(imdata):
    '''
        permet d'afficher une image

        paramètres
        ----------
        imdata : np.array, shape (n_height, n_width, 3)
                 tenseur correspondant à l'image convertie
                 en hsv et dont les valeurs sont normalisées
                 dans [-1,1]
    '''
    global CST_MISSING
    height, width, _ = imdata.shape
    imdata = imdata.copy()

    imdata_flat = imdata.flatten()
    ind_invalid = np.argwhere(imdata_flat == CST_MISSING)
    imdata_flat[ind_invalid] = np.nan
    imdata = imdata_flat.reshape(height, width, 3)

    # -> [0,1]
    imdata = normalize_data(imdata)
    # hsv -> rgb
    #imdata = matplotlib.colors.hsv_to_rgb(imdata)
    plt.imshow(imdata)
    plt.show()

def get_patch(i, j, h, im):
    '''
        permet de retourner le patch centré en (i,j) et de longueur h
        d'une image im

        paramètres
        ----------
        i, j : int, int
               coordonnées du pixel dans l'image (hauteur, largeur)
        h : int
            longueur des côtés du patch
        im : np.array, shape (height, width, 3)
             image représentée sous la forme d'un tenseur 3d
        renvoie
        -------
        patch : np.array, shape (patch_height, patch_width, 3)
                tenseur 3d correspondant à un patch
    '''

    assert h % 2 == 1 and h > 1 # pour pouvoir centrer en (i,j)

    height, width, _ = im.shape

    i_min = i-int(h/2)
    i_max = i+int(h/2)+1
    j_min = j-int(h/2)
    j_max = j+int(h/2)+1

    # si le pixel est au bord de l'image alors le patch est plus petit
    # peut-être qu'il faudra changer ça plus tard ?
    # on peut aussi supposer que l'on ne centrera jamais sur les pixels au bord ?
    if i_min < 0:
        i_min = 0
    if j_min < 0:
        j_min = 0
    if i_max >= height:
        i_max = height
    if j_max >= width:
        j_max = width

    patch = im[i_min:i_max,j_min:j_max]
    return patch

def from_patch_to_vect(patch):
    '''
        convertit un patch en vecteur

        paramètres
        ----------
        patch : np.array, shape (h, h, 3)
                tenseur 3d correspondant à un patch
        renvoie
        -------
        vect : np.array, (3 * h^2, )
               vecteur correspondant au patch
    '''
    vect = patch.flatten()
    return vect

def from_vect_to_patch(vect, h):
    '''
        convertit un vecteur en patch

        paramètres
        ----------
        vect : np.array, shape (3 * h^2, )
               vecteur
        h : longueur des côtés d'un patch
        renvoie
        -------
        patch : np.array, (h, h, 3)
               tenseur 3d correspondant au vecteur
    '''
    assert h % 2 == 1 and h > 1

    patch = vect.reshape(h, h, 3)
    return patch

def noise(im, prc):
    '''
        permet de supprimer au hasard un pourcentage de pixel
        dans l'image

        paramètres
        ----------
        im : np.array, shape (height, width, 3)
             image représentée sous la forme d'un tenseur 3d
        prc : float (0 <= prc <= 100)
              pourcentage de l'image à bruiter
        renvoie
        -------
        noisy_im : np.array, shape (height, width, 3)
                   image dont prc % de l'image a été bruité
                   (les pixels manquants ont pour valeur
                   -100)
    '''
    global CST_MISSING
    assert 0 <= prc and prc <= 100
    prc = prc / 100
    height, width, _ = im.shape
    n_pixels = height * width

    #im_flat = im.flatten()
    im_reshaped = im.reshape(n_pixels, 3).copy()

    # sélection aléatoire du premier canal de prc*n_pixels
    ind_to_delete = np.array(random.sample(range(0, n_pixels), int(prc*n_pixels)))

    # suppression des pixels correspondant aux canaux
    im_reshaped[ind_to_delete] = CST_MISSING

    im = im_reshaped.reshape(height, width, 3)
    return im

def delete_rect(im, i, j, height, width):
    '''
        permet de supprimer tout un rectangle de l'image

        paramètres
        ----------
        im : np.array, shape (height, width, 3)
             image représentée sous la forme d'un tenseur 3d
        i,j : int, int
              coordonnées du pixel au coin en haut à gauche
              du rectangle
        height : int
                 hauteur du rectangle
        width : int
                largeur du rectangle
        renvoie
        -------
        noisy_im : np.array, shape (height, width, 3)
                   image dont le rectangle spécifié a été supprimé
    '''
    global CST_MISSING
    n,p,_ = im.shape

    ind_i = np.arange(i,i+height+1)
    ind_j = np.arange(j,j+width+1)

    noisy_im = im.copy()

    for i in ind_i:
        for j in ind_j:
            noisy_im[i,j,:] = CST_MISSING

    return noisy_im

def get_missing_pixel_patches(im, h):
    '''
        permet de renvoyer les patchs de l'image qui contiennent
        des pixels manquants

        paramètres
        ----------
        im : np.array, shape (height, width, 3)
             image représentée sous la forme d'un tenseur 3d
        h : int
            longueur des côtés du patch
        renvoie
        -------
        patches : list of np.array, shape (patch_height, patch_width, 3)
                  liste des patchs contenant des pixels manquants
        ind : list of tuple (int, int)
              liste des indices des centres des patchs contenant des
              pixels manquants
    '''
    global CST_MISSING
    assert h % 2 == 1 and h > 1

    height, width, _ = im.shape
    patches = []
    ind = []

    for i in range(int(h/2),height, h):
        for j in range(int(h/2), width, h):
            p = get_patch(i, j, h, im)
            if CST_MISSING in from_patch_to_vect(p):
                patches.append(p)
                ind.append((i,j))

    return patches, ind


def get_random_full_patch(im, h):
    '''
        renvoie un patch ne contenant aucun pixel manquant et qui
        ne fait pas partie du dictionnaire

        im : np.array, shape (height, width, 3)
             image représentée sous la forme d'un tenseur 3d
        h : int
            longueur des côtés du patch
    '''
    assert h % 2 == 1 and h > 1

    height, width, _ = im.shape
    patches = []

    for i in range(int(h/2)+1,height, h):
        for j in range(int(h/2)+1, width, h):
            p = get_patch(i, j, h, im)
            if CST_MISSING not in from_patch_to_vect(p):
                patches.append(p)

    return patches[random.randint(0,len(patches)-1)]


def get_dictionary(im, h):
    '''
        permet de renvoyer les patchs qui ne contiennent aucun pixel
        manquant

        paramètres
        ----------
        im : np.array, shape (height, width, 3)
             image représentée sous la forme d'un tenseur 3d
        h : int
            longueur des côtés du patch
        renvoie
        -------
        patches : list of np.array, shape (patch_height, patch_width, 3)
                  liste des patchs ne contenant aucun pixel manquant
    '''
    global CST_MISSING
    assert h % 2 == 1 and h > 1

    height, width, _ = im.shape
    patches = []

    # construire des patchs tous les h pixels
    for i in range(int(h/2),height, h):
        for j in range(int(h/2), width, h):
            p = get_patch(i, j, h, im)
            if CST_MISSING not in from_patch_to_vect(p):
                patches.append(p)

    return patches

def compute_vectors_for_optimization(patch, dictionary):
    '''
        transforme un patch en vecteur et le dictionnaire en matrice
        en retirant les pixels correspondant aux manquants du patch

        paramètres
        ----------
        patch : np.array, shape (h, h, 3)
                tenseur 3d correspondant à un patch
        dictionary : list of np.array, shape (patch_height, patch_width, 3)
                     liste des patchs ne contenant aucun pixel manquant
        renvoie
        -------
        vect : np.array, shape (3 * h^2, )
               vecteur correspondant au patch restreint aux pixels exprimés
        X : np.array, shape (n_dict, 3 * h^2)
            matrice des vecteurs correspondant aux patchs du dictionnaire,
            restreints aux pixels exprimés dans patch
        missing_ind : np.array, shape (n_missing,1)
                      vecteur contenant les indices des pixels manquants
                      dans vect
    '''
    global CST_MISSING
    vect = from_patch_to_vect(patch)
    missing_ind = np.argwhere(vect == CST_MISSING)
    vect = np.delete(vect, missing_ind) # restriction aux pixels exprimés

    X = np.array([from_patch_to_vect(p) for p in dictionary])
    X = np.delete(X, missing_ind, 1) # restriction aux pixels exprimés du patch
    return vect, X, missing_ind

def compute_weights_for_patch(vect, X, alpha=1.0, max_iter=1000):
    '''
        rend le vecteur de poids sur le dictionnaire qui approxime au
        mieux le patch en utilisant l'algorithme du Lasso

        paramètres
        ----------
        vect : np.array, shape (3 * h^2, )
               vecteur correspondant au patch restreint aux pixels exprimés
        X : np.array, shape (n_dict, 3 * h^2)
            matrice des vecteurs correspondant aux patchs du dictionnaire,
            restreints aux pixels exprimés dans patch
        alpha : float (1.0 par défaut)
                constante multiplicatrice du terme L1
        renvoie
        -------
        lasso.coef_ : np.array, shape (n_dict,)
                      vecteur des poids associés à chaque patch du dictionnaire
    '''
    lasso = Lasso(alpha=alpha,max_iter=max_iter)

    # X.T : matrice avec autant de lignes que de pixels dans le patch
    #       et autant de colonnes que de patchs dans le dictionnaire
    # vect : vérité terrain i.e. valeur de chaque pixel exprimé
    lasso.fit(X.T, vect)
    return lasso.coef_

def get_missing_pixel_indices_in_patch(patch):
    vect = from_patch_to_vect(patch)
    missing_ind = np.argwhere(vect == CST_MISSING)
    return missing_ind

def predict_missing_pixels(missing_ind, dictionary, w):
    '''
        prédit les pixels manquants en les complétant par ceux de la
        combinaison linéaire de w avec les patchs du dictionnaire

        paramètres
        ----------
        missing_ind : np.array, shape (1, n_missing)
                      vecteur contenant les indices des pixels manquants
                      dans vect
        dictionary : list of np.array, shape (patch_height, patch_width, 3)
                     liste des patchs ne contenant aucun pixel manquant
        w : np.array, shape (n_dict,)
            vecteur des poids associés à chaque patch du dictionnaire
        renvoie
        -------
        predictions : np.array, shape (n_missing, )
                      vecteur des prédictions pour chaque pixel manquant

    '''
    X = np.array([from_patch_to_vect(p) for p in dictionary])
    predictions = np.dot(X.T, w)[missing_ind]
    # pour garder les valeurs entre -1 et 1
    #predictions_flat = predictions.flatten()
    #predictions_flat[(np.where(predictions_flat < -1))[0]] = -1
    #predictions_flat[(np.where(predictions_flat > 1))[0]] = 1
    #predictions = predictions_flat.reshape(predictions.shape[0],predictions.shape[1])
    return predictions

def loss(predictions, ground_truth, w, alpha):
    l = np.mean((predictions - ground_truth)**2) + alpha * \
        np.linalg.norm(w, ord=1)
    return l
def fill_patch(patch, h, dictionary, alpha=1, max_iter=1000):
    full_vect = from_patch_to_vect(patch)
    vect, X, missing_ind = compute_vectors_for_optimization(patch, dictionary)
    w = compute_weights_for_patch(vect, X, alpha=alpha, max_iter=max_iter)
    full_vect[missing_ind] = predict_missing_pixels(missing_ind, dictionary, w)
    patch_filled = from_vect_to_patch(full_vect, h)
    return patch_filled

def fill_image(im, h, dictionary, alpha=1.0, max_iter=1000):
    height, width, _ = im.shape
    im_filled = im.copy()
    for i in range(int(h/2),int(height/2)):
        for j in range(int(h/2),width):
            p1 = get_patch(i, j, h, im_filled)
            p2 = get_patch(height-(i+1), j, h, im_filled)

            if CST_MISSING in from_patch_to_vect(p1):
                i_min = i-int(h/2)
                i_max = i+int(h/2)+1
                j_min = j-int(h/2)
                j_max = j+int(h/2)+1
                #print("p1 : " , i_min,i_max,j_min,j_max)
                im_filled[i_min:i_max,j_min:j_max] = fill_patch(p1, h, dictionary, alpha=alpha, max_iter=max_iter)
            if CST_MISSING in from_patch_to_vect(p2):
                i_min = (height-i-1)-int(h/2)
                i_max = (height-i)+int(h/2)
                j_min = j-int(h/2)
                j_max = j+int(h/2)+1
                #print("p2 : ", i_min,i_max,j_min,j_max)
                im_filled[i_min:i_max,j_min:j_max] = fill_patch(p2, h, dictionary, alpha=alpha, max_iter=max_iter)

    return im_filled