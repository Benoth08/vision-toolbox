"""Filtrage frequentiel (FFT) : retirer des motifs periodiques que l'espace image ne sait pas separer.

Cas typiques en siderurgie : traces regulieres de cylindres de laminoir, rayures fixes d'une
camera lineaire, trame d'un tissu ou d'une grille. Un motif periodique est un pic etroit dans le
spectre : on l'eteint (coupe-bande), et l'image garde tout le reste, defauts compris.

Frequences exprimees en cycles par image (0 = composante continue, au centre du spectre).
Equivalent OpenCvSharp : Cv2.Dft / Cv2.Idft sur une Mat flottante a deux canaux, puis
multiplication par le masque de frequences (le recentrage fftshift se fait en permutant les
quadrants).
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

import numpy as np  # numpy : calcul sur des tableaux de nombres ; np.fft = transformee de Fourier
from scipy.ndimage import maximum_filter, median_filter  # filtres maximum et median sur un voisinage (scipy)

from vtb.preprocess.dtypes import as_float, from_float, per_channel  # aller-retour vers les flottants ; application par canal
from vtb.preprocess.registry import register  # decorateur qui inscrit l'operation dans le registre


def _radius_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:  # renvoie les grilles de frequences (verticale, horizontale)
    """Coordonnees frequentielles centrees (cycles par image) de chaque case du spectre recentre."""
    fy = np.arange(height) - height // 2  # frequences verticales : -H/2 ... H/2 - 1 (0 au centre)
    fx = np.arange(width) - width // 2  # frequences horizontales : -W/2 ... W/2 - 1
    return np.meshgrid(fy, fx, indexing="ij")  # deux tableaux (H, W) : frequence verticale et horizontale de chaque case


# filtre un canal dans le domaine des frequences ; renvoie le canal filtre (flottants)
def _filter_channel(channel: np.ndarray, gain: np.ndarray, keep_mean: bool) -> np.ndarray:
    mean = float(channel.mean())  # niveau moyen du canal
    # FFT (transformee de Fourier) = decomposition de l'image en ondes ; fftshift met la frequence 0 au centre
    spectrum = np.fft.fftshift(np.fft.fft2(channel - mean))
    filtered = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * gain)))  # multiplie par le masque, revient en image (partie reelle)
    return filtered + (mean if keep_mean else 0.0)  # rajoute le niveau moyen si demande


# construit le masque coupe-bande ; renvoie un tableau de gains entre 0 et 1
def _notch_gain(shape: tuple[int, int], peaks: list[tuple[float, float]], radius: float) -> np.ndarray:
    """Masque multiplicatif : 1 partout, creux gaussien autour de chaque pic et de son symetrique."""
    yy, xx = _radius_grid(*shape)  # frequences de chaque case
    gain = np.ones(shape, dtype=np.float64)  # au depart on garde tout (gain 1)
    for py, px in peaks:  # pour chaque pic a eteindre
        for sy, sx in ((py, px), (-py, -px)):  # le pic et son symetrique (le spectre d'une image reelle est symetrique)
            gain *= 1.0 - np.exp(-((yy - sy) ** 2 + (xx - sx) ** 2) / (2.0 * radius**2))  # creux gaussien : 0 au pic, 1 loin
    return gain  # renvoie le masque


def find_periodic_peaks(  # detecte les pics du spectre dus a des motifs periodiques ; renvoie leurs coordonnees
    channel: np.ndarray,  # canal en flottants
    protect_radius: float = 8.0,  # rayon du disque central protege (le contenu normal de l'image)
    threshold: float = 8.0,  # seuil de detection, en ecarts absolus medians
    max_peaks: int = 20,  # nombre maximal de pics retenus
) -> list[tuple[float, float]]:  # renvoie une liste de (fy, fx)
    """Pics etroits du spectre (motifs periodiques), hors du disque central protege (contenu de l'image).

    Un pic est retenu s'il depasse le fond spectral local de `threshold` ecarts absolus medians :
    un spectre naturel decroit doucement, un motif periodique fait une pointe isolee.
    """
    height, width = channel.shape  # dimensions du canal
    magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(channel - channel.mean()))))  # amplitude du spectre en log (recentree)
    residual = magnitude - median_filter(magnitude, size=9, mode="reflect")  # amplitude moins le fond local : seules les pointes restent
    yy, xx = _radius_grid(height, width)  # frequences de chaque case
    outside = np.hypot(yy, xx) >= protect_radius  # True hors du disque central protege
    # ecart absolu median (MAD) du residu : mesure robuste de son niveau de bruit ; + 1e-9 evite 0
    mad = np.median(np.abs(residual[outside] - np.median(residual[outside]))) + 1e-9
    # candidats : hors disque, au-dessus du seuil, et maximum local dans une fenetre 5x5
    candidates = outside & (residual > threshold * mad) & (residual == maximum_filter(residual, size=5))
    order = np.argsort(residual[candidates])[::-1]  # ordre des candidats du plus fort au plus faible
    coords = list(zip(yy[candidates][order].tolist(), xx[candidates][order].tolist()))  # coordonnees (fy, fx) dans cet ordre
    peaks: list[tuple[float, float]] = []  # pics retenus
    for py, px in coords:  # un pic et son symetrique : on n'en garde qu'un, le masque traite les deux
        if (-py, -px) not in peaks:  # si son symetrique n'est pas deja retenu
            peaks.append((py, px))  # on retient ce pic
        if len(peaks) >= max_peaks:  # assez de pics
            break  # on arrete
    return peaks  # renvoie les pics trouves


@register("fft_notch", family="frequentiel")  # inscrit l'operation "fft_notch", famille frequentiel
def fft_notch(  # coupe-bande : eteint les motifs periodiques ; renvoie l'image filtree
    image: np.ndarray,  # image d'entree
    frequencies: list[list[float]] | None = None,  # pics [fy, fx] a eteindre ; None = detection automatique
    protect_radius: float = 8.0,  # rayon protege au centre du spectre (detection automatique)
    threshold: float = 8.0,  # seuil de detection des pics
    notch_radius: float = 2.0,  # largeur du creux autour de chaque pic
    max_peaks: int = 20,  # nombre maximal de pics eteints
) -> np.ndarray:  # renvoie une image du meme type que l'entree
    """Coupe-bande : eteint les pics periodiques du spectre (automatiquement, ou aux frequences [fy, fx] donnees).

    Mode automatique sans `frequencies` : les pics sont detectes image par image. En production,
    si le motif est fixe (camera lineaire, cylindre), mieux vaut les mesurer une fois et les
    passer explicitement : le resultat ne depend alors plus de l'image.
    """

    def one(channel: np.ndarray) -> np.ndarray:  # traite un canal ; renvoie des flottants
        x, _ = as_float(channel)  # canal en flottants [0, 1]
        if frequencies:  # frequences donnees dans la configuration
            peaks = [(float(f[0]), float(f[1])) for f in frequencies]  # converties en couples (fy, fx)
        else:  # sinon detection automatique
            peaks = find_periodic_peaks(x, protect_radius, threshold, max_peaks)  # cherche les pics dans ce canal
        if not peaks:  # aucun pic : rien a eteindre
            return x  # canal inchange
        return _filter_channel(x, _notch_gain(x.shape, peaks, notch_radius), keep_mean=True)  # applique le masque coupe-bande

    return from_float(per_channel(one, image), image)  # applique a chaque canal, retour au type d'origine


@register("fft_lowpass", family="frequentiel")  # inscrit l'operation "fft_lowpass"
def fft_lowpass(image: np.ndarray, cutoff: float = 30.0) -> np.ndarray:  # passe-bas ; renvoie l'image adoucie
    """Passe-bas gaussien (cutoff en cycles par image) : garde les variations lentes, retire grain et bruit fin."""

    def one(channel: np.ndarray) -> np.ndarray:  # traite un canal ; renvoie des flottants
        x, _ = as_float(channel)  # canal en flottants [0, 1]
        yy, xx = _radius_grid(*x.shape)  # frequences de chaque case
        # masque gaussien : 1 au centre (basses frequences), tend vers 0 au-dela de cutoff
        return _filter_channel(x, np.exp(-(yy**2 + xx**2) / (2.0 * cutoff**2)), keep_mean=True)

    return from_float(per_channel(one, image), image)  # applique a chaque canal, retour au type d'origine


@register("fft_highpass", family="frequentiel")  # inscrit l'operation "fft_highpass"
def fft_highpass(image: np.ndarray, cutoff: float = 10.0) -> np.ndarray:  # passe-haut ; renvoie l'image sans variations lentes
    """Passe-haut gaussien (cutoff en cycles par image) : retire l'eclairage et les gradients lents, garde les details.

    La moyenne est rajoutee : l'image reste centree sur son niveau de gris d'origine.
    """

    def one(channel: np.ndarray) -> np.ndarray:  # traite un canal ; renvoie des flottants
        x, _ = as_float(channel)  # canal en flottants [0, 1]
        yy, xx = _radius_grid(*x.shape)  # frequences de chaque case
        # masque inverse du passe-bas : 0 au centre (variations lentes retirees), 1 pour les hautes frequences
        return _filter_channel(x, 1.0 - np.exp(-(yy**2 + xx**2) / (2.0 * cutoff**2)), keep_mean=True)

    return from_float(per_channel(one, image), image)  # applique a chaque canal, retour au type d'origine
