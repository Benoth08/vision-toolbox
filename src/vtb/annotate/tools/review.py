"""Revue des anomalies : valider ou rejeter les images signalees par le detecteur, des plus suspectes aux moins suspectes.

    g : normal (good)     d : defaut       s : passer       retour arriere : annuler       q : quitter

Entree : un CSV avec au moins path et score (sortie de predict d'une recette d'anomalies), et
facultativement map_path (carte d'anomalie .npy ou image) pour la superposition. Sortie : le
meme tableau avec une colonne review (good / defect), reecrit a chaque decision. Les images
validees "good" enrichissent le jeu normal ; les "defect" forment un jeu de test reel.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import logging  # messages d'information et d'avertissement, plus souples que print
from dataclasses import dataclass, field  # dataclass : classe "fiche de donnees" ecrite en quelques lignes
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux a lignes et colonnes nommees (DataFrame), lecture / ecriture de CSV

logger = logging.getLogger(__name__)  # journal propre a ce module

DECISIONS = {"g": "good", "d": "defect"}  # touche -> decision enregistree dans la colonne review


@dataclass  # genere le constructeur a partir des champs ci-dessous
class ReviewState:  # etat de la revue, sans aucun affichage : testable sans ecran
    """Etat de la revue : tableau trie par score decroissant, decisions, historique."""

    items: pd.DataFrame  # images a revoir (une ligne par image : path, score, ...)
    decisions: dict[str, str] = field(default_factory=dict)  # chemin d'image -> "good" ou "defect"
    index: int = 0  # numero de la ligne affichee
    history: list[int] = field(default_factory=list)  # numeros deja traites, pour pouvoir annuler

    def __post_init__(self) -> None:  # appelee automatiquement juste apres le constructeur ; ne renvoie rien
        if not {"path", "score"} <= set(self.items.columns):  # <= entre ensembles : "path" et "score" sont-elles des colonnes ?
            raise ValueError("le CSV de revue doit contenir path et score")  # arret avec message clair
        # trie du score le plus haut (plus suspect) au plus bas ; renumerote les lignes 0, 1, 2...
        self.items = self.items.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
        self._advance()  # saute les images deja decidees (reprise d'une revue)

    def _advance(self) -> None:  # avance index jusqu'a la prochaine image sans decision ; ne renvoie rien
        while self.index < len(self.items) and self.items.loc[self.index, "path"] in self.decisions:  # tant que deja decidee
            self.index += 1  # passe a la suivante

    @property  # se lit comme un attribut : state.done
    def done(self) -> bool:  # True quand toutes les images ont ete vues
        return self.index >= len(self.items)  # l'index a depasse la derniere ligne

    def current(self) -> pd.Series | None:  # renvoie la ligne de l'image a afficher, ou None si fini
        return None if self.done else self.items.loc[self.index]  # .loc[i] = ligne numero i du tableau

    def handle_key(self, key: str) -> str:  # applique une touche ; renvoie le nom de l'action faite
        if key in ("q", "esc"):  # q ou Echap
            return "quit"  # demande d'arret
        if key in ("backspace", "u"):  # retour arriere ou u : annuler
            if not self.history:  # rien a annuler
                return "ignored"  # touche sans effet
            self.index = self.history.pop()  # revient a la derniere image traitee (et la retire de l'historique)
            self.decisions.pop(str(self.items.loc[self.index, "path"]), None)  # efface sa decision (None : pas d'erreur si absente)
            return "undo"  # action faite : annulation
        if self.done:  # plus aucune image a traiter
            return "done"  # rien a faire
        if key in DECISIONS:  # g ou d : decision
            self.decisions[str(self.items.loc[self.index, "path"])] = DECISIONS[key]  # enregistre "good" ou "defect"
            self.history.append(self.index)  # memorise pour pouvoir annuler
            self.index += 1  # image suivante
            self._advance()  # saute celles deja decidees
            return "decide"  # action faite : decision
        if key == "s":  # s : passer sans decider
            self.history.append(self.index)  # memorise pour pouvoir revenir
            self.index += 1  # image suivante
            return "skip"  # action faite : image passee
        return "ignored"  # toute autre touche est sans effet

    def to_frame(self) -> pd.DataFrame:  # renvoie le tableau avec une colonne review ajoutee
        out = self.items.copy()  # copie : le tableau d'origine reste intact
        out["review"] = [self.decisions.get(str(p), "") for p in out["path"]]  # decision de chaque image, "" si aucune
        return out  # renvoie le tableau complete


def load_map(map_path: str) -> np.ndarray | None:  # charge une carte d'anomalie ; renvoie un tableau 2D ou None
    """Carte d'anomalie depuis un .npy ou une image en niveaux de gris ; None si absente."""
    if not map_path or not Path(map_path).exists():  # chemin vide ou fichier absent
        return None  # pas de carte
    if map_path.endswith(".npy"):  # .npy = tableau numpy sauvegarde tel quel
        return np.load(map_path)  # relit le tableau
    from vtb.io.images import read_image  # import a la demande (lecture d'image)

    return read_image(map_path, color="gray").astype(np.float32)  # image en niveaux de gris, en flottants


# lance la fenetre de revue et la boucle clavier ; renvoie l'etat final (decisions)
def run_review(csv_in: Path, csv_out: Path, alpha: float = 0.45, max_display: int = 900) -> ReviewState:
    """Boucle d'affichage OpenCV (fenetre ; non testable sans ecran)."""
    import cv2  # OpenCV : fenetre, clavier, redimensionnement, texte

    from vtb.annotate.draw import overlay_heatmap  # superpose la carte d'anomalie en fausses couleurs
    from vtb.annotate.tools.keys import key_name  # code clavier -> nom de touche
    from vtb.io.images import read_image, to_uint8  # lecture d'image ; conversion en octets 0-255

    items = pd.read_csv(csv_in, keep_default_na=False)  # lit le CSV d'entree ; cases vides = "" et pas NaN (valeur manquante)
    previous = pd.read_csv(csv_out, keep_default_na=False) if csv_out.exists() else None  # revue precedente, si elle existe
    # decisions deja prises lors d'une session precedente (seulement les lignes avec une decision non vide)
    decisions = {} if previous is None else {str(p): r for p, r in zip(previous["path"], previous["review"]) if r}
    state = ReviewState(items, decisions=decisions)  # cree l'etat (trie, et saute les images deja decidees)
    window = "vtb - revue (g normal, d defaut, s passer, retour arriere annuler, q quitter)"  # titre de la fenetre = aide-memoire
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)  # cree une fenetre redimensionnable
    while not state.done:  # tant qu'il reste des images a revoir
        row = state.current()  # ligne de l'image courante
        image = to_uint8(read_image(row["path"], color="rgb"))  # lit l'image en couleur, octets 0-255
        heat = load_map(str(row.get("map_path", "")))  # carte d'anomalie si la colonne existe et le fichier aussi
        # image et image + carte cote a cote ; sans carte, l'image seule
        view = np.concatenate([image, overlay_heatmap(image, heat, alpha)], axis=1) if heat is not None else image
        scale = min(1.0, max_display / max(view.shape[:2]))  # facteur de reduction pour tenir dans max_display pixels
        # reduit si trop grand (INTER_AREA : interpolation adaptee a la reduction) ; sinon simple copie
        view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else view.copy()
        view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)  # OpenCV affiche en BGR (bleu, vert, rouge) : on convertit
        cv2.putText(  # ecrit la progression et le score en haut a gauche
            view,  # sur l'image affichee
            f"{state.index + 1}/{len(state.items)}  score {float(row['score']):.4f}",  # "3/120  score 0.8123"
            (8, 20),  # position du texte (x, y) en pixels
            cv2.FONT_HERSHEY_SIMPLEX,  # police
            0.5,  # taille de la police
            (0, 255, 255),  # couleur jaune en BGR
            1,  # epaisseur du trait
        )
        cv2.imshow(window, view)  # affiche l'image dans la fenetre
        if state.handle_key(key_name(cv2.waitKeyEx(0))) == "quit":  # attend une touche (0 = sans limite) et l'applique
            break  # q ou Echap : on sort de la boucle
        state.to_frame().to_csv(csv_out, index=False)  # sauvegarde apres chaque touche : rien n'est perdu en cas d'arret
    cv2.destroyAllWindows()  # ferme la fenetre
    state.to_frame().to_csv(csv_out, index=False)  # sauvegarde finale
    logger.info("%d decisions -> %s", len(state.decisions), csv_out)  # message de bilan
    return state  # renvoie l'etat final
