"""Trieuse de classification au clavier : une image s'affiche, une touche la range dans une classe.

    touches 1..9 : classe 1..9          retour arriere : annuler la derniere decision
    s : passer (elle reviendra au prochain lancement)   q ou Echap : quitter (tout est deja enregistre)

Le CSV (path, label) est reecrit a chaque decision : on peut quitter et reprendre sans rien
perdre, les images deja triees ne sont pas reproposees. Avec copy_to, chaque image est aussi
copiee dans copy_to/<classe>/ : le resultat est directement un dossier "un sous-dossier par
classe", lisible par le catalogue (layout class_folders).

La logique (SorterState) est separee de la fenetre (run_sorter) : elle se teste sans ecran.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import logging  # messages d'information et d'avertissement, plus souples que print
import shutil  # copie et suppression de fichiers
from dataclasses import dataclass, field  # dataclass : classe "fiche de donnees" ecrite en quelques lignes
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import pandas as pd  # pandas : tableaux a lignes et colonnes nommees (DataFrame), lecture / ecriture de CSV

logger = logging.getLogger(__name__)  # journal propre a ce module


@dataclass  # genere le constructeur a partir des champs ci-dessous
class SorterState:  # etat de la trieuse, sans aucun affichage : testable sans ecran
    """Etat de la trieuse : images, decisions prises, position courante, historique pour annuler."""

    paths: list[str]  # chemins des images a trier, dans l'ordre de presentation
    class_names: list[str]  # classes possibles (touche 1 = premiere classe, etc.)
    decisions: dict[str, str] = field(default_factory=dict)  # chemin d'image -> classe choisie
    index: int = 0  # position de l'image affichee dans paths
    history: list[tuple[int, str | None]] = field(default_factory=list)  # (position, chemin decide ou None si passe)

    def __post_init__(self) -> None:  # appelee automatiquement juste apres le constructeur ; ne renvoie rien
        if len(self.class_names) > 9:  # une seule touche chiffre par classe : 9 au maximum
            raise ValueError("la trieuse gere au plus 9 classes (touches 1 a 9)")  # arret avec message clair
        self.index = self._next_open(0)  # commence a la premiere image pas encore triee (reprise)

    def _next_open(self, start: int) -> int:  # renvoie la position de la prochaine image sans decision
        """Premiere image sans decision a partir de start (len(paths) si tout est trie)."""
        i = start  # on part de la position donnee
        while i < len(self.paths) and self.paths[i] in self.decisions:  # tant que l'image est deja triee
            i += 1  # on avance
        return i  # premiere position libre (ou la fin de la liste)

    @property  # se lit comme un attribut : state.done
    def done(self) -> bool:  # True quand il ne reste plus d'image a montrer
        return self.index >= len(self.paths)  # la position a depasse la derniere image

    def current(self) -> str | None:  # renvoie le chemin de l'image a afficher, ou None si fini
        return None if self.done else self.paths[self.index]  # image a la position courante

    def handle_key(self, key: str) -> tuple[str, str | None, str | None]:  # applique une touche ; renvoie (action, chemin, classe)
        """Applique une touche ; renvoie (action, chemin concerne, classe) pour que l'appelant fasse les copies."""
        if key in ("q", "esc"):  # q ou Echap
            return "quit", None, None  # demande d'arret
        if key in ("backspace", "u"):  # retour arriere ou u : annuler
            if not self.history:  # rien a annuler
                return "ignored", None, None  # touche sans effet
            position, decided = self.history.pop()  # derniere action : sa position et l'image decidee (None si passee)
            label = self.decisions.pop(decided, None) if decided else None  # efface la decision et recupere la classe
            self.index = position  # revient a cette image
            return "undo", decided, label  # l'appelant supprimera la copie eventuelle
        if self.done:  # plus aucune image
            return "done", None, None  # rien a faire
        path = self.paths[self.index]  # image courante
        if key == "s":  # s : passer sans decider
            self.history.append((self.index, None))  # memorise la position (None = aucune decision)
            self.index = self.index + 1  # image suivante
            while self.index < len(self.paths) and self.paths[self.index] in self.decisions:  # saute les images deja triees
                self.index += 1  # on avance
            return "skip", path, None  # image passee
        if key.isdigit() and 1 <= int(key) <= len(self.class_names):  # un chiffre qui correspond a une classe
            label = self.class_names[int(key) - 1]  # touche 1 -> classe d'indice 0
            self.decisions[path] = label  # enregistre la decision
            self.history.append((self.index, path))  # memorise pour pouvoir annuler
            self.index = self._next_open(self.index + 1)  # passe a la prochaine image non triee
            return "assign", path, label  # l'appelant fera la copie eventuelle
        return "ignored", None, None  # toute autre touche est sans effet

    def to_frame(self) -> pd.DataFrame:  # renvoie les decisions sous forme de tableau (path, label)
        return pd.DataFrame({"path": list(self.decisions), "label": list(self.decisions.values())})  # une ligne par image triee


def save_decisions(state: SorterState, csv_path: Path) -> None:  # ecrit les decisions dans le CSV ; ne renvoie rien
    csv_path.parent.mkdir(parents=True, exist_ok=True)  # cree le dossier du CSV s'il n'existe pas
    state.to_frame().to_csv(csv_path, index=False)  # ecrit le tableau sans la colonne de numeros de ligne


def load_decisions(csv_path: Path) -> dict[str, str]:  # relit les decisions d'une session precedente ; renvoie chemin -> classe
    if not csv_path.exists():  # pas encore de CSV : premiere session
        return {}  # aucune decision
    table = pd.read_csv(csv_path, keep_default_na=False)  # lit le CSV ; cases vides = "" et pas NaN (valeur manquante)
    return dict(zip(table["path"].astype(str), table["label"].astype(str)))  # associe chaque chemin a sa classe


# copie ou retire le fichier image selon l'action ; ne renvoie rien
def apply_copy(action: str, path: str | None, label: str | None, copy_to: Path | None) -> None:
    """Effet de bord d'une decision : copie (assign) ou suppression de la copie (undo)."""
    if copy_to is None or path is None or label is None:  # pas de dossier de copie, ou action sans image / classe
        return  # rien a faire
    target = copy_to / label / Path(path).name  # destination : copy_to/<classe>/<nom de l'image>
    if action == "assign":  # nouvelle decision
        target.parent.mkdir(parents=True, exist_ok=True)  # cree le dossier de la classe si besoin
        shutil.copy2(path, target)  # copie l'image (copy2 garde la date du fichier)
    elif action == "undo" and target.exists():  # annulation et la copie existe
        target.unlink()  # supprime la copie


# lance la fenetre de tri et la boucle clavier ; renvoie l'etat final (decisions)
def run_sorter(
    paths: list[str],  # images a trier
    class_names: list[str],  # classes possibles (9 au plus)
    csv_path: Path,  # CSV des decisions (relu au lancement, reecrit a chaque decision)
    copy_to: Path | None = None,  # dossier ou copier les images par classe ; None = pas de copie
    max_display: int = 900,  # taille maximale d'affichage en pixels
) -> SorterState:  # renvoie l'etat final
    """Boucle d'affichage OpenCV (fenetre ; non testable sans ecran)."""
    import cv2  # OpenCV : fenetre, clavier, redimensionnement, texte

    from vtb.annotate.tools.keys import key_name  # code clavier -> nom de touche
    from vtb.io.images import read_image, to_uint8  # lecture d'image ; conversion en octets 0-255

    state = SorterState(paths, class_names, decisions=load_decisions(csv_path))  # etat initial, avec les decisions deja prises
    window = "vtb - trieuse (1-9 classe, retour arriere annuler, s passer, q quitter)"  # titre = aide-memoire des touches
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)  # cree une fenetre redimensionnable
    while not state.done:  # tant qu'il reste des images a trier
        image = to_uint8(read_image(state.current(), color="rgb"))  # lit l'image courante en couleur, octets 0-255
        scale = min(1.0, max_display / max(image.shape[:2]))  # facteur de reduction pour tenir dans max_display pixels
        # reduit si trop grand (INTER_AREA : adapte a la reduction) ; sinon simple copie
        view = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
        view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)  # OpenCV affiche en BGR : on convertit
        legend = "  ".join(f"{i + 1}:{name}" for i, name in enumerate(class_names))  # "1:ok  2:rayure  3:tache"
        cv2.putText(  # ecrit la progression et la legende en haut a gauche
            view,  # sur l'image affichee
            f"{state.index + 1}/{len(paths)}  {legend}",  # "12/300  1:ok  2:rayure"
            (8, 20),  # position (x, y) en pixels
            cv2.FONT_HERSHEY_SIMPLEX,  # police
            0.5,  # taille de la police
            (0, 255, 255),  # couleur jaune en BGR
            1,  # epaisseur du trait
            cv2.LINE_AA,  # trait lisse (anti-crenelage)
        )
        cv2.imshow(window, view)  # affiche l'image
        action, path, label = state.handle_key(key_name(cv2.waitKeyEx(0)))  # attend une touche (sans limite) et l'applique
        if action == "quit":  # q ou Echap
            break  # sort de la boucle
        apply_copy(action, path, label, copy_to)  # copie ou retire le fichier si besoin
        if action in ("assign", "undo"):  # si les decisions ont change
            save_decisions(state, csv_path)  # sauvegarde tout de suite : rien n'est perdu en cas d'arret
    cv2.destroyAllWindows()  # ferme la fenetre
    logger.info("%d images triees -> %s", len(state.decisions), csv_path)  # message de bilan
    return state  # renvoie l'etat final
