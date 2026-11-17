"""Codes clavier de cv2.waitKeyEx ramenes a des noms : les fleches et Suppr different selon le systeme."""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

SPECIAL_KEYS = {  # code numerique renvoye par OpenCV -> nom lisible, pour les touches qui ne sont pas des lettres
    27: "esc",  # Echap
    8: "backspace",  # retour arriere (Windows, Linux)
    127: "backspace",  # macOS
    13: "enter",  # Entree (retour chariot)
    10: "enter",  # Entree (saut de ligne, selon le systeme)
    32: "space",  # barre d'espace
    3014656: "delete",  # Windows
    65535: "delete",  # Linux (GTK)
    2424832: "left",  # Windows
    2555904: "right",  # fleche droite sous Windows
    65361: "left",  # Linux (GTK)
    65363: "right",  # fleche droite sous Linux (GTK)
    63234: "left",  # macOS
    63235: "right",  # fleche droite sous macOS
}


def key_name(code: int) -> str:  # traduit un code clavier OpenCV en nom de touche ; renvoie "" si rien d'utile
    """Code renvoye par cv2.waitKeyEx -> nom ("1", "a", "esc", "delete", "left"...) ; "" si aucune touche."""
    if code < 0:  # -1 = aucune touche pressee pendant l'attente
        return ""  # rien
    if code in SPECIAL_KEYS:  # touche speciale connue
        return SPECIAL_KEYS[code]  # son nom
    low = code & 0xFF  # garde l'octet de poids faible (le caractere), sans les bits de modificateurs (Maj, Ctrl...)
    return chr(low).lower() if 32 < low < 127 else ""  # caractere imprimable -> lettre/chiffre en minuscule ; sinon ""
