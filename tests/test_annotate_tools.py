"""Logique des outils d'annotation (sans fenetre), accord entre annotateurs, pre-annotation, etiquettes suspectes."""

# Les outils d'annotation (trieuse, annotateur de boites, revue) separent la logique ("etat" : que se passe-t-il
# quand on appuie sur telle touche) de la fenetre OpenCV. On teste ici la logique seule : pas besoin d'ecran.

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

import json  # json : lire des fichiers JSON (format texte de donnees structurees)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux de donnees avec colonnes nommees (DataFrame)
import pytest  # pytest : lance les tests ; pytest.approx compare des nombres a virgule avec une tolerance

# agreement : mesures d'accord entre deux annotateurs (classes, boites, masques) et appariement de boites
from vtb.annotate.agreement import box_agreement, classification_agreement, mask_agreement, match_boxes
from vtb.annotate.draw import draw_annotation, overlay_heatmap  # dessin des annotations et superposition d'une carte de chaleur
from vtb.annotate.label_issues import find_label_issues  # repere les images dont l'etiquette semble fausse

# preannotate : mesures d'incertitude du modele, tri des images a relire en premier, export des predictions
from vtb.annotate.preannotate import classification_uncertainty, detection_uncertainty, export_predictions, order_by_uncertainty
from vtb.annotate.schema import Box, ImageAnnotation, load_annotations  # classes du format pivot et relecture d'un vtb.json

# box_annotator : logique de l'outil qui trace des boites a la souris, sauvegarde et reprise d'une session
from vtb.annotate.tools.box_annotator import BoxAnnotatorState, save_session, start_session
from vtb.annotate.tools.keys import key_name  # traduit un code clavier brut d'OpenCV en nom lisible ("esc", "delete"...)
from vtb.annotate.tools.review import ReviewState  # logique de l'outil de revue des images les plus suspectes

# sorter : logique de la trieuse au clavier, copie des images triees, sauvegarde / relecture des decisions
from vtb.annotate.tools.sorter import SorterState, apply_copy, load_decisions, save_decisions


# Verifie que les codes clavier bruts (differents sous Windows et Linux) donnent le meme nom de touche.
# Attraperait : un outil qui marche sous Linux mais ou Suppr ou les fleches ne font rien sous Windows.
def test_key_names_cover_windows_and_linux_codes():
    # touche "3" (code ASCII), Echap (27) et retour arriere (8)
    assert key_name(ord("3")) == "3" and key_name(27) == "esc" and key_name(8) == "backspace"
    assert key_name(3014656) == key_name(65535) == "delete"  # Suppr : code Windows et code Linux donnent le meme nom
    # fleche droite Windows / Linux ; -1 (aucune touche pressee) donne un nom vide
    assert key_name(2555904) == key_name(65363) == "right" and key_name(-1) == ""


# Verifie le parcours complet de la trieuse : ranger, passer, touche invalide, annuler, reprendre, quitter.
# Attraperait : une annulation qui oublie d'effacer la copie, ou une reprise qui repropose des images deja triees.
def test_sorter_assign_skip_undo_resume_and_copies(tmp_path: Path):
    paths = []  # chemins des 4 fausses images
    for i in range(4):  # cree 4 fichiers 0.png a 3.png
        p = tmp_path / "in" / f"{i}.png"  # chemin du fichier i
        p.parent.mkdir(parents=True, exist_ok=True)  # cree le dossier "in" s'il n'existe pas
        p.write_bytes(b"x")  # contenu bidon : la trieuse ne lit pas l'image, elle la copie seulement
        paths.append(str(p))  # memorise le chemin
    state = SorterState(paths, ["ok", "defect"])  # trieuse a 2 classes : touche 1 = ok, touche 2 = defect
    action, path, label = state.handle_key("2")  # touche 2 : range l'image courante (la 0) dans "defect"
    assert (action, path, label) == ("assign", paths[0], "defect")  # action "ranger", sur l'image 0, classe defect
    apply_copy(action, path, label, tmp_path / "sorted")  # applique la decision : copie dans sorted/defect/
    assert (tmp_path / "sorted" / "defect" / "0.png").exists()  # la copie existe bien
    assert state.handle_key("s")[0] == "skip" and state.current() == paths[2]  # "s" passe l'image 1 : on arrive a l'image 2
    assert state.handle_key("9")[0] == "ignored"  # pas de 9e classe
    state.handle_key("1")  # range l'image 2 dans "ok"
    action, path, label = state.handle_key("backspace")  # retour arriere : annule la derniere decision
    # l'annulation concerne l'image 2 (classe ok) et on revient sur elle
    assert (action, path, label) == ("undo", paths[2], "ok") and state.current() == paths[2]
    apply_copy("undo", paths[0], "defect", tmp_path / "sorted")  # annulation d'une copie : doit supprimer sorted/defect/0.png
    assert not (tmp_path / "sorted" / "defect" / "0.png").exists()  # la copie a bien ete retiree
    save_decisions(state, tmp_path / "decisions.csv")  # enregistre les decisions (seule l'image 0 est rangee)
    # relance la trieuse comme apres une fermeture, en reprenant les decisions enregistrees
    resumed = SorterState(paths, ["ok", "defect"], decisions=load_decisions(tmp_path / "decisions.csv"))
    assert resumed.current() == paths[1]  # l'image 0 est deja triee, l'image 1 avait ete passee
    assert resumed.handle_key("q")[0] == "quit"  # "q" demande de quitter (sur la trieuse relancee)


# Verifie l'annotateur de boites : choix de classe, trace a la souris sur un affichage reduit, clic involontaire,
# souris hors image, suppression, sauvegarde et reprise. Attraperait : des boites aux mauvaises coordonnees (oubli
# du facteur d'echelle), des boites minuscules parasites, ou un travail perdu a la reprise.
def test_box_annotator_mouse_keys_scale_and_session(tmp_path: Path, voc_folder: Path):
    image = sorted((voc_folder / "images").glob("*.jpg"))[0]  # premiere image du jeu VOC (200 x 100)
    ann = ImageAnnotation(str(image), 200, 100)  # annotation vide de cette image : l'outil va la remplir
    state = BoxAnnotatorState(ann, ["crack", "hole"], scale=0.5)  # affichage reduit de moitie
    assert state.handle_key("2") == "class"  # touche 2 : choisit la classe 2 ("hole") pour les prochaines boites
    state.on_mouse("down", 10, 10)  # bouton enfonce au point (10, 10) de l'affichage
    state.on_mouse("move", 30, 20)  # la souris se deplace (apercu de la boite)
    state.on_mouse("up", 40, 30)  # bouton relache : la boite est creee
    assert ann.boxes == [Box("hole", 20.0, 20.0, 80.0, 60.0)]  # coordonnees ramenees en pixels image
    state.on_mouse("down", 5, 5)  # nouveau clic...
    state.on_mouse("up", 5.5, 5.5)  # clic involontaire : pas de boite
    assert len(ann.boxes) == 1  # toujours une seule boite
    state.on_mouse("down", 90, 40)  # debut d'une boite pres du coin bas droit
    state.on_mouse("up", 500, 500)  # la souris sort de l'image : boite bornee a l'image
    assert ann.boxes[-1].x2 == 200 and ann.boxes[-1].y2 == 100  # le coin est ramene au bord (largeur 200, hauteur 100)
    assert state.handle_key("delete") == "delete" and len(ann.boxes) == 1  # Suppr efface la derniere boite
    assert state.handle_key("n") == "next" and state.handle_key("esc") == "quit"  # n = image suivante, Echap = quitter
    # sauvegarde : un vtb.json pour tout + un .txt YOLO par image
    save_session([ann], ["crack", "hole"], tmp_path / "boxes.vtb.json", tmp_path / "yolo")
    # la ligne YOLO commence par le numero de classe 1 (= "hole", 2e classe, numerotee a partir de 0)
    assert (tmp_path / "yolo" / f"{image.stem}.txt").read_text().startswith("1 ")
    assert load_annotations(tmp_path / "boxes.vtb.json")[0][0].boxes == ann.boxes  # le vtb.json relu contient les memes boites
    resumed = start_session([str(image)], tmp_path / "boxes.vtb.json")  # reprise de la session depuis le vtb.json
    assert resumed[0].boxes == ann.boxes  # les boites deja tracees sont retrouvees


# Verifie l'outil de revue : les images sont presentees de la plus suspecte (score le plus haut) a la moins
# suspecte, et l'annulation efface la decision. Attraperait : un ordre de revue faux ou une decision fantome.
def test_review_sorts_by_score_and_undo():
    items = pd.DataFrame({"path": ["a", "b", "c"], "score": [0.1, 0.9, 0.5]})  # 3 images avec leur score d'anomalie
    state = ReviewState(items)  # l'outil trie par score decroissant : b (0.9), c (0.5), a (0.1)
    assert state.current()["path"] == "b"  # on commence par la plus suspecte
    assert state.handle_key("d") == "decide" and state.current()["path"] == "c"  # "d" = defect pour b, puis on passe a c
    state.handle_key("g")  # "g" = good pour c
    assert state.handle_key("backspace") == "undo" and state.current()["path"] == "c"  # annule : retour sur c
    # tableau final : b garde "defect", c redevient sans decision, a n'a jamais ete vue
    assert state.to_frame().set_index("path")["review"].to_dict() == {"b": "defect", "c": "", "a": ""}


# Verifie les mesures d'accord entre deux annotateurs : classes (kappa), boites (precision, rappel, classes) et
# masques (IoU). Attraperait : des scores d'accord faux, donc une mauvaise idee de la qualite des etiquettes.
def test_classification_and_box_agreement():
    result = classification_agreement(["a", "a", "b", "b"], ["a", "a", "b", "a"])  # deux annotateurs, 4 images, 1 desaccord
    # accord brut 3/4 ; kappa (accord corrige du hasard) entre 0 et 1 exclus : ni parfait ni nul
    assert result["raw_agreement"] == 0.75 and 0 < result["kappa"] < 1
    ref = [ImageAnnotation("x", 100, 100, boxes=[Box("c", 0, 0, 10, 10), Box("h", 50, 50, 70, 70)])]  # annotateur de reference
    # second annotateur : boite 1 presque identique, boite 2 bien placee mais classe differente, plus une boite en trop
    other = [ImageAnnotation("x", 100, 100, boxes=[Box("c", 1, 1, 10, 10), Box("c", 50, 50, 70, 70), Box("h", 80, 80, 90, 90)])]
    pairs, unmatched_a, unmatched_b = match_boxes(ref[0].boxes, other[0].boxes)  # apparie les boites (meme classe, assez recouvrantes)
    # une seule paire (0, 0) ; la boite 1 de ref et les boites 1 et 2 de other restent seules
    assert [(i, j) for i, j, _ in pairs] == [(0, 0)] and unmatched_a == [1] and unmatched_b == [1, 2]
    agreement = box_agreement(ref, other)  # bilan global de l'accord de detection
    # rappel = 1 retrouvee sur 2 de ref ; precision = 1 confirmee sur 3 de other
    assert agreement["recall"] == 0.5 and agreement["precision"] == pytest.approx(1 / 3)
    assert agreement["label_agreement"] == 0.5  # la boite bien placee en (50, 50) porte une autre classe
    a, b = np.zeros((10, 10)), np.zeros((10, 10))  # deux masques vides 10 x 10
    a[:5], b[:4] = 1, 1  # a : 5 premieres lignes (50 pixels) ; b : 4 premieres lignes (40 pixels)
    # IoU (intersection sur union) = 40 pixels communs / 50 pixels au total = 0.8
    assert mask_agreement([a], [b])["mean_iou"] == pytest.approx(0.8)


# Verifie la pre-annotation : les images ou le modele hesite le plus passent en premier, les detections trop faibles
# sont filtrees a l'export. Attraperait : un ordre de relecture faux ou des propositions parasites a effacer a la main.
def test_preannotation_export_and_uncertainty_order(tmp_path: Path, voc_folder: Path):
    image = str(sorted((voc_folder / "images").glob("*.jpg"))[0])  # une image reelle du jeu VOC (meme image pour les 3)
    preds = [  # 3 predictions de detection fictives
        ImageAnnotation(image, 200, 100, boxes=[Box("crack", 0, 0, 10, 10, 0.95)]),  # tres sure (score 0.95)
        # une boite a 0.55 (hesitation maximale vers 0.5) et une a 0.1
        ImageAnnotation(image, 200, 100, boxes=[Box("crack", 0, 0, 10, 10, 0.55), Box("hole", 5, 5, 20, 20, 0.1)]),
        ImageAnnotation(image, 200, 100),  # aucune detection
    ]
    # incertitude = 1 - |2 x score - 1| (maximale a 0.5) ; on garde la pire de l'image ; image vide = 0
    assert detection_uncertainty(preds).tolist() == pytest.approx([0.1, 0.9, 0.0])
    assert order_by_uncertainty(preds) == [1, 0, 2]  # la plus incertaine d'abord
    # probabilites de classification (une ligne par image, une colonne par classe) : sure, tres hesitante, moyenne
    probs = np.array([[0.98, 0.01, 0.01], [0.4, 0.35, 0.25], [0.7, 0.2, 0.1]])
    assert order_by_uncertainty(probs) == [1, 2, 0]  # tri par marge entre les deux classes les plus probables
    # l'entropie (mesure de "dispersion" des probabilites) est plus grande pour l'image hesitante que pour l'image sure
    assert classification_uncertainty(probs, "entropy")[1] > classification_uncertainty(probs, "entropy")[0]
    # export vers Label Studio en retirant les objets de score < 0.25
    out = export_predictions(preds, "labelstudio", tmp_path / "tasks.json", document_root=voc_folder, min_score=0.25)
    tasks = json.loads(out.read_text())  # relit le fichier de taches produit
    assert len(tasks[1]["predictions"][0]["result"]) == 1  # la boite a 0.1 est filtree
    export_predictions(preds, "cvat", tmp_path / "pre.xml", class_names=["crack", "hole"])  # export CVAT : ne doit pas planter
    export_predictions(preds, "xanylabeling", tmp_path / "labelme")  # export X-AnyLabeling (JSON LabelMe, un par image)
    # les 3 predictions visent le meme fichier : le JSON final est celui de la 3e (vide)
    assert json.loads((tmp_path / "labelme" / f"{Path(image).stem}.json").read_text())["shapes"] == []  # 3e image (meme nom) ecrase


# Verifie qu'une etiquette volontairement inversee est detectee comme suspecte, et elle seule.
# Attraperait : un detecteur d'erreurs d'etiquetage muet (il ne trouve rien) ou trop bavard (fausses alertes).
def test_label_issues_flags_a_swapped_label():
    rng = np.random.default_rng(0)  # generateur aleatoire a graine fixe : test reproductible
    labels = np.array([0] * 20 + [1] * 20)  # 40 images : 20 de classe 0 puis 20 de classe 1
    # probabilites "hors pli" (predites par un modele qui n'a pas vu l'image) : 0.9 pour la classe donnee, 0.1 pour l'autre,
    # plus un petit bruit aleatoire pour eviter des valeurs toutes identiques
    probs = np.where(labels[:, None] == np.arange(2)[None, :], 0.9, 0.1) + rng.uniform(0, 0.02, (40, 2))
    probs[3] = [0.05, 0.95]  # image 3 etiquetee 0 mais le modele hors pli est sur que c'est 1
    probs = probs / probs.sum(axis=1, keepdims=True)  # renormalise : chaque ligne somme a 1, comme de vraies probabilites
    table = find_label_issues(labels, probs, class_names=["ok", "defect"])  # tableau trie de la plus suspecte a la moins suspecte
    # la plus suspecte est l'image 3, elle est marquee suspecte, et c'est la seule
    assert table.iloc[0]["index"] == 3 and bool(table.iloc[0]["suspect"]) and table["suspect"].sum() == 1


# Verifie que les fonctions de dessin renvoient une image couleur RGB (3 canaux) de la taille d'origine, meme a partir
# d'une image grise. Attraperait : une forme d'image inattendue qui ferait planter l'affichage ou l'enregistrement.
def test_drawing_helpers_return_rgb_images():
    image = np.zeros((40, 60), np.uint8)  # image grise noire 40 x 60 (un seul canal)
    # dessine une boite avec son score
    drawn = draw_annotation(image, ImageAnnotation("x", 60, 40, boxes=[Box("c", 5, 5, 20, 20, 0.5)]))
    assert drawn.shape == (40, 60, 3) and drawn.sum() > 0  # image RGB de meme taille, et quelque chose a ete dessine
    heat = overlay_heatmap(image, np.random.default_rng(0).random((10, 15)))  # superpose une carte 10 x 15 (agrandie a l'image)
    assert heat.shape == (40, 60, 3) and heat.dtype == np.uint8  # RGB 8 bits de la taille de l'image
