"""Garde-fous de tools/runpod/run_remote.py, sans reseau ni cle : lecture de la cle, archive du code, simulation."""

# run_remote.py lance un calcul sur un GPU loue chez RunPod (service de GPU a la demande). Ces tests verifient
# les garde-fous sans rien louer : la cle secrete est bien lue, l'archive envoyee ne contient ni donnees ni
# secrets, et le mode simulation (--dry-run) ne cree rien.

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

import importlib.util  # importlib.util : charger un fichier Python comme module a partir de son chemin
import tarfile  # tarfile : lire / ecrire des archives .tar.gz
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import pytest  # pytest : lance les tests ; raises, monkeypatch...

REPO = Path(__file__).resolve().parents[1]  # racine du depot (le dossier parent de tests/)


# charge tools/runpod/run_remote.py comme un module et le renvoie (ce script ne fait pas partie du paquet vtb)
def _module():
    # decrit le module a charger : son nom et le chemin du fichier
    spec = importlib.util.spec_from_file_location("run_remote", REPO / "tools" / "runpod" / "run_remote.py")
    module = importlib.util.module_from_spec(spec)  # cree un module vide a partir de cette description
    spec.loader.exec_module(module)  # execute le fichier dans ce module (definit ses fonctions)
    return module  # renvoie le module, utilisable comme un import normal


# Verifie l'ordre de lecture de la cle d'API : variable d'environnement d'abord, sinon fichier .env, sinon arret clair.
# Attraperait : une cle introuvable qui provoquerait une erreur obscure, ou un .env qui ecraserait la variable choisie.
def test_api_key_comes_from_environment_or_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_remote = _module()  # charge le script
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)  # retire la variable d'environnement le temps du test (si elle existe)
    with pytest.raises(SystemExit, match="RUNPOD_API_KEY absente"):  # on attend un arret du programme avec ce message...
        run_remote.load_api_key(tmp_path / ".env")  # ...car ni variable ni fichier .env
    (tmp_path / ".env").write_text("OTHER=1\nRUNPOD_API_KEY='abc123'\n", encoding="utf-8")  # fichier .env avec la cle (entre apostrophes)
    assert run_remote.load_api_key(tmp_path / ".env") == "abc123"  # cle lue dans le fichier, apostrophes retirees
    monkeypatch.setenv("RUNPOD_API_KEY", "from-env")  # definit maintenant la variable d'environnement
    assert run_remote.load_api_key(tmp_path / ".env") == "from-env"  # la variable d'environnement est prioritaire


# Verifie que l'archive du code envoyee au GPU distant ne contient ni donnees, ni sorties, ni poids, ni secrets.
# Attraperait : un envoi de gigaoctets de donnees (lent et couteux) ou la fuite du fichier .env et de sa cle.
def test_code_archive_never_contains_data_outputs_weights_or_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_remote = _module()  # charge le script
    fake = tmp_path / "repo"  # faux depot dans le dossier temporaire
    # pour chaque fichier a creer : du code, des donnees, une sortie, des poids, un secret, une doc
    for relative in ("src/vtb/a.py", "data/raw/x.png", "outputs/run/model.pt", "weights/w.pth", ".env", "recipes/r/README.md"):
        (fake / relative).parent.mkdir(parents=True, exist_ok=True)  # cree son dossier
        (fake / relative).write_text("x", encoding="utf-8")  # ecrit un contenu bidon
    monkeypatch.setattr(run_remote, "REPO", fake)  # le script croit que le depot est ce faux depot (le temps du test)
    names = tarfile.open(run_remote.make_archive(tmp_path / "code.tar.gz")).getnames()  # cree l'archive et liste son contenu
    # le code et les recettes sont bien dans l'archive
    assert "vision-toolbox/src/vtb/a.py" in names and "vision-toolbox/recipes/r/README.md" in names
    # aucun nom ne contient /data/, /outputs/, /weights/ ou .env
    assert not [n for n in names if any(part in n for part in ("/data/", "/outputs/", "/weights/", ".env"))]


# Verifie que l'adresse SSH (connexion a distance) est lue dans la description du pod (machine louee).
# Attraperait : une connexion tentee sur le mauvais port, ou un plantage quand le pod n'est pas encore pret.
def test_ssh_endpoint_is_read_from_the_pod_runtime() -> None:
    run_remote = _module()  # charge le script
    pod = {  # description d'un pod telle que la renvoie l'API RunPod (simplifiee)
        "runtime": {  # informations disponibles une fois le pod demarre
            "ports": [  # ports ouverts : port interne -> adresse et port publics
                {"privatePort": 8888, "isIpPublic": True, "ip": "1.2.3.4", "publicPort": 1},  # port 8888 (Jupyter) : pas celui cherche
                {"privatePort": 22, "isIpPublic": True, "ip": "1.2.3.4", "publicPort": 40022},  # port 22 = SSH, expose sur 40022
            ]
        }
    }
    assert run_remote.ssh_endpoint(pod) == ("1.2.3.4", 40022)  # adresse et port publics du SSH
    assert run_remote.ssh_endpoint({"runtime": None}) is None  # pod pas encore demarre : None, sans plantage


# Verifie que le mode simulation (--dry-run) se termine bien, sans cle et sans rien creer.
# Attraperait : une simulation qui lancerait (et facturerait) un vrai GPU.
def test_dry_run_creates_nothing(monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # capsys : capture ce que le script affiche
    run_remote = _module()  # charge le script
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)  # la simulation ne lit meme pas la cle
    monkeypatch.setattr(  # remplace les arguments de la ligne de commande le temps du test
        "sys.argv",  # la liste des arguments recus par le script (sys.argv)
        # comme si on tapait : run_remote.py run --gpu "NVIDIA RTX A5000" --max-minutes 10 --command pytest --dry-run
        ["run_remote.py", "run", "--gpu", "NVIDIA RTX A5000", "--max-minutes", "10", "--command", "pytest", "--dry-run"],
    )
    assert run_remote.main() == 0  # code de retour 0 = tout s'est bien passe
