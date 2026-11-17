"""Lancer une experience sur un GPU RunPod, avec garde-fous de cout : confirmation, duree maximale, suppression garantie.

    python tools/runpod/run_remote.py --list                       # pods existants (a lancer apres chaque session)
    python tools/runpod/run_remote.py --gpus                       # types de GPU et prix horaires
    python tools/runpod/run_remote.py --terminate <pod_id>
    python tools/runpod/run_remote.py run --gpu "NVIDIA RTX A5000" --max-minutes 60 \\
        --setup "pip install -e .[detection] && python tools/get_data/get_neu_det.py" \\
        --command "python recipes/detection/scripts/run.py --config recipes/detection/configs/yolo11n_neudet.yaml" \\
        --fetch outputs/detection/yolo11n_neudet
    python tools/runpod/run_remote.py run ... --dry-run            # montre ce qui serait fait, sans rien creer

Regles (voir CLAUDE.md) :
  - la cle est lue dans la variable d'environnement RUNPOD_API_KEY ou dans un fichier .env ignore
    par git ; elle n'est jamais affichee, ecrite, ni passee en argument ;
  - avant de creer un pod, le script affiche GPU, prix horaire, duree maximale et cout maximal,
    puis demande de taper "oui" (--yes saute la question : a ne passer qu'apres accord explicite) ;
  - chaque etape distante a un delai ; quoi qu'il arrive (erreur, Ctrl+C, depassement), le pod est
    SUPPRIME dans un bloc finally, puis la liste des pods restants est affichee.

Fonctionnement : le code du depot (sans data/, outputs/, weights/) part en archive par scp, le pod
installe et telecharge lui-meme les donnees publiques (--setup), lance --command, puis les dossiers
--fetch reviennent par scp. Il faut une cle SSH locale (~/.ssh/id_ed25519.pub ou id_rsa.pub) :
sa partie publique est donnee au pod (variable PUBLIC_KEY des images RunPod).
"""

# Pod : une machine louee chez RunPod (un conteneur avec un GPU), facturee a l'heure tant qu'elle existe.
# SSH / scp : programmes pour se connecter a une machine distante et y copier des fichiers.
from __future__ import annotations  # permet d'ecrire les types "float | None" sans les evaluer a l'execution

import argparse  # lecture des options de la ligne de commande
import logging  # journal de messages avec l'heure
import os  # variables d'environnement (RUNPOD_API_KEY)
import shlex  # protege un texte pour le passer sans risque a un terminal distant (guillemets, espaces)
import subprocess  # lance d'autres programmes (ssh, scp) depuis Python
import sys  # sys.exit : renvoie un code de sortie au terminal
import tarfile  # creation et lecture d'archives .tar.gz
import tempfile  # dossiers temporaires, effaces automatiquement
import time  # horloge : delais et durees
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

logger = logging.getLogger("run_remote")  # journal nomme d'apres le script

REPO = Path(__file__).resolve().parents[2]  # racine du depot (deux dossiers au-dessus de tools/runpod/)
EXCLUDED = {  # noms de dossiers jamais envoyes au pod (lourds, inutiles ou prives)
    "data",  # donnees : le pod les telecharge lui-meme
    "outputs",  # resultats d'experiences
    "weights",  # poids pre-entraines
    "runs",  # sorties d'Ultralytics
    ".git",  # historique git
    ".venv",  # environnement Python local
    "venv",  # autre nom courant d'environnement Python
    "__pycache__",  # fichiers Python compiles
    ".pytest_cache",  # cache des tests
    ".ruff_cache",  # cache de ruff
    ".ipynb_checkpoints",  # sauvegardes automatiques de Jupyter
}
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"  # image Docker du pod : PyTorch + CUDA deja installes
REMOTE_DIR = "/workspace/vision-toolbox"  # dossier du code sur le pod


# --- cle et API ----------------------------------------------------------------------------------


def load_api_key(env_file: Path = REPO / ".env") -> str:  # lit la cle secrete de l'API RunPod ; la renvoie
    """RUNPOD_API_KEY depuis l'environnement, sinon depuis .env (lignes CLE=valeur). Jamais affichee."""
    key = os.environ.get("RUNPOD_API_KEY", "").strip()  # d'abord la variable d'environnement ("" si absente)
    if not key and env_file.exists():  # si elle est absente et qu'un fichier .env existe
        for line in env_file.read_text(encoding="utf-8").splitlines():  # pour chaque ligne du fichier
            name, _, value = line.partition("=")  # coupe "NOM=valeur" au premier "="
            if name.strip() == "RUNPOD_API_KEY":  # si c'est la bonne ligne
                key = value.strip().strip('"').strip("'")  # garde la valeur, sans espaces ni guillemets autour
    if not key:  # toujours pas de cle
        # quitte le programme avec un message qui explique ou definir la cle
        raise SystemExit("RUNPOD_API_KEY absente : la definir dans l'environnement ou dans .env (jamais dans un fichier versionne)")
    return key  # renvoie la cle (jamais affichee)


def api():  # prepare la bibliotheque runpod avec la cle ; renvoie le module runpod
    import runpod  # bibliotheque officielle de RunPod (importee ici : seulement si on en a besoin)

    runpod.api_key = load_api_key()  # donne la cle a la bibliotheque
    return runpod  # renvoie le module pret a l'emploi


def hourly_price(runpod, gpu_type: str, cloud: str) -> float | None:  # renvoie le prix horaire du GPU, ou None si inconnu
    """Prix horaire a la demande (securise ou communautaire) d'un type de GPU, None si inconnu."""
    info = runpod.get_gpu(gpu_type) or {}  # fiche du GPU demandee a RunPod ({} si rien)
    key = "securePrice" if cloud == "SECURE" else "communityPrice"  # champ du prix selon le type de cloud
    price = info.get(key) or (info.get("lowestPrice") or {}).get("uninterruptablePrice")  # ce prix, sinon le prix plancher
    return float(price) if price else None  # en nombre decimal, ou None


def list_pods(runpod) -> list[dict]:  # affiche les pods existants ; renvoie leur liste
    pods = runpod.get_pods() or []  # liste des pods du compte
    if not pods:  # si aucun pod
        logger.info("aucun pod : rien n'est facture")  # on rassure
    for pod in pods:  # pour chaque pod
        # identifiant, nom, etat voulu et cout horaire
        logger.info("pod %s | %s | %s | %s $/h", pod.get("id"), pod.get("name"), pod.get("desiredStatus"), pod.get("costPerHr", "?"))
    return pods  # renvoie la liste


# --- archive du code -----------------------------------------------------------------------------


def make_archive(target: Path) -> Path:  # cree l'archive du code a envoyer ; renvoie son chemin
    """tar.gz du depot sans donnees, sorties, poids ni caches (quelques Mo) ; .env n'est JAMAIS inclus."""

    def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:  # filtre : renvoie le fichier a garder, ou None pour l'exclure
        parts = Path(info.name).parts[1:]  # morceaux du chemin dans l'archive, sans le dossier racine "vision-toolbox"
        if any(p in EXCLUDED for p in parts) or Path(info.name).name.startswith(".env"):  # dossier exclu ou fichier secret .env
            return None  # on ne l'ajoute pas
        return info  # sinon on le garde

    with tarfile.open(target, "w:gz") as tar:  # ouvre une archive compressee en ecriture, fermee a la fin du bloc
        tar.add(REPO, arcname="vision-toolbox", filter=keep)  # ajoute tout le depot, en passant chaque fichier par le filtre
    return target  # renvoie le chemin de l'archive


# --- SSH -----------------------------------------------------------------------------------------


def public_key() -> str:  # renvoie la cle SSH publique de l'utilisateur (texte)
    for name in ("id_ed25519.pub", "id_rsa.pub"):  # noms de fichiers de cle publique les plus courants
        path = Path.home() / ".ssh" / name  # dans le dossier ~/.ssh
        if path.exists():  # si ce fichier existe
            return path.read_text(encoding="utf-8").strip()  # renvoie son contenu (la partie publique, sans danger)
    raise SystemExit("aucune cle SSH publique (~/.ssh/id_ed25519.pub) : la creer avec ssh-keygen -t ed25519")  # aucune cle trouvee


def ssh_endpoint(pod: dict) -> tuple[str, int] | None:  # renvoie (adresse IP, port) pour SSH, ou None si pas encore pret
    """(ip, port) public du port 22 du pod, une fois demarre."""
    runtime = pod.get("runtime") or {}  # informations d'execution (vides tant que le pod demarre)
    for port in runtime.get("ports") or []:  # pour chaque port ouvert du pod
        if port.get("privatePort") == 22 and port.get("isIpPublic"):  # le port 22 (SSH) expose sur une IP publique
            return port["ip"], int(port["publicPort"])  # adresse et numero de port vus de l'exterieur
    return None  # pas encore disponible


def ssh_base(ip: str, port: int) -> list[str]:  # debut de la commande ssh vers le pod ; renvoie la liste des morceaux
    return [  # liste des morceaux de la commande (subprocess les recoit separes)
        "ssh",  # le programme ssh
        "-p",  # option : numero de port...
        str(port),  # ...celui du pod
        "-o",  # option de configuration...
        "StrictHostKeyChecking=no",  # ...accepter la machine sans confirmation (pod neuf a chaque fois)
        "-o",  # option de configuration...
        "UserKnownHostsFile=/dev/null",  # ...ne pas memoriser la machine (son identite change a chaque pod)
        "-o",  # option de configuration...
        "ServerAliveInterval=30",  # ...envoyer un signe de vie toutes les 30 s pour garder la connexion ouverte
        f"root@{ip}",  # utilisateur root sur l'adresse du pod
    ]


def run_checked(command: list[str], timeout: float, what: str) -> None:  # lance une commande avec un delai maximum ; ne renvoie rien
    if timeout <= 0:  # s'il ne reste plus de temps
        raise TimeoutError(f"duree maximale atteinte avant : {what}")  # on arrete (le pod sera supprime par le finally)
    logger.info("%s (delai %.0f s)", what, timeout)  # annonce l'etape et son delai
    subprocess.run(command, check=True, timeout=timeout)  # lance ; erreur si la commande echoue ou depasse le delai


# --- execution complete --------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:  # cree le pod, execute, rapatrie, supprime ; renvoie 0 si succes, 1 si annule
    deadline = time.monotonic() + args.max_minutes * 60  # instant limite (horloge qui ne recule jamais)
    remaining = lambda: deadline - time.monotonic()  # noqa: E731 - secondes restantes avant la limite
    plan = {  # resume de ce qui va etre fait (affiche en simulation)
        "gpu": args.gpu,  # type de GPU
        "cloud": args.cloud,  # cloud communautaire ou securise
        "image": args.image,  # image Docker
        "max_minutes": args.max_minutes,  # duree maximale
        "setup": args.setup,  # commandes d'installation
        "command": args.command,  # commande principale
        "fetch": args.fetch,  # dossiers a rapatrier
    }
    if args.dry_run:  # mode simulation : rien n'est cree ni facture
        logger.info("simulation (--dry-run), rien n'est cree : %s", plan)  # affiche le plan
        archive = make_archive(Path(tempfile.gettempdir()) / "vtb_code_dry_run.tar.gz")  # fabrique quand meme l'archive pour verifier
        logger.info("archive du code : %s (%.1f Mo)", archive, archive.stat().st_size / 1e6)  # chemin et taille de l'archive
        return 0  # fin de la simulation

    runpod = api()  # bibliotheque runpod prete (avec la cle)
    price = hourly_price(runpod, args.gpu, args.cloud)  # prix horaire du GPU choisi
    cost = f"{price * args.max_minutes / 60:.2f} $" if price else "inconnu"  # cout maximal = prix horaire x duree maximale
    print(f"\nGPU : {args.gpu} ({args.cloud})\nPrix horaire : {price if price else 'inconnu'} $/h")  # affiche le GPU et son prix
    print(f"Duree maximale : {args.max_minutes} min\nCout maximal : {cost}\nCommande : {args.command}\n")  # duree, cout, commande
    if not args.yes and input('Taper "oui" pour creer le pod : ').strip().lower() != "oui":  # sans --yes, il faut taper "oui"
        logger.info("annule : aucun pod cree")  # autre reponse : on annule
        return 1  # code 1 = annule

    pod_id = None  # identifiant du pod (None tant qu'il n'est pas cree)
    started = time.monotonic()  # instant de creation, pour mesurer la duree facturee
    try:  # le bloc finally supprimera le pod quoi qu'il arrive
        pod = runpod.create_pod(  # demande la creation du pod
            name=f"vtb-{int(time.time())}",  # nom unique (horodatage)
            image_name=args.image,  # image Docker
            gpu_type_id=args.gpu,  # type de GPU
            cloud_type=args.cloud,  # communautaire ou securise
            support_public_ip=True,  # IP publique, pour s'y connecter en SSH
            start_ssh=True,  # demarre le serveur SSH
            container_disk_in_gb=args.disk_gb,  # taille du disque
            ports="22/tcp",  # ouvre le port 22 (SSH)
            env={"PUBLIC_KEY": public_key()},  # donne notre cle publique au pod pour qu'il nous laisse entrer
        )
        pod_id = pod["id"]  # retient l'identifiant : desormais le finally saura quoi supprimer
        logger.info("pod %s cree ; attente du demarrage", pod_id)  # annonce
        endpoint = None  # adresse SSH (None tant que le pod n'est pas pret)
        while endpoint is None:  # tant que le pod n'est pas pret
            if remaining() <= 0 or time.monotonic() - started > args.start_timeout:  # si on a trop attendu
                raise TimeoutError("le pod n'a pas demarre a temps")  # on abandonne (le pod sera supprime)
            time.sleep(10)  # attend 10 s avant de redemander
            endpoint = ssh_endpoint(runpod.get_pod(pod_id) or {})  # redemande l'etat du pod et cherche le port SSH
        ip, port = endpoint  # adresse et port SSH du pod
        base = ssh_base(ip, port)  # debut de la commande ssh vers ce pod
        time.sleep(15)  # sshd demarre quelques secondes apres l'ouverture du port
        with tempfile.TemporaryDirectory() as tmp:  # dossier temporaire, efface a la fin du bloc
            archive = make_archive(Path(tmp) / "code.tar.gz")  # archive du code
            run_checked(  # envoie l'archive au pod avec scp
                [  # commande scp, morceau par morceau
                    "scp",  # programme de copie par SSH
                    "-P",  # option : numero de port (P majuscule pour scp)...
                    str(port),  # ...celui du pod
                    "-o",  # option de configuration...
                    "StrictHostKeyChecking=no",  # ...accepter la machine sans confirmation
                    "-o",  # option de configuration...
                    "UserKnownHostsFile=/dev/null",  # ...ne pas memoriser la machine
                    str(archive),  # fichier local a envoyer
                    f"root@{ip}:/workspace/code.tar.gz",  # destination sur le pod
                ],
                min(remaining(), 600),  # delai : au plus 10 min, et pas plus que le temps restant
                "envoi du code",  # nom de l'etape pour le journal
            )
        # commande distante : deballe le code, installe (setup ; "true" = ne rien faire), puis lance la commande
        remote = f"cd /workspace && tar xzf code.tar.gz && cd {REMOTE_DIR} && ({args.setup or 'true'}) && ({args.command})"
        run_checked([*base, f"bash -lc {shlex.quote(remote)}"], remaining() - 120, "execution distante")  # 2 min gardees pour rapatrier
        for folder in args.fetch:  # pour chaque dossier de resultats a rapatrier
            local = REPO / folder  # ou il arrivera en local
            local.parent.mkdir(parents=True, exist_ok=True)  # cree son dossier parent
            pack = f"cd {REMOTE_DIR} && tar czf /workspace/fetch.tar.gz {shlex.quote(folder)}"  # commande distante : archive le dossier
            run_checked([*base, f"bash -lc {shlex.quote(pack)}"], min(remaining(), 300), f"archive de {folder}")  # la lance (5 min max)
            with tempfile.TemporaryDirectory() as tmp:  # dossier temporaire local
                target = Path(tmp) / "fetch.tar.gz"  # ou recevoir l'archive
                run_checked(  # recupere l'archive avec scp
                    [  # commande scp, morceau par morceau
                        "scp",  # programme de copie par SSH
                        "-P",  # option : numero de port...
                        str(port),  # ...celui du pod
                        "-o",  # option de configuration...
                        "StrictHostKeyChecking=no",  # ...accepter la machine sans confirmation
                        "-o",  # option de configuration...
                        "UserKnownHostsFile=/dev/null",  # ...ne pas memoriser la machine
                        f"root@{ip}:/workspace/fetch.tar.gz",  # fichier distant a recuperer
                        str(target),  # destination locale
                    ],
                    min(remaining(), 600),  # delai : au plus 10 min
                    f"rapatriement de {folder}",  # nom de l'etape pour le journal
                )
                with tarfile.open(target) as tar:  # ouvre l'archive recue
                    tar.extractall(REPO, filter="data")  # la deballe dans le depot ; filter="data" refuse les chemins dangereux
            logger.info("rapatrie : %s", local)  # annonce
        return 0  # succes
    finally:  # quoi qu'il arrive (succes, erreur, Ctrl+C, delai depasse)
        if pod_id is not None:  # si un pod a ete cree
            try:  # la suppression elle-meme peut echouer (reseau...)
                runpod.terminate_pod(pod_id)  # supprime le pod : la facturation s'arrete
                logger.info("pod %s SUPPRIME apres %.1f min", pod_id, (time.monotonic() - started) / 60)  # duree de location
            except Exception as error:  # noqa: BLE001 - on veut absolument voir l'echec de suppression
                # message tres visible : il faut alors supprimer le pod a la main
                logger.error("ECHEC de suppression du pod %s : %s -> le supprimer a la main sur runpod.io", pod_id, error)
        list_pods(runpod)  # affiche les pods restants (verification qu'il n'en reste aucun)


def main() -> int:  # lit les options et lance l'action demandee ; renvoie le code de sortie
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")  # messages avec l'heure
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # aide = premiere ligne de la docstring
    parser.add_argument("--list", action="store_true", help="liste les pods existants")  # lister les pods
    parser.add_argument("--gpus", action="store_true", help="liste les types de GPU")  # lister les GPU
    parser.add_argument("--terminate", metavar="POD_ID")  # supprimer un pod par son identifiant
    sub = parser.add_subparsers(dest="action")  # sous-commandes (une seule : run)
    r = sub.add_parser("run", help="cree un pod, execute, rapatrie, supprime")  # sous-commande run
    r.add_argument("--gpu", required=True, help='identifiant RunPod, ex. "NVIDIA RTX A5000" (voir --gpus)')  # type de GPU
    r.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])  # communautaire (moins cher) ou securise
    r.add_argument("--image", default=DEFAULT_IMAGE)  # image Docker
    r.add_argument("--max-minutes", type=float, required=True, help="duree maximale totale, pod supprime au-dela")  # garde-fou de cout
    r.add_argument("--start-timeout", type=float, default=600, help="secondes pour que le pod demarre")  # attente maximale au demarrage
    r.add_argument("--disk-gb", type=int, default=40)  # taille du disque en Go
    r.add_argument("--setup", default="pip install -e .", help="commandes d'installation sur le pod")  # installation
    r.add_argument("--command", required=True, help="commande a executer depuis la racine du depot")  # commande principale
    r.add_argument("--fetch", nargs="*", default=[], help="dossiers a rapatrier (chemins relatifs au depot)")  # resultats a rapatrier
    r.add_argument("--yes", action="store_true", help="ne pas demander confirmation (seulement apres accord explicite)")  # sans question
    r.add_argument("--dry-run", action="store_true")  # simulation
    args = parser.parse_args()  # lit les options tapees
    if args.action == "run":  # si la sous-commande run est demandee
        return run(args)  # lance l'experience complete
    runpod = api()  # sinon : actions simples sur le compte
    if args.gpus:  # --gpus : liste des GPU
        for gpu in sorted(runpod.get_gpus(), key=lambda g: g.get("memoryInGb", 0)):  # du moins de memoire au plus
            logger.info("%-40s %3s Go", gpu.get("id"), gpu.get("memoryInGb"))  # nom et memoire du GPU
    if args.terminate:  # --terminate : suppression d'un pod
        runpod.terminate_pod(args.terminate)  # supprime le pod
        logger.info("pod %s supprime", args.terminate)  # annonce
    if args.list or args.terminate:  # --list, ou apres une suppression
        list_pods(runpod)  # affiche les pods restants
    return 0  # code 0 = succes


if __name__ == "__main__":  # vrai seulement quand le script est lance directement
    sys.exit(main())  # lance main et transmet son code de sortie au terminal
