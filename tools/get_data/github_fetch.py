"""Telechargement sans git de quelques dossiers d'un depot GitHub, a un commit fige.

Les scripts get_*.py font un clone partiel avec git (sparse checkout). Sur un poste sans git (cas
frequent sous Windows), ils passent par ici : l'API GitHub donne l'arbre du commit (quelques appels
par dossier), chaque fichier est lu sur raw.githubusercontent.com puis verifie par son empreinte git
(SHA-1 du blob). On obtient donc les memes fichiers, au meme commit, que par le clone partiel.

Les fichiers arrivent d'abord dans <target>/_partiel/ et ne sont deplaces a leur place qu'une fois
tous verifies : un telechargement interrompu ne laisse jamais un dossier incomplet que le script
prendrait ensuite pour "deja telecharge" (la reprise ne relit que les fichiers manquants).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import ssl
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

TREE_URL = "https://api.github.com/repos/{owner}/{repo}/git/trees/{sha}"
RAW_URL = "https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{path}"
HEADERS = {"User-Agent": "vision-toolbox-get-data"}
# Un seul contexte TLS pour tous les fichiers : sans lui, urllib en recree un par requete et relit
# chaque fois le magasin de certificats du systeme (couteux sous Windows, des milliers de fichiers ici).
TLS_CONTEXT = ssl.create_default_context()


def _get(url: str, attempts: int = 4) -> bytes:
    """GET avec reprises : une connexion lente ou une coupure passagere ne doit pas tout arreter."""
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=120, context=TLS_CONTEXT) as response:
                return response.read()
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(url)


def _tree(owner: str, repo: str, sha: str, recursive: bool) -> list[dict]:
    url = TREE_URL.format(owner=owner, repo=repo, sha=sha) + ("?recursive=1" if recursive else "")
    data = json.loads(_get(url))
    if data.get("truncated"):
        raise RuntimeError(f"arbre tronque par l'API GitHub : {url}")
    return data["tree"]


def _folder_sha(owner: str, repo: str, commit: str, folder: str) -> str:
    """Descend dossier par dossier depuis la racine : l'arbre recursif du depot entier peut etre tronque."""
    sha = commit
    for part in PurePosixPath(folder).parts:
        entry = next((e for e in _tree(owner, repo, sha, recursive=False) if e["path"] == part), None)
        if entry is None or entry["type"] != "tree":
            raise FileNotFoundError(f"dossier {folder!r} absent du commit {commit}")
        sha = entry["sha"]
    return sha


def git_blob_sha(data: bytes) -> str:
    """L'empreinte que git donne a un fichier : SHA-1 de 'blob <taille>\\0' + contenu."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch_folders(repo_url: str, commit: str, folders: list[str], target: Path, workers: int = 8) -> int:
    """Telecharge `folders` (chemins du depot) au `commit` dans `target` ; renvoie le nombre de fichiers."""
    owner, repo = urllib.parse.urlparse(repo_url).path.strip("/").removesuffix(".git").split("/")
    folders = [folder.strip("/") for folder in folders]
    files: list[tuple[str, str]] = []
    for folder in folders:
        tree = _tree(owner, repo, _folder_sha(owner, repo, commit, folder), recursive=True)
        files += [(f"{folder}/{entry['path']}", entry["sha"]) for entry in tree if entry["type"] == "blob"]
    print(f"git absent : {len(files)} fichiers de {owner}/{repo} au commit {commit[:7]} (API GitHub + empreintes git)")
    partial = target / "_partiel"

    def fetch(item: tuple[str, str]) -> None:
        path, blob_sha = item
        out = partial / path
        if out.exists() and git_blob_sha(out.read_bytes()) == blob_sha:
            return
        data = _get(RAW_URL.format(owner=owner, repo=repo, commit=commit, path=urllib.parse.quote(path)))
        if git_blob_sha(data) != blob_sha:
            raise ValueError(f"empreinte git differente pour {path} (fichier Git LFS ou corrompu)")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)

    with ThreadPoolExecutor(workers) as pool:
        for done, _ in enumerate(pool.map(fetch, files), start=1):
            if done % 500 == 0 or done == len(files):
                print(f"  {done}/{len(files)} fichiers")
    for folder in folders:
        final = target / folder
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(final, ignore_errors=True)
        (partial / folder).rename(final)
    shutil.rmtree(partial, ignore_errors=True)
    return len(files)
