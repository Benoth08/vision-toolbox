# Tache : regenerer le site de documentation et le publier

Contexte : `site/build_site.py` produit un site statique (une page HTML autonome par article,
images incluses, recherche hors ligne, aucune requete externe) a partir des articles de `docs/`,
des README des recettes et du cours. Lis `site/deploy/DEPLOY.md`.

## Etapes

1. `bash site/build_site.sh --with-docs` (ou l'equivalent PowerShell de `site/deploy/publish.ps1`).
2. Ouvrir `out/site/index.html` en local (file://) : navigation, recherche, telechargements.
3. `python site/test_offline.py` : aucune ressource externe.
4. Publication sur l'hebergement choisi (GitHub Pages, GitLab Pages ou Read the Docs) selon
   `DEPLOY.md`. Depot public ou prive : me demander avant de publier quoi que ce soit.
5. Verifier depuis un navigateur que les pages et les zips se chargent.

## Livrable

L'adresse du site, la date de publication, la taille des zips, et la confirmation qu'aucune
donnee interne ni secret n'est dans le site.
