# prompts/ : taches pretes a confier a Claude Code (VS Code, mode agent)

Chaque fichier est une tache complete et autonome : la copier dans Claude Code, ouvert a la
racine du depot. Le fichier `CLAUDE.md` du depot est relu automatiquement et fixe les regles
(contrat des recettes, tests, ASCII, secrets, RunPod) ; les prompts ne les repetent pas toutes.

| ordre | fichier | machine | duree indicative |
|---|---|---|---|
| 1 | `01_setup_and_checks.md` | poste Windows (CPU) | 2 a 3 h (dont 20 notebooks) |
| 2 | `02_csharp_build_and_golden.md` | poste Windows avec SDK .NET 8 | 1 h |
| 3 | `03_gpu_benchmarks_runpod.md` | poste + GPU RunPod | 3 a 5 h, budget a fixer |
| 4 | `04_anomalib_full_benchmark.md` | GPU RunPod | 2 h |
| 5 | `05_annotation_tools_checks.md` | poste Windows avec ecran | 1 h |
| 6 | `06_new_dataset_recipe.md` | selon le jeu | variable |
| 7 | `07_extensions_ocr_tracking.md` | poste + GPU eventuel | variable |
| 8 | `08_site_publish.md` | poste | 30 min |

Regle commune : a la fin de chaque tache, Claude Code donne un compte rendu (ce qui a ete fait,
chiffres obtenus, ce qui reste), met a jour les README concernes avec les chiffres reels, et
laisse `pytest` et `ruff` au vert.
