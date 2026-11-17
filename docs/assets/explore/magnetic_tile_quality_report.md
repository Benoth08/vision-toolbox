# Controle du jeu : data/raw/magnetic_tile

## Alertes

- 1225 tailles d'image differentes : prevoir un resize
- 4 doublon(s) avec deux etiquettes differentes : erreurs d'annotation probables
- 4 image(s) atypique(s) a regarder (voir outliers.csv)

## Chiffres

- images : 1344 (lisibles : 1344)
- classes : {'MT_Free': 952, 'MT_Blowhole': 115, 'MT_Uneven': 103, 'MT_Break': 85, 'MT_Crack': 57, 'MT_Fray': 32}
- tailles les plus frequentes : {'375x265': 6, '286x123': 5, '285x122': 4, '273x218': 3, '242x594': 3}
- paires de doublons : 301 (exacts : 0)

## Raccourcis possibles (une statistique seule predit-elle la classe ?)

| statistic          |   balanced_accuracy |   chance |
|:-------------------|--------------------:|---------:|
| height             |               0.222 |    0.167 |
| width              |               0.192 |    0.167 |
| mean               |               0.177 |    0.167 |
| std                |               0.167 |    0.167 |
| sharpness          |               0.167 |    0.167 |
| dark_fraction      |               0.167 |    0.167 |
| saturated_fraction |               0.166 |    0.167 |
