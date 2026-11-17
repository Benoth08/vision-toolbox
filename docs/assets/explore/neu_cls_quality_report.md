# Controle du jeu : data/raw/neu_cls_flat

## Alertes

- 1 doublon(s) a cheval sur deux sous-ensembles : FUITE, decouper par groupe
- 47 image(s) atypique(s) a regarder (voir outliers.csv)

## Chiffres

- images : 1800 (lisibles : 1800)
- classes : {'crazing': 300, 'inclusion': 300, 'patches': 300, 'pitted': 300, 'rolled': 300, 'scratches': 300}
- tailles les plus frequentes : {'200x200': 1800}
- paires de doublons : 2 (exacts : 1)

## Raccourcis possibles (une statistique seule predit-elle la classe ?)

| statistic          |   balanced_accuracy |   chance |
|:-------------------|--------------------:|---------:|
| sharpness          |               0.585 |    0.167 |
| std                |               0.506 |    0.167 |
| mean               |               0.404 |    0.167 |
| saturated_fraction |               0.352 |    0.167 |
| dark_fraction      |               0.224 |    0.167 |
