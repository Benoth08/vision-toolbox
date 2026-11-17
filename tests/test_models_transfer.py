"""Tests du zoo et des strategies de transfert, sans telecharger de poids (pretrained=False)."""

# Apprentissage par transfert : on part d'un reseau deja entraine sur un grand jeu (ImageNet), on remplace sa
# derniere couche (la "tete", qui donne les classes) et on choisit quelles parties continuent d'apprendre.
# Geler un bloc = ses poids ne bougent plus (requires_grad=False : PyTorch ne calcule pas son gradient).

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

import numpy as np  # numpy : calcul sur des tableaux de nombres
import pytest  # pytest : lance les tests ; parametrize, approx, raises...
import torch  # PyTorch : tenseurs (tableaux de nombres) et reseaux de neurones
import torch.nn as nn  # nn : briques de reseaux (nn.Module est la classe de base d'un modele)

from vtb.models.transfer import (  # outils de construction et de transfert des modeles de classification
    ModelConfig,  # configuration du modele (nom, poids pre-entraines, nombre de canaux d'entree...)
    ProgressiveUnfreezer,  # degele les blocs un par un au fil des epoques
    TransferConfig,  # configuration de la strategie de transfert (quoi geler, taux d'apprentissage par bloc)
    apply_strategy,  # gele / degele les parametres selon la strategie
    build_classifier,  # construit le modele avec une nouvelle tete ; renvoie (modele, description de ses blocs)
    freeze_frozen_batchnorm,  # garde en mode evaluation les BatchNorm des blocs geles
    param_groups,  # groupes de parametres avec chacun son taux d'apprentissage
)
from vtb.models.zoo import ZOO, get_module, get_spec  # catalogue des modeles, acces a un sous-module par son nom, description
from vtb.tasks.classification.features import fit_logistic_head  # entraine une regression logistique et la copie dans la tete


# renvoie le nom des blocs de premier niveau (fc, layer3...) qui ont au moins un parametre entrainable
def _trainable_prefixes(model: nn.Module) -> set[str]:
    # named_parameters donne ("layer4.1.conv2.weight", parametre) ; on garde le debut du nom si le parametre apprend
    return {name.split(".")[0] for name, p in model.named_parameters() if p.requires_grad}


@pytest.mark.parametrize("name", sorted(ZOO))  # le test tourne pour chaque modele du catalogue
# Verifie que chaque modele recoit une tete au bon nombre de classes, accepte des images grises, et que les blocs declares
# existent. Attraperait : une description de modele fausse (tete ou premiere couche introuvable) ou une sortie mal formee.
def test_every_zoo_model_gets_a_new_head_and_a_gray_input(name: str):
    # modele sans poids telecharges, 1 canal d'entree (gris), 4 classes
    model, spec = build_classifier(ModelConfig(name=name, pretrained=False, in_channels=1), num_classes=4)
    assert get_module(model, spec.head).out_features == 4  # la tete sort 4 valeurs (une par classe)
    assert get_module(model, spec.first_conv).in_channels == 1  # la premiere convolution attend 1 canal
    for block in spec.blocks:  # tous les blocs declares existent vraiment
        get_module(model, block)  # leve une erreur si le nom ne correspond a aucun sous-module
    with torch.no_grad():  # pas de calcul de gradient : on ne fait que predire
        # eval() : mode prediction ; 2 images grises 64 x 64 -> 2 lignes de 4 scores
        assert model.eval()(torch.zeros(2, 1, 64, 64)).shape == (2, 4)


# Verifie que l'adaptation au gris (somme des poids des 3 canaux) equivaut a donner l'image grise recopiee en RGB.
# Attraperait : une adaptation qui perdrait le benefice des poids pre-entraines.
def test_gray_adaptation_matches_rgb_replication():
    """Une image grise dans la conv adaptee donne la meme sortie que l'image recopiee sur 3 canaux dans l'originale."""
    rgb, spec = build_classifier(ModelConfig(name="resnet18", pretrained=False), 3)  # ResNet18 couleur, 3 classes
    gray, _ = build_classifier(ModelConfig(name="resnet18", pretrained=False, in_channels=1), 3)  # meme modele, entree grise
    # copie les poids du modele couleur ; pour la 1re convolution, somme des poids des 3 canaux en un seul
    gray.load_state_dict({k: (v.sum(1, keepdim=True) if k == "conv1.weight" else v) for k, v in rgb.state_dict().items()})
    x = torch.rand(2, 1, 32, 32)  # 2 images grises aleatoires
    with torch.no_grad():  # pas de gradient
        # modele couleur sur l'image recopiee 3 fois == modele gris sur l'image grise (a 1e-5 pres)
        assert torch.allclose(rgb.eval()(x.repeat(1, 3, 1, 1)), gray.eval()(x), atol=1e-5)


# Verifie ce que chaque strategie laisse apprendre. Attraperait : un bloc cense etre gele qui apprend quand meme
# (il "oublie" ce qu'il savait) ou un bloc cense apprendre qui reste fige.
def test_strategies_freeze_what_they_should():
    model, spec = build_classifier(ModelConfig(name="resnet18", pretrained=False), 3)  # ResNet18, 3 classes
    apply_strategy(model, spec, TransferConfig(strategy="head"))  # "head" : seule la tete apprend
    assert _trainable_prefixes(model) == {"fc"}  # fc = la tete du ResNet
    apply_strategy(model, spec, TransferConfig(strategy="partial", unfreeze_last=2))  # "partial" : tete + 2 derniers blocs
    assert _trainable_prefixes(model) == {"fc", "layer3", "layer4"}  # les 2 derniers blocs sont layer3 et layer4
    apply_strategy(model, spec, TransferConfig(strategy="features"))  # "features" : tout gele (la tete est ajustee a part)
    assert _trainable_prefixes(model) == set()  # aucun parametre n'apprend par descente de gradient
    apply_strategy(model, spec, TransferConfig(strategy="full"))  # "full" : tout le reseau apprend
    assert all(p.requires_grad for p in model.parameters())  # tous les parametres apprennent


# Verifie les taux d'apprentissage par bloc : plus un bloc est profond (proche de l'entree), plus son taux est petit.
# Attraperait : des blocs pre-entraines modifies trop fort, ce qui detruit ce qu'ils avaient appris.
def test_param_groups_give_lower_rates_to_deeper_blocks():
    model, spec = build_classifier(ModelConfig(name="resnet18", pretrained=False), 3)  # ResNet18, 3 classes
    # tete + 2 derniers blocs ; corps a 0.1 x le taux de la tete ; chaque bloc plus profond encore x 0.5
    transfer = TransferConfig(strategy="partial", unfreeze_last=2, backbone_lr_factor=0.1, layer_decay=0.5)
    apply_strategy(model, spec, transfer)  # applique le gel
    groups = {g["name"]: g["lr"] for g in param_groups(model, spec, 1e-3, transfer)}  # nom du groupe -> taux, base 0.001
    # tete 1e-3 ; layer4 = 1e-3 x 0.1 = 1e-4 ; layer3 = 1e-4 x 0.5 = 5e-5
    assert groups == pytest.approx({"head": 1e-3, "layer4": 1e-4, "layer3": 5e-5})


# Verifie le degel progressif : un bloc de plus toutes les 2 epoques, en partant de la tete.
# Attraperait : un degel trop tot, trop tard, ou dans le mauvais ordre.
def test_progressive_unfreezes_one_block_every_k_epochs():
    model, spec = build_classifier(ModelConfig(name="resnet18", pretrained=False), 3)  # ResNet18, 3 classes
    # strategie "progressive" : un bloc degele toutes les 2 epoques
    unfreezer = ProgressiveUnfreezer(model, spec, TransferConfig(strategy="progressive", progressive_every=2))
    changes = [unfreezer.update(epoch) for epoch in range(1, 8)]  # appelle update aux epoques 1 a 7 ; True = un bloc degele
    assert changes == [False, False, True, False, True, False, True]  # epoques 3, 5, 7
    assert _trainable_prefixes(model) == {"fc", "layer2", "layer3", "layer4"}  # 3 blocs degeles, du plus haut au plus bas


# Verifie que les BatchNorm des blocs geles restent en mode evaluation meme quand le modele passe en entrainement.
# BatchNorm : couche qui normalise avec des moyennes calculees sur les lots ; en mode train, elle met a jour ces moyennes.
# Attraperait : un bloc "gele" dont les statistiques derivent quand meme, ce qui degrade le modele pre-entraine.
def test_frozen_batchnorm_stays_in_eval_mode():
    model, spec = build_classifier(ModelConfig(name="resnet18", pretrained=False), 3)  # ResNet18, 3 classes
    apply_strategy(model, spec, TransferConfig(strategy="head"))  # tout gele sauf la tete
    model.train()  # passe tout le modele en mode entrainement (comme au debut de chaque epoque)
    freeze_frozen_batchnorm(model)  # remet en mode evaluation les BatchNorm geles
    assert not model.bn1.training and not model.layer4[1].bn2.training  # deux BatchNorm geles sont bien en mode evaluation


# Verifie qu'un modele cree de zero (sans pre-entrainement) apprend entierement, quelle que soit la strategie.
# Attraperait : un petit modele maison gele par erreur, qui n'apprendrait presque rien.
def test_custom_models_train_entirely():
    model, spec = build_classifier(ModelConfig(name="small_resnet", extra={"channels": [8, 16]}), 2)  # petit ResNet maison, 2 classes
    apply_strategy(model, spec, TransferConfig(strategy="head"))  # ignore : un modele de zero apprend tout
    assert all(p.requires_grad for p in model.parameters())  # tous les parametres apprennent


# Verifie que la regression logistique (avec sa normalisation) est recopiee exactement dans la couche lineaire de tete.
# Attraperait : une tete exportee qui ne donne pas les memes predictions que le classifieur entraine.
def test_logistic_head_is_folded_into_an_equivalent_linear_layer(rng):  # rng : generateur aleatoire de conftest.py
    from sklearn.preprocessing import StandardScaler  # normalisation : retire la moyenne et divise par l'ecart-type

    model, spec = build_classifier(ModelConfig(name="simple_cnn", extra={"channels": [4, 8]}), 3)  # petit CNN, 3 classes
    features = rng.normal(size=(60, 8)).astype(np.float32)  # 60 vecteurs de caracteristiques de taille 8 (sortie du corps)
    targets = np.repeat([0, 1, 2], 20)  # classes : 20 x 0, 20 x 1, 20 x 2
    features[targets == 1] += 2.0  # decale la classe 1 pour qu'elle soit separable
    clf = fit_logistic_head(model, spec, features, targets, c=1.0, seed=0)  # entraine la regression et la copie dans la tete
    head = get_module(model, spec.head)  # la couche lineaire de tete du modele
    with torch.no_grad():  # pas de gradient
        folded = head(torch.tensor(features)).argmax(1).numpy()  # classe predite par la tete (indice du plus grand score)
    expected = clf.predict(StandardScaler().fit(features).transform(features))  # classes predites par scikit-learn (donnees normalisees)
    assert (folded == expected).all()  # memes predictions pour les 60 exemples


# Verifie qu'un nom de modele inconnu donne un message clair. Attraperait : une faute de frappe qui produirait une
# erreur incomprehensible plus loin.
def test_unknown_model_name_is_a_clear_error():
    with pytest.raises(ValueError, match="catalogue"):  # on attend une ValueError dont le message parle du catalogue
        get_spec("resnet1800")  # nom qui n'existe pas
