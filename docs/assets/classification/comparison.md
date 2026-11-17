| experience           | modele       | strategie   |   params_entraines |   meilleure_epoque |   temps_s |   test_accuracy |   test_f1_macro |   onnx_ms |
|:---------------------|:-------------|:------------|-------------------:|-------------------:|----------:|----------------:|----------------:|----------:|
| quick                | resnet18     | head        |               3078 |                  2 |      39.7 |           0.941 |           0.941 |      2.75 |
| resnet18_features    | resnet18     | features    |               3078 |                  1 |       9.8 |           0.978 |           0.978 |     10.56 |
| resnet18_full        | resnet18     | full        |           11179590 |                  7 |     190.9 |           1     |           1     |      3.98 |
| resnet18_gray        | resnet18     | partial     |            8396806 |                  9 |     121.5 |           1     |           1     |      3.98 |
| resnet18_head        | resnet18     | head        |               3078 |                  8 |     131.1 |           0.959 |           0.959 |      4.13 |
| resnet18_partial     | resnet18     | partial     |            8396806 |                  7 |     477.9 |           1     |           1     |     20.58 |
| resnet18_progressive | resnet18     | progressive |           11170054 |                  7 |     599.3 |           1     |           1     |      4.43 |
| simple_cnn           | simple_cnn   | partial     |             390438 |                 10 |     239.9 |           0.97  |           0.971 |      1.63 |
| small_resnet         | small_resnet | partial     |            1229030 |                 15 |    1420.5 |           0.952 |           0.951 |      6.4  |