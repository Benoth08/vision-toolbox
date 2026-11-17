| pipeline      |   f1_macro_mean |   f1_macro_std |   preprocess_ms_per_image |   seconds | steps                     |
|:--------------|----------------:|---------------:|--------------------------:|----------:|:--------------------------|
| none          |           0.960 |          0.014 |                     0.230 |    33.000 | (aucun)                   |
| shading       |           0.956 |          0.019 |                     0.700 |    33.000 | shading_division          |
| median_clahe  |           0.952 |          0.015 |                     0.760 |    31.900 | median_blur -> clahe      |
| shading_clahe |           0.950 |          0.020 |                     1.060 |    32.200 | shading_division -> clahe |
| stretch       |           0.948 |          0.017 |                     1.050 |    26.500 | percentile_stretch        |
| clahe         |           0.946 |          0.011 |                     0.370 |    29.100 | clahe                     |
| dog           |           0.939 |          0.020 |                     0.530 |    32.000 | difference_of_gaussians   |
