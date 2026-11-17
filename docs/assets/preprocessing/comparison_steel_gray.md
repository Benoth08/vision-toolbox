| pipeline      |   f1_macro_mean |   f1_macro_std |   preprocess_ms_per_image |   seconds | steps                     |
|:--------------|----------------:|---------------:|--------------------------:|----------:|:--------------------------|
| none          |           0.956 |          0.029 |                     0.170 |    23.200 | (aucun)                   |
| stretch       |           0.952 |          0.016 |                     0.980 |    30.400 | percentile_stretch        |
| clahe         |           0.950 |          0.018 |                     0.590 |    25.700 | clahe                     |
| shading       |           0.947 |          0.018 |                     0.810 |    18.500 | shading_division          |
| shading_clahe |           0.947 |          0.026 |                     1.300 |    26.500 | shading_division -> clahe |
| dog           |           0.946 |          0.016 |                     0.720 |    40.500 | difference_of_gaussians   |
| median_clahe  |           0.945 |          0.025 |                     0.670 |    21.900 | median_blur -> clahe      |
