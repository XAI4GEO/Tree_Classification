## Data used in the project

## Folder `all_cutouts`

Data in this folder are created by [step02_data_cleaning.ipynb](../step0_data_preparation_examples/step02_data_cleaning.ipynb).

Cleaned cutout tiles from three sources:

1. [Netflora](https://github.com/NetFlora/Netflora/tree/main/) prediction on Brazil Orthomosaic dataset (`Map1_Orthomosaic_export_SatJun10172428194829.tif`). Three predictions which are significantly different from each others are used: 
    - acai: label0_murumuru_embrapa00.zarr
    - buriti: label1_netflora_buriti_emprapa00.zarr
    - tucuma: label3_reforestree_banana.zarr

2. [Reforestree dataset](https://github.com/gyrrei/ReforesTree). Three classes are used:
    - banana: label3_reforestree_banana.zarr
    - cacao: label4_reforestree_cacao.zarr
    - fruit: label5_reforestree_fruit.zarr

3. Manually labelled data sources:
    - label6_mannual_palmtree.zarr

Following processing are applied to clean the data:

1. resample `Reforestree` data (1cm resolution) to the same resolution as `Netflora` data (6cm resolution). This is done by a 6x6 average window.
2. padding black border to make all cutouts with a size of 128x128
3. assign the following class labels:
    ```python
    description_labels = {0: 'label0_murumuru_embrapa00.zarr',
                      1: 'netflora_buriti_emprapa00', 
                      2: 'netflora_tucuma_emprapa00',
                      3: 'reforestree_banana',
                      4: 'reforestree_cacao',
                      5: 'reforestree_fruit',}
    ```
4. Save each class in as separate Zarr file

These cutouts will be used for:
1. Making pairs to train the Siamese network
2. benckmark datasets for few-shot learning
3. testing data for the trained model

## Folder `selected_cutouts`

Data in this folder are created by [step03_make_training_pairs](../step0_data_preparation_examples/step03_make_training_pairs). These are manually selected cutouts from `all_cutouts` folder. The selection is performed based on visual assessment of the similarity within the same class and the difference between different classes.