# Classification workflow

## Step 0 prepare datasets
In folder `step0_data_preparation_examples`, there are examples of how to make cutouts from different sources, clean the data, and make pairs for training the Siamese network. This step takes online open datasets and generates the `data` folder, which is the assumed input for the classification workflow.

## Step 1 classification
In the notebook `step1_classification.ipynb`, an example of classify an example image based on Siamese network and benchmark data with known classes is provided.

## Step 2 explainability
In the notebook `step2_explaination.ipynb`, an example of how to explain the classification result is provided.

## Step 3 accuracy assessment
In the notebook `step3_accuracy_evaluation.ipynb`, an example of assessing the accuracy of the classification result is provided. We performed classification for all the benchmark data and assess the accuracy of each benchmark class.