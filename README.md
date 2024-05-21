## Improved GBDT Fitting to U-Net
An innovative causal analysis mechanism is proposed to explain the internal computational logic of U-Net. A "grey box" fitting model is constructed by introducing the Boosting integrator learning model and adopting the improved gradient boosting decision tree algorithm to globally fit the U-Net network. Since the structure of the Boosting integrator is not convenient for direct computational rule extraction, a method to extract the corresponding key rules from the Boosting integrator based on specific conditions is also proposed.

### Feature Extraction of U-Net Segmentation Results
1.Get the eigenvalues of the U-Net segmentation results on the four MRI sequences via **gainFeatures/get_features.py**.<br>
2.Normalise the feature extraction results from the first step of this section **gainFeatures/get_features.py** and store them in ***data/data.csv***.
<br>
3.Split ***data.csv*** by **dataSplit.py** to fit training data to U-Net ***/data/feature_train.csv*** and fitting test data ***/data/feature_test.csv***.
### Improved GBDT Fitting to U-Net
1.The features extracted from the U-Net segmentation results are fitted by **fittingTrain.py** to get the fitted model of U-Net ***/model/GBDT.pkl***.
2.Fit performance test of U-Net's fitted model ***model/GBDT.pkl*** by **testGBDT.py**

### Tree structure visualisation of fitted models for U-Net
Visualisation of the tree structure of the fitted model via **visualTree.py**
<img src="https://github.com/CodingVoyagers/BMFM_U-Net-fitting-model/blob/main/76_DTtree.png" />
### Tree structure visualisation of fitted models for U-Net
1.Causal rule extraction for GBDT-based U-Net's fitted model via **rulesExtraction.py**. The causal rule extraction results are stored in ***/RULE/rules3.pkl***.<br>
2.The performance of fitting the causal rules extracted in the first step of this section was tested by **testRules.py**.
