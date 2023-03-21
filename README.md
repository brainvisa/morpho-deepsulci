# Deep learning methods for Morphologist sulci recognition

This repository contains the methods described in the following articles:

### [Borne L., Rivière D., Mancip M. and Mangin J.F., 2020. Automatic labeling of cortical sulci using patch-or CNN-based segmentation techniques combined with bottom-up geometric constraints. *Medical Image Analysis*](https://doi.org/10.1016/j.media.2020.101651)

This paper proposes and compares methods to automatically label the cortical folds.
The code developed for the UNET model is available [here](https://github.com/brainvisa/morpho-deepsulci/tree/master/python/deepsulci/sulci_labeling/method).

If you want to appply the model on your own dataset, the trained model is usable in the latest of Morphologist in [BrainVisa](https://brainvisa.info/).

### [Borne L., Rivière D., Cachia A., Roca P., Mellerio C., Oppenheim C. and Mangin J.F., 2021. Automatic recognition of specific local cortical folding patterns. *NeuroImage*](https://doi.org/10.1016/j.neuroimage.2021.118208)

The second paper proposes 3 methods to automatically classify local cortical folding patterns:
the first one based on a Support Vector Machine (SVM) classifier,
the second one based on Scoring by Non-local Image Patch Estimator (SNIPE)
and the third one based on a convolutionnal neural networks (Resnet).
The code developed for these 3 methods is available [here](https://github.com/brainvisa/morpho-deepsulci/tree/master/python/deepsulci/pattern_classification/method).
 
