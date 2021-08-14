# shadow-removal-with-image-decomposition

* ### structure of the Data Folder

The Data folder contains the sub-folder ISTD_Dataset , that contains two subfolders , the 'train' and
the 'test' folder.
Each of the subfolder of expression have three subfolders, of suffix '_A' , '_B' and '_C':
the '_A' contains images with shadow
the '_B' contains images contains mask of shadow
the '_C' contains images without shadow.

It also contains the file BDRAR.pth, which is the shadow detector network
trained on another databse.

Remarks, I didn't read the article yet, but I suppose , that the the images with and without shadow,
were taken by fixed position of the camera, though, the images were taken on different moment of the day,
which implies a change on the colometry, between the image with shadow and without shadow, that is tackled
 by a linear regression (that would be described later)



Description of the model:
The model is in fact composed of 3 network:

* ### the shadow detector network
It computes a shadow mask from the input image using the model of the article **"Bidirectional Feature Pyramid Network with Recurrent Attention Residual Modules for Shadow Detection"**
which is composed of a:
* CNN network, that learns features to computes the shadow mask at each scale,
the shallow layers focusing on local shadow detail, and the deeper layers learning informations about
the whole semantic.
* two sequence of RAN (residual attentionnal networks), each sequence starting from the first CNN layer 
  (or the last layer),  and merging succesively with the subsequent CNN layer.
  The RAN uses (attention mechanism), that learns an "attention map",
  which help and enhancing and selecting the good features while merging/
  
* a final block that relies again on attention mechanishm and that combines thet features map,
  which are created by the sequence of RAN.

The Resnext network , is used as the basic CNN for the  feature extraction,
and initialized with training on the imageNet database.

This whole network is finetuned on the ISTD dataset.

* The second network if the SP network, that takes as input, the shadow mask
and the input image, to compute a relit image
