# shadow-removal-with-image-decomposition


The Data folder contains the sub-folder ISTD_Dataset , that contains two subfolders , the 'train' and
the 'test' folder.
Each of the subfolder of expression have three subfolders, of suffix '_A' , '_B' and '_C':
the '_A' contains images with shadow
the '_B' contains images contains mask of shadow
the '_C' contains images without shadow.

Remarks, I didn't read the article yet, but I suppose , that the the images with and without shadow,
were taken by fixed position of the camera, though, the images were taken on different moment of the day,
which implies a change on the colometry, between the image with shadow and without shadow, that is tackled
 by a linear regression (that would be described later)
