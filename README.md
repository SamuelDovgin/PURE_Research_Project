# PURE Research Mutual Information Estimation with K-Nearest-Neighbors
By: Samuel Dovgin
Completed: 12/10/2018

This reasearch project focuses on the optimizations of finding Mutual Information between features in a data set when using the 
K-Nearest-Neighbor Algorithm. Specifically the KSG Nearest Neighbor Estimator is used which is a naive algorithm which takes into
account the distances between each pair of points. Due to the necessity of n^2 comparisons, optimizations are necessary to
work with large datasets in a reasonable amount of time.

<img src="https://github.com/SamuelDovgin/PURE_Research_Project/blob/master/poster_assets/dovgin2_and_tinfang2_poster-1.jpg" />

# Background Links
[KSG Algorithm Original Research Paper](https://github.com/SamuelDovgin/PURE_Research_Project/blob/master/KSG_estimator/KSG%20original.pdf)
[Mentor-written Intro to Topic](https://github.com/SamuelDovgin/PURE_Research_Project/blob/master/KSG_estimator/MI_estimation_writeup_rev1.pdf)

# Conclusion
We were able to develop several strategies and compare different methods of more efficiently computing K Nearest Neighbors. 
This can be applied to our choice of the KSG Mutual Information Estimator however it can improve other algorithms as well. 
Future work would be to continue improving KNN efficiency or that of other algorithms.

Throughout this project I learned how to utilize new data structures such as KD-trees and implementing them to decrease 
runtime. Additionally, I gained experience in how to parse large machine learning datasets and apply specific classifiers
in PYTHON. Experience with python libraries Matplotlib and Numpy will be extremely useful in my future projects and 
endevors.

# Acknowledgments
Mentor: Alan Yang
PURE administrative team
