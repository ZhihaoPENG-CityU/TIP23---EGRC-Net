# reviewing---GACN
<!-- # Graph Augmentation Clustering Network -->
+ This work is being reviewed, we will release the code soon.

DOI:  

URL: 

VIDEO:  
<!-- 
We appreciate it if you use this code and cite our paper, which can be cited as follows,
> @inproceedings{peng2021attention, <br>
>   title={Attention-driven Graph Clustering Network}, <br>
>   author={Peng, Zhihao and Liu, Hui and Jia, Yuheng and Hou, Junhui},  <br>
>   booktitle={Proceedings of the 29th ACM International Conference on Multimedia},  <br>
>   pages={935--943}, <br>
>   year={2021}
> } <br>
 -->
# Environment
+ Python[3.7.9]
+ Pytorch[1.7.1]
+ GPU (GeForce RTX 2080 Ti) & (NVIDIA GeForce RTX 3090) & (Quadro RTX 8000)

# Hyperparameters
<!-- + The learning rates of USPS, HHAR, ACM, and DBLP datasets are set to 0.001, and the learning rates of Reuters and CiteSeer datasets are set to 0.0001. lambda1 and lambda2 are set to {1000, 1000} for USPS, {1, 0.1} for HHAR, {10, 10} for Reuters, and {0.1, 0.01} for graph datasets. -->

# To run code
+ Step 1: set the hyperparameters for the specific dataset;
+ Step 2: python main_XXX.py
+ 
* For examle, if u would like to run AGCN on the DBLP dataset, u need to
* first set {0.01, 0.01, 0.1} for USPS;
* then run the command "python main_DBLP.py"

# Data
Due to the limitation of GitHub, we share the data in [<a href="https://drive.google.com/drive/folders/1l8H662Yj5Cn6Af2EsgxhojYYmGY2_ASE?usp=sharing">here</a>].
