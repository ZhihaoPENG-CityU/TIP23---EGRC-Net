<!--# reviewing work
 # Graph Augmentation Clustering Network 
+ This work is being reviewed, we will release the code soon. -->

DOI: https://ieeexplore.ieee.org/abstract/document/10326461

URL: https://arxiv.org/abs/2211.10627

We have added comments in the code, and the specific details can correspond to the explanation in the paper. Please get in touch with me (zhihapeng3-c@my.cityu.edu.hk) if you have any issues.

We appreciate it if you use this code and cite our related papers, which can be cited as follows,

> @article{peng2023egrc, <br>
>   title={EGRC-Net: Embedding-Induced Graph Refinement Clustering Network}, <br>
>   author={Peng, Zhihao and Liu, Hui and Jia, Yuheng and Hou, Junhui},  <br>
>   journal={IEEE Transactions on Image Processing}, <br>
>   volume={32}, <br>
>   pages={6457--6468}, <br>
>   year={2023}, <br>
>   publisher={IEEE}
> } <br>

> @article{peng2022deep, <br>
>   title={Deep Attention-guided Graph Clustering with Dual Self-supervision}, <br>
>   author={Peng, Zhihao and Liu, Hui and Jia, Yuheng and Hou, Junhui},  <br>
>   journal={IEEE Transactions on Circuits and Systems for Video Technology},  <br>
>   year={2022}, <br>
>   publisher={IEEE}
> } <br>

> @inproceedings{peng2021attention, <br>
>   title={Attention-driven graph clustering network}, <br>
>   author={Peng, Zhihao and Liu, Hui and Jia, Yuheng and Hou, Junhui},  <br>
>   booktitle={Proceedings of the 29th ACM International Conference on Multimedia},  <br>
>   pages={935--943},  <br>
>   year={2021}
> } <br>

# Environment
+ Python[3.7.9]
+ Pytorch[1.7.1]
+ GPU (GeForce RTX 2080 Ti) & (NVIDIA GeForce RTX 3090) & (Quadro RTX 8000)

# Hyperparameters
<!-- + The learning rates of USPS, HHAR, ACM, and DBLP datasets are set to 0.001, and the learning rates of Reuters and CiteSeer datasets are set to 0.0001. lambda1 and lambda2 are set to {1000, 1000} for USPS, {1, 0.1} for HHAR, {10, 10} for Reuters, and {0.1, 0.01} for graph datasets. -->

# To run code
+ Step 1: set the hyperparameters for the specific dataset;
+ Step 2: python EGRC-Net.py

* For example, if u would like to run AGCN on the DBLP dataset, u need to
* first set {0.01, 0.01, 0.1} for USPS;
* then run the command "python main_DBLP.py"

# Data
Due to the limitation of GitHub, we share the data in [<a href="https://drive.google.com/drive/folders/1l8H662Yj5Cn6Af2EsgxhojYYmGY2_ASE?usp=sharing">here</a>].
