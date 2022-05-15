# DLS_Project 
# Cloud based High-Dimensional Object Detection using Deep Neural Network


goal:

Explore and compare Deep Learning methodologies for 3D Object Recognition

solution approach:

Models: PointNet, MVCNN
Implementation: PyTorch, Google Colab Notebook
Compare model performance on different datasets

value/benefit:
This experiment helps us find effective approaches to recognize 3D objects

Code Structure:

There are two models being used to test 3D detection, which are PointNet and MV-CNN. PointNet is in pointnet.py, and MV-CNN is in mvcnn_pytorch.

The PointNet architecture proposed for point clouds classification and semantic segmentation tasks are shown in the below image. The top blue path specifies the classification network while the bottom yellow path is the semantic segmentation network.

MV-CNN is a network topology that combines information from different views into fully connected layers to classify the voxel where the planes cross. The multi-view approach can be considered as a 2.5D CNN given that it incorporates information from each image plane, but does not use the full 3D neighborhood of the queried voxel. This results in a lower computational complexity when compared to 3D-kernel methods

Example commands to execute the code:
User can execute 'python pointnet.py' to start pointnet training and testing of PointNet.
User can execute 'python train_mvcnn.py' to start pointnet training and testing of MV-CNN.


Result:

Overall accuracy on ModelNet40:


PointNet: 0.88243, MVCNN: 0.90072


Conclusion:

For overall accuracy with ModelNet40 dataset, MVCNN seems to perform better than PointNet
We believe it is likely because sparse point clouds lose lots of information (while dense point clouds are computation expensive), and for most 3D objects having 2D views is enough to classify
We proposed a hypothesis that with the advantage of having internal structure information, PointNet may perform better than MVCNN on those objects with complex internal structures. 
Result shows slightly improvement. MVCNN performs a bit worse when it tries to distinguish between wardrobes and bookshelves, while PointNet seems to capture their internals and predict slightly better.
MVCNN trains faster than PointNet





Related Work:

Qi et. al., PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
https://github.com/gkadusumilli/kaolin_1/tree/master/Documents/kaolin-0.1
https://arxiv.org/abs/1612.00593
https://www.sciencedirect.com/science/article/pii/S2468502X21000395
https://ieeexplore.ieee.org/document/9696941


Github repo: 
jongchyisu/mvcnn_pytorch
nikitakaraevv/pointnet
gkadusumilli/kaolin_1
