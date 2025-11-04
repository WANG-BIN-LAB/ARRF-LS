ARRF-LS
=======
Abstract
--------
Reliable individual identification via functional connectivity (FC) enables accurate prediction of cognitive and behavioral traits, and facilitates the advancement of personalized medicine. Compared to statistical and deep learning methods, component decomposition demonstrates great potential as it explicitly models  common components thereby captures individual-specific components through residuals. However, current decomposition methods are unable to directly extract individual-specific features, which introduces errors and limits the performance of individual identification. Facing this bottleneck, we find that low-rank and sparse (LS) decomposition algorithms can distinctly separate commonalities and individual-specific variations through low-rank and sparse components. To design an LS decomposition algorithm for individual identification, we need to tackle two challenges. On the one hand, current algorithms rely on a pre-set sparse threshold, making it impossible to determine whether the sparse threshold is optimal. On the other hand, the existing LS decomposition algorithm has a slow convergence rate. In our work, we propose a Low-Rank and Sparse Decomposition Model with Adaptive Regularization and Residual Feedback (ARRF-LS). The adaptive regularization strategy based on sparse feedback significantly improves the recognition accuracy by self-calibrating the sparse threshold to determine the optimal threshold. The residual balancing mechanism that adjusts the step size through iterative feedback greatly accelerates the convergence rate in LS decomposition. Extensive experiments demonstrate that our ARRF-LS framework outperforms all state-of-the-art methods and exhibits strong performance in cross-task individual identification. 

Dependencies
------------
python==3.11.10<br>
torch==2.1.0+cu118<br>
numpy==1.26.4<br>
pandas==2.2.3<br>
scipy==1.31.1<br>

Installation
-----------
Run the following command to create and configure the environment:<br>
``` 
#Create environment
conda create --name ARRF-LS python=3.11.10
#Activate environment
conda activate ARRF-LS
#Install dependency packages
pip install -r requirements.txt
```
🚀 Usage
---------
Run the following command to start the main program:<br>
``` 
python main.py
