# CS-433 Machine Learning - Project 2
## StochasticGnocchiDescent Team

Francesco Borg, Arturo Cerasi, Davide Mazzali

### Reproducibility of results
The models have been trained on the MNIST dataset using the networks contained in `neur_nets.py`, each model can be trained using the notebook `running_notebook.ipynb` and all the graphical representations can be obtained by running `plots.ipynb`. For the comparison between PAGE_64, PAGE_256, SGD_64 and SGD_256 with LeNet, also already trained models' parameters are provided in the folder `trained_models`.

### Content
- `report`: The final report with the results
- `logs_for_performance_plots`: a folder containig all the logs containing number of gradients, running time, training loss and test accuracy, which have been used to obtain the results
- `logs_for_running_times_tables`: a folder containing all the logs used to create the running time tables contained in `report`
- `aggr.py`: Python script to perform aggregation on the logs for graphical representations
- `aggregate_running_time_data.ipynb`: Notebook used to aggregate the running times data
- `average_running_time.py`: Python script containing useful functions for aggregating running times
- `load_data.py`: Python script used to load dataset used in the experiments
- `neur_nets.py`: Python script containing all the Neural Networks configurations used
- `page.py`: Python script containing the more practical implementation of PAGE optimizer based on epochs
- `plain_page.py`: Python script containing the more theoretical implementation of PAGE optimizer
- `plots.ipynb`: Notebook producing all the graphs contained in `report`
- `run_plain_page_mlp.ipynb`: Notebook to train a simple shallow Neural Network using the plain implementation of PAGE returning the last set of weights
- `run_plain_page_mlp_random_return.ipynb`: Notebook to train a simple shallow Neural Network using the plain implementation of PAGE returning a random set of weights as stated in the pseudocode of the original paper
- `running_notebook.ipynb`: principal Notebook on which to run the main comparison between PAGE and SGD

### System
We ran the code on Google Colaboratory and on Ubuntu 20.04.3 LTS. For Ubuntu, see below the packages and versions used.

#### Python Version
- python 3.9.7

#### Required Imports
- csv
- copy
- matplotlib 3.4.2
- numpy 1.19.2
- pytorch 1.8.1
- seaborn 0.11.2
- time
- torchvision 0.2.2

Should there be any dependency issue, please refer too `misc/dump_packages_versions.txt` for details on the packages in our Ubuntu conda (anaconda3) environment

#### Only for ProxSARAH
To run experiments for ProxSARAH we have slightly modified the author's code to output data in a more convenient formato to us. All credits to
* N. H. Pham, L. M. Nguyen, D. T. Phan, and Q. Tran-Dinh. **[Proxsarah: An efficient algorithmic framework for stochastic composite non-convex optimization](http://jmlr.org/papers/v21/19-248.html)**. <em>Journal of Machine Learning Research</em>, 21(110):1–48,2020.
The code we used for ProxSARAH can be found in `prox_sarah` and correspondes to their repo's folder `tensorflow_src`.
The lines with modified in `method_ProxSARAH.py` are labelled as `# modified print by StochasticGnocchiDescent`.
To run this code, use Python 3.6.3 and TensorFlow 1.12. Other needed packages are keras, sklearn, matplotlib, argParser.
Running this code on Ubuntu 20.04.3 we experienced some issues in the beginning, hence the full dump of packages in conda environment we used can be found in `misc/dump_packages_versions_proxsarah.txt`
