# Topological_temporal_properties
Code repository of the paper Topological-temporal properties of evolving networks.




In this repository you find code to run experiments of the paper "Topological-temporal properties of evolving networks". 
Full Code and Data are avalable at https://surfdrive.surf.nl/files/index.php/s/3dZJjaG4aDQKKEF
If you use our code in your research or projects, please consider citing us. 


```bibtex

@article{ceria2022topological,
  title={Topological--temporal properties of evolving networks},
  author={Ceria, Alberto and Havlin, Shlomo and Hanjalic, Alan and Wang, Huijuan},
  journal={Journal of Complex Networks},
  volume={10},
  number={5},
  pages={cnac041},
  year={2022},
  publisher={Oxford University Press}
}
```


## Repository organization


### Description of Data Folder

-`randomization.py`
Utilities to randomize the labelled higher-order network.

-`resilience_clustering_nx.py`
Utilities to compute the (complementary) order contribution to the number of triangles, to the global largest connected component, and to the number of nodes with each label in the largest connected component of the network.
(Based on NetworkX library)

-`joint_distribution_overlapp.py`
Utilities to compute the group composition probability of a labelled higher-order network.

-`link_weights.py`
Utilities to compute the order contribution and order relevance to the sum of the link weights in a labelled higher-order network.

-`utils.py`
Utilities for our experiments.

### Description of Code Folder



