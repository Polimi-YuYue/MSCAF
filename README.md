# MSI2DA-Net

Experimental codes for paper "Importance-aware Subgraph Convolutional Networks Based on Multi-source Information Fusion for Cross-domain Mechanical Fault Diagnosis".

<div align=center>
<img src="https://github.com/Polimi-YuYue/MSCAF/blob/main/Overall%20Framework.png" width="700px">
</div>

# Abstract

Intelligent fault diagnosis using multi-source sensor fusion holds significant promise but faces challenges related to reliability due to variations in signal quality across sensors and inconsistencies in fault features. To tackle these issues, a multi-source sensor
correlation adaptive fusion (MSCAF) framework with uncertainty quantification is proposed to enhance identification trustworthiness for intelligent fault diagnosis. The proposed MSCAF integrates Dempster-Shafer theory with Dirichlet distribution to model sensor uncertainty and split multi-source sensors into high-confidence and lowconfidence sensors based on the consistency of cross-sensor fault information. Highconfidence sensors are given greater weight, ensuring more reliable fusion. Then, the reward and penalty functions are introduced to assess their correlation weights. Meanwhile, Convolutional and graph neural networks are employed to enhance feature extraction and output category probabilities, which can ensure robust fusion across
varying diagnostic scenarios. This approach allows adaptive weighting, optimizes fusion reliability, and enables manual intervention for low-confidence sensors. Experimental results demonstrate that the proposed MSCAF achieves superior diagnostic performance compared to state-of-the-art methods, confirming its efficacy in extracting reliable features with uncertainty quantification for intelligent fault diagnosis.


# Paper

# A two-stage importance-aware subgraph convolutional network based on multi-source sensors for cross-domain fault diagnosis

a. Yue Yu, a. Hamid Reza Karimi, b. Len Gelman, c. Jinghui Tian, d. Peng Mei

a Department of Mechanical Engineering, Politecnico di Milano, via La Masa 1, Milan 20156, Italy

b School of Computing and Engineering, University of Huddersfield, Queensgate, Huddersfield, HD1 3DH, UK

c Advanced Manufacturing Center, Ningbo Institute of Technology, Beihang University, Ningbo, 315800, China

d School of Transportation Science and Engineering, Beihang University, Beijing, 100191, China

https://www.sciencedirect.com/science/article/pii/S0951832025010129

# If this code is helpful to you, please cite this paper as follows, thank you!
# Citation

@article{yu2025novel,
  title={A novel multi-source sensor correlation adaptive fusion framework with uncertainty quantification for intelligent fault diagnosis},
  author={Yu, Yue and Karimi, Hamid Reza and Gelman, Len and Tian, Jinghui and Mei, Peng},
  journal={Reliability Engineering \& System Safety},
  pages={111812},
  year={2025},
  publisher={Elsevier}
}

