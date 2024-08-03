# Depression-Detection-by-Federated-Learning

The code for the paper [Depression Detection Through Smartphone Sensing: A Federated Learning Approach](https://www.researchgate.net/publication/367041442_Depression_Detection_Through_Smartphone_Sensing_A_Federated_Learning_Approach) [iJIM 2023].

#  Abstract

Depression is one of the most common mental health disorders which affects thousands of lives worldwide. The variation of depressive symptoms among individuals makes it difficult to detect and diagnose early. Moreover, the diagnosing procedure relies heavily on human intervention, making it prone to mistakes. Previous research shows that smartphone sensor data correlates to the users’ mental conditions. By applying machine learning algorithms to sensor data, the mental health status of a person can be predicted. However, traditional machine learning faces privacy challenges as it involves gathering patient data for training. Newly, federated learning has emerged as an effective solution for addressing the privacy issues of classical machine learning. In this study, we apply federated learning to predict depression severity using smartphone sensing capabilities. We develop a deep neural network model and measure its performance in centralized and federated learning settings. The results are quite promising, which validates the potential of federated learning as an alternative to traditional machine learning, with the added benefit of data privacy.

#  Demo

<p float="left">
  <img src="https://github.com/Nawrin14/Depression-Detection-by-Federated-Learning/blob/main/Figure%203%20(Part%202).png" width="250" />
  <img src="https://github.com/Nawrin14/Depression-Detection-by-Federated-Learning/blob/main/Figure%203%20(Part%203).png" width="250" /> 
  <img src="https://github.com/Nawrin14/Depression-Detection-by-Federated-Learning/blob/main/Figure%203%20(Part%204).png" width="250" />
</p>

#  Requirements

- Deeplearning4J (DL4J) => 1.0.0-beta4
- Android Studio => 2021.2.1
- Firebase cloud storage => 19.2.0 
- Firebase real-time database => 19.4.0


#  Results

Table: Performance of the global model after each round

| Performance Metrics  | Round 1 |  Round 2  | Round 3 |  Round 4  | Round 5 |
| ------------- | ------------- |  ------------- | ------------- |  ------------- | ------------- |
| Accuracy  | 0.6854  |  0.6792  | 0.6661  |  0.6576  | 0.6517  |
| Precision  | 0.6899  |  0.7146  | 0.7224  |  0.7110  | 0.6978  |
| Recall  | 0.4844  |  0.4821  | 0.4793    | 0.4731  |  0.4665
| F1-Score  | 0.5215  |  0.5139  | 0.5056  |  0.4914  |  0.4759  |

#  Cite

```
@article{tabassum2023depression,
  title={Depression Detection Through Smartphone Sensing: A Federated Learning Approach.},
  author={Tabassum, Nawrin and Ahmed, Mustofa and Shorna, Nushrat Jahan and Sowad, Ur Rahman and Mejbah, MD and Haque, HM},
  journal={International Journal of Interactive Mobile Technologies},
  volume={17},
  number={1},
  pages={40-56},
  year={2023},
  doi={10.3991/ijim.v17i01.35131}
}
```
