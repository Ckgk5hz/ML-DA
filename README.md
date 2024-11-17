# ML-DA
# About Project
## Dataset:
The data consists of 52,416 observations of energy consumption on a 10-minute window in Tetouan, a city north of Morocco. Tetouan occupies an area of 10375 km² and its population is about 550.374 inhabitants. Morocco's per capita energy consumption is 0.56 toe (around 42% below the North Africa average), including around 900 kWh of electricity (38% below the regional average) (2020). 
Every observation is described by 9 feature columns.
1.	Date Time: Time window of ten minutes.
2.	Temperature: Weather Temperature.
3.	Humidity: Weather Humidity.
4.	Wind Speed: Wind Speed.
5.	General Diffuse Flows: “Diffuse flow” is a catchall term to describe low-temperature (< 0.2° to ~ 100°C) fluids that slowly discharge through sulfide mounds, fractured lava flows, and assemblages of bacterial mats and macrofauna.
6.	Diffuse Flows
7.	Quads (Zone 1) Power Consumption
8.	Smir (Zone 2) Power Consumption
9.	Boussafou (Zone 3) Power Consumption

## Objective:
To derive meaningful observations by focusing on a particular zone from the dataset by performing clustering algorithms on it.
Functionalities/Implementations:
Firstly, we started with the importation of the obvious modules which would be required for this project and then went on to import the dataset using the ‘pandas’ module. From there, we split the class label of ‘Datetime’ by converting it into a datetime format and then separating the ‘Hour’, ‘Day’ and ‘Month’ sections from it.
Now that we obtained these new features, we determined the most effective features for the project by performing feature engineering. This resulted in us using all the features available to cluster the Zone 1.

![image](https://github.com/user-attachments/assets/57c3357c-5b38-41bd-a7e0-577d7908a5a4)
![image](https://github.com/user-attachments/assets/3c971fca-992d-4406-b701-3f58a4eab112)
![image](https://github.com/user-attachments/assets/85206cd1-31d7-4ef7-83b2-2b67b9c5e86c)

After determining the features, we realized that there are a lot of missing values, which we decided to fill in using the SimpleImputer function via the sklearn library with the ‘mean’ strategy.
Then we split the dataset into train and test via the train_test_split function, opting to use an 80-20 balance for training and testing. To scale the features of the training and testing datasets we use StandardScaler function from the preprocessing library of the sklearn module.
Initial attempts to cluster the dataset quickly revealed that the computing strength required for the task was too high, which made us explore different avenues. We settled on using Principal Component Analysis (PCA), to reduce the dimensionality which in turn allowed faster runtimes without compromising on performance rates.

![image](https://github.com/user-attachments/assets/77be3572-5184-4d49-beef-3b1ed570e116)

With the newly reduced dimensionality dataset, we perform a naïve clustering method, K-means clustering on 3 clusters and obtained the inertia and silhouette scores.
K-Means Inertia: 258382.4940796833
K-Means Silhouette Score: 0.23

![image](https://github.com/user-attachments/assets/dba4769e-df8a-444c-85d5-469d6a0e86a0)

We went onto use Density Based Scan Clustering, where we used the metric of Silhouette score.
DBSCAN Silhouette Score: -0.5034450625416471

![image](https://github.com/user-attachments/assets/50bd447e-35a3-431c-8dbd-892a7630d7df)

We were not impressed by the simple scores of the silhouette scores of K-Means and DBScan, which were 0.23 and -0.5 each respectively.
Therefore, we went out of our way to implement and compare performance with three other models, which were Agglometrive Clustering, Gaussian Mixture Model (GMM) and Spectral Clustering.
Below is the Silhouette Score for Agglometrive Clustering followed by the obtained graph.
Agglomerative Clustering Silhouette Score: 0.21679402300698916

![image](https://github.com/user-attachments/assets/ae568a9f-f394-451c-9deb-ac10c964b411)

In a similar manner, we performed it for the Gaussian Mixture Model,
Gaussian Mixture Model Silhouette Score: 0.11250577875994525

![image](https://github.com/user-attachments/assets/3ef663d2-7692-4221-9a04-999e91ad6f47)

And finally Spectral Clustering Method,
Spectral Clustering Silhouette Score: 0.08131653601450714

![image](https://github.com/user-attachments/assets/c41024e5-fd1f-4f27-baf9-c507964e27a8)

# Observations

Our first approach of cleansing and pre-processing the dataset which was clustered by K-Means and then by DBScan proved to be poor, which compelled us to move further onto varied models which were Agglometrive Clustering, Gaussian Mixture Model and Spectral Clustering.
The silhouette results of the K-Means with 3 clusters were 0.23484576163926824 and DBSCAN Silhouette Score gave us -0.5034450625416471. Agglometrive Clustering is 0.21679402300698916 while Gaussian Mixture Model Silhouette Score was 0.11250577875994525 and finally Spectral Clustering Silhouette Score resulted in 0.08131653601450714.

# Additional Functionalities

We went onto apply Log Transformation on the skewed features which was done by adding a small constant to avoid log(0). This was followed Standard Scaling and then using a MinMax scaler on top of that and then performing normalization on it.
Using a subset of the dataset enabled us to reduce computing strain, which was followed by Dimensionality Reduction using Kernel RBF and Kernel Poly.
The results are as below:

### KMeans
KMeans on Kernel PCA (RBF)
KMeans Silhouette Score: 0.7429782957014517 

![image](https://github.com/user-attachments/assets/39201d7a-88cf-45ab-bdac-98a0280efea2)

KMeans on Kernel PCA (Poly)
KMeans Silhouette Score: 0.6021024748192707

![image](https://github.com/user-attachments/assets/f4aa8361-6c2d-41c5-bbc1-dfcefab45c0c)

KMeans on Min-Max Scaled
KMeans Silhouette Score: 0.5412402568034891

![image](https://github.com/user-attachments/assets/8f8b50fb-a200-4ae4-86dd-a58386da0da5)

KMeans on Normalized
KMeans Silhouette Score: 0.47551634818190414

![image](https://github.com/user-attachments/assets/a74b1dcb-1558-4298-ab43-8d88544fe935)

### Agglometrive Clustering

Agglomerative on Kernel PCA (RBF)
Agglomerative Silhouette Score: 0.7076846893491316

![image](https://github.com/user-attachments/assets/32265602-d8c4-4241-8aca-f0fce6c81c37)

Agglomerative on Kernel PCA (Poly)
Agglomerative Silhouette Score: 0.5481265230187977

![image](https://github.com/user-attachments/assets/6d0e72da-3cef-4788-bd26-61fc895bcb26)

Agglomerative on Min-Max Scaled
Agglomerative Silhouette Score: 0.5250097492076684

![image](https://github.com/user-attachments/assets/dab3fc49-6eb1-4fa2-834b-43ad151a8361)

Agglomerative on Normalized
Agglomerative Silhouette Score: 0.4542098433162382

![image](https://github.com/user-attachments/assets/3bc21a76-6319-4dee-9a67-6bde12d4f461)

### Gaussian Mixture Model

GaussianMixture on Kernel PCA (RBF)
GaussianMixture Silhouette Score: 0.6309643202953807

![image](https://github.com/user-attachments/assets/518e4fa9-e81c-437b-a516-e96426c0bbc9)

GaussianMixture on Kernel PCA (Poly)
GaussianMixture Silhouette Score: 0.5230578253457938

![image](https://github.com/user-attachments/assets/02611ef7-277f-459b-a8d6-434bda8b3a92)

GaussianMixture on Min-Max Scaled
GaussianMixture Silhouette Score: 0.48489960195339965

![image](https://github.com/user-attachments/assets/02018ae7-5ff7-4d6a-894b-968b7c41e0ca)

GaussianMixture on Normalized
GaussianMixture Silhouette Score: 0.4474291608391861

![image](https://github.com/user-attachments/assets/54abb787-d9ae-4fe2-af30-01cc3812759a)

### Spectral Clustering

Spectral on Kernel PCA (RBF)
Spectral Silhouette Score: 0.7385784022981675

![image](https://github.com/user-attachments/assets/54515087-0980-4d76-8a25-7d7b2de70ae0)

Spectral on Kernel PCA (Poly)
Spectral Silhouette Score: 0.6084187729897249

![image](https://github.com/user-attachments/assets/c3d6a50f-f1e0-4371-8490-96920f8f6f1e)

Spectral on Min-Max Scaled
Spectral Silhouette Score: 0.5361547510235075

![image](https://github.com/user-attachments/assets/c0923fee-116d-44f8-8da2-7d5a0d45e90b)

Spectral on Normalized
Spectral Silhouette Score: 0.33189456040843224

![image](https://github.com/user-attachments/assets/eabb8a96-f2a8-4e49-9dd3-909aa1f66eb5)

# Conclusion

Best Overall Scores:
1.	KMeans with Kernel PCA (RBF): 0.7429
2.	Spectral Clustering with Kernel PCA (RBF): 0.7386
3.	Agglomerative Clustering with Kernel PCA (RBF): 0.7077

The RBF Kernel PCA transformation paired with KMeans and Spectral Clustering produced the highest silhouette scores, indicating these combinations best captured the structure of the data. This non-linear transformation likely emphasized clusters more effectively than traditional scaling methods, leading to clearer separation and cohesion within clusters. 

These results suggest that leveraging non-linear transformations like RBF Kernel PCA with KMeans or Spectral Clustering could be optimal for this dataset.












