This script performs unsupervised customer segmentation

Uses scaling, clustering, dimensionality reduction, and visualization

Evaluates clusters using multiple metrics

Saves models for production-ready deployment or API integration

====================================================================================

## Project Workflow

### 1. Data Acquisition

- Dataset: Wholesale customers data.csv (UCI Machine Learning Repository).

- Features: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen.

- Target: None (unsupervised).

### 2. Data Preprocessing

- StandardScaler applied to normalize features.

- Saved fitted scaler as scaler.pkl.

### 3. Model Training

- K-Means clustering applied to the dataset.

- Optimal number of clusters determined using the elbow method.

- Trained model saved as kmeans.pkl.

### 4. Model Serving

- model.py contains training and preprocessing logic.

- app.py provides a Gradio interface for interactive predictions.

- Input: Six numerical features.

- Output: Cluster label with interpretation.

### 5. Containerization

- Dockerfile builds a reproducible environment for the app.

- .dockerignore ensures lean image size.

- Image tagged as customer-segmentation.

### 6. Deployment

- deployment.yaml: Kubernetes Deployment manifest.

- service.yaml: Kubernetes Service manifest.

- k8s.yaml: Combined manifest for quick deployment.

- Supports NodePort or LoadBalancer for external access.

===========================================================================

docker build -t wholesale-gradio-app .

docker images

docker run -p 7860:7860 wholesale-gradio-app

eval $(minikube docker-env)

kubectl apply -f deployment.yaml

kubectl get pods

minikube service wholesale-gradio-service

minikube service wholesale-segmentation-service



