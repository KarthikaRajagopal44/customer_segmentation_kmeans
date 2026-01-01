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

karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ docker build -t wholesale-gradio-app .
Sending build context to Docker daemon  808.5MB
Step 1/7 : FROM python:3.11-slim
3.11-slim: Pulling from library/python
02d7611c4eae: Pull complete 
1473863bb010: Pull complete 
7cc13cb22d92: Pull complete 
654a090213c0: Pull complete 
Digest: sha256:aa9aac8eacc774817e2881238f52d983a5ea13d7f5a1dee479a1a1d466047951
Status: Downloaded newer image for python:3.11-slim
 ---> 955f4ccb5624
Step 2/7 : WORKDIR /app
 ---> Running in 49d586432ee4
Removing intermediate container 49d586432ee4
 ---> ef7078f85ba9
Step 3/7 : COPY requirements.txt .
 ---> 25d3c575dbf0
Step 4/7 : RUN pip install --no-cache-dir -r requirements.txt
 ---> Running in abd8e4c291cc
Collecting pandas==2.2.2 (from -r requirements.txt (line 1))
  Downloading pandas-2.2.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (19 kB)
Collecting numpy==1.26.4 (from -r requirements.txt (line 2))
  Downloading numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.0/61.0 kB 2.1 MB/s eta 0:00:00
Collecting scikit-learn==1.3.2 (from -r requirements.txt (line 3))
  Downloading scikit_learn-1.3.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
Collecting joblib==1.3.2 (from -r requirements.txt (line 4))
  Downloading joblib-1.3.2-py3-none-any.whl.metadata (5.4 kB)
Collecting matplotlib==3.9.2 (from -r requirements.txt (line 5))
  Downloading matplotlib-3.9.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
Collecting seaborn==0.13.2 (from -r requirements.txt (line 6))
  Downloading seaborn-0.13.2-py3-none-any.whl.metadata (5.4 kB)
Collecting gradio==3.50.2 (from -r requirements.txt (line 7))
  Downloading gradio-3.50.2-py3-none-any.whl.metadata (17 kB)
Collecting huggingface_hub==0.20.3 (from -r requirements.txt (line 8))
  Downloading huggingface_hub-0.20.3-py3-none-any.whl.metadata (12 kB)
Collecting python-dateutil>=2.8.2 (from pandas==2.2.2->-r requirements.txt (line 1))
  Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting pytz>=2020.1 (from pandas==2.2.2->-r requirements.txt (line 1))
  Downloading pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
Collecting tzdata>=2022.7 (from pandas==2.2.2->-r requirements.txt (line 1))
  Downloading tzdata-2025.3-py2.py3-none-any.whl.metadata (1.4 kB)
Collecting scipy>=1.5.0 (from scikit-learn==1.3.2->-r requirements.txt (line 3))
  Downloading scipy-1.16.3-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (62 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.0/62.0 kB 5.9 MB/s eta 0:00:00
Collecting threadpoolctl>=2.0.0 (from scikit-learn==1.3.2->-r requirements.txt (line 3))
  Downloading threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
Collecting contourpy>=1.0.1 (from matplotlib==3.9.2->-r requirements.txt (line 5))
  Downloading contourpy-1.3.3-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.5 kB)
Collecting cycler>=0.10 (from matplotlib==3.9.2->-r requirements.txt (line 5))
  Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
Collecting fonttools>=4.22.0 (from matplotlib==3.9.2->-r requirements.txt (line 5))
  Downloading fonttools-4.61.1-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (114 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 114.2/114.2 kB 4.7 MB/s eta 0:00:00
Collecting kiwisolver>=1.3.1 (from matplotlib==3.9.2->-r requirements.txt (line 5))
  Downloading kiwisolver-1.4.9-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (6.3 kB)
Collecting packaging>=20.0 (from matplotlib==3.9.2->-r requirements.txt (line 5))
  Downloading packaging-25.0-py3-none-any.whl.metadata (3.3 kB)
Collecting pillow>=8 (from matplotlib==3.9.2->-r requirements.txt (line 5))
  Downloading pillow-12.0.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.8 kB)
Collecting pyparsing>=2.3.1 (from matplotlib==3.9.2->-r requirements.txt (line 5))
  Downloading pyparsing-3.3.1-py3-none-any.whl.metadata (5.6 kB)
Collecting aiofiles<24.0,>=22.0 (from gradio==3.50.2->-r requirements.txt (line 7))
  Downloading aiofiles-23.2.1-py3-none-any.whl.metadata (9.7 kB)
Collecting altair<6.0,>=4.2.0 (from gradio==3.50.2->-r requirements.txt (line 7))
  Downloading altair-5.5.0-py3-none-any.whl.metadata (11 kB)
Collecting fastapi (from gradio==3.50.2->-r requirements.txt (line 7))
  Downloading fastapi-0.128.0-py3-none-any.whl.metadata (30 kB)
Collecting ffmpy (from gradio==3.50.2->-r requirements.txt (line 7))
  Downloading ffmpy-1.0.0-py3-none-any.whl.metadata (3.0 kB)
Collecting gradio-client==0.6.1 (from gradio==3.50.2->-r requirements.txt (line 7))
  Downloading gradio_client-0.6.1-py3-none-any.whl.metadata (7.1 kB)
Collecting httpx (from gradio==3.50.2->-r requirements.txt (line 7))
  Downloading httpx-0.28.1-py3-none-any.whl.metadata (7.1 kB)
Collecting importlib-resources<7.0,>=1.3 (from gradio==3.50.2->-r requirements.txt (line 7))
  Downloading importlib_resources-6.5.2-py3-none-any.whl.metadata (3.9 kB)
Collecting jinja2<4.0 (from gradio==3.50.2->-r requirements.txt (line 7))
  Downloading jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
Collecting markupsafe~=2.0 (from gradio==3.50.2->-r requirements.txt (line 7))
  Downloading MarkupSafe-2.1.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.0 kB)
Collecting orjson~=3.0 (from gradio==3.50.2->-r requirements.txt (line 7))
  Downloading orjson-3.11.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (41 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 41.6/41.6 kB 64.6 MB/s eta 0:00:00
Collecting pillow>=8 (from matplotlib==3.9.2->-r requirements.txt (line 5))
  Downloading pillow-10.4.0-cp311-cp311-manylinux_2_28_x86_64.whl.metadata (9.2 kB)
Collecting pydantic!=1.8,!=1.8.1,!=2.0.0,!=2.0.1,<3.0.0,>=1.7.4 (from gradio==3.50.2->-r requirements.txt (line 7))
  Downloading pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 90.6/90.6 kB 3.2 MB/s eta 0:00:00
Collecting pydub (from gradio==3.50.2->-r requirements.txt (line 7))
  Downloading pydub-0.25.1-py2.py3-none-any.whl.metadata (1.4 kB)
Collecting python-multipart (from gradio==3.50.2->-r requirements.txt (line 7))
  Downloading python_multipart-0.0.21-py3-none-any.whl.metadata (1.8 kB)
Collecting pyyaml<7.0,>=5.0 (from gradio==3.50.2->-r requirements.txt (line 7))
  Downloading pyyaml-6.0.3-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (2.4 kB)
Collecting requests~=2.0 (from gradio==3.50.2->-r requirements.txt (line 7))
  Downloading requests-2.32.5-py3-none-any.whl.metadata (4.9 kB)
Collecting semantic-version~=2.0 (from gradio==3.50.2->-r requirements.txt (line 7))
  Downloading semantic_version-2.10.0-py2.py3-none-any.whl.metadata (9.7 kB)
Collecting typing-extensions~=4.0 (from gradio==3.50.2->-r requirements.txt (line 7))
  Downloading typing_extensions-4.15.0-py3-none-any.whl.metadata (3.3 kB)
Collecting uvicorn>=0.14.0 (from gradio==3.50.2->-r requirements.txt (line 7))
  Downloading uvicorn-0.40.0-py3-none-any.whl.metadata (6.7 kB)
Collecting websockets<12.0,>=10.0 (from gradio==3.50.2->-r requirements.txt (line 7))
  Downloading websockets-11.0.3-cp311-cp311-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.6 kB)
Collecting filelock (from huggingface_hub==0.20.3->-r requirements.txt (line 8))
  Downloading filelock-3.20.1-py3-none-any.whl.metadata (2.1 kB)
Collecting fsspec>=2023.5.0 (from huggingface_hub==0.20.3->-r requirements.txt (line 8))
  Downloading fsspec-2025.12.0-py3-none-any.whl.metadata (10 kB)
Collecting tqdm>=4.42.1 (from huggingface_hub==0.20.3->-r requirements.txt (line 8))
  Downloading tqdm-4.67.1-py3-none-any.whl.metadata (57 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57.7/57.7 kB 4.1 MB/s eta 0:00:00
Collecting jsonschema>=3.0 (from altair<6.0,>=4.2.0->gradio==3.50.2->-r requirements.txt (line 7))
  Downloading jsonschema-4.25.1-py3-none-any.whl.metadata (7.6 kB)
Collecting narwhals>=1.14.2 (from altair<6.0,>=4.2.0->gradio==3.50.2->-r requirements.txt (line 7))
  Downloading narwhals-2.14.0-py3-none-any.whl.metadata (13 kB)
Collecting annotated-types>=0.6.0 (from pydantic!=1.8,!=1.8.1,!=2.0.0,!=2.0.1,<3.0.0,>=1.7.4->gradio==3.50.2->-r requirements.txt (line 7))
  Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
Collecting pydantic-core==2.41.5 (from pydantic!=1.8,!=1.8.1,!=2.0.0,!=2.0.1,<3.0.0,>=1.7.4->gradio==3.50.2->-r requirements.txt (line 7))
  Downloading pydantic_core-2.41.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
Collecting typing-inspection>=0.4.2 (from pydantic!=1.8,!=1.8.1,!=2.0.0,!=2.0.1,<3.0.0,>=1.7.4->gradio==3.50.2->-r requirements.txt (line 7))
  Downloading typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
Collecting six>=1.5 (from python-dateutil>=2.8.2->pandas==2.2.2->-r requirements.txt (line 1))
  Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Collecting charset_normalizer<4,>=2 (from requests~=2.0->gradio==3.50.2->-r requirements.txt (line 7))
  Downloading charset_normalizer-3.4.4-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (37 kB)
Collecting idna<4,>=2.5 (from requests~=2.0->gradio==3.50.2->-r requirements.txt (line 7))
  Downloading idna-3.11-py3-none-any.whl.metadata (8.4 kB)
Collecting urllib3<3,>=1.21.1 (from requests~=2.0->gradio==3.50.2->-r requirements.txt (line 7))
  Downloading urllib3-2.6.2-py3-none-any.whl.metadata (6.6 kB)
Collecting certifi>=2017.4.17 (from requests~=2.0->gradio==3.50.2->-r requirements.txt (line 7))
  Downloading certifi-2025.11.12-py3-none-any.whl.metadata (2.5 kB)
Collecting click>=7.0 (from uvicorn>=0.14.0->gradio==3.50.2->-r requirements.txt (line 7))
  Downloading click-8.3.1-py3-none-any.whl.metadata (2.6 kB)
Collecting h11>=0.8 (from uvicorn>=0.14.0->gradio==3.50.2->-r requirements.txt (line 7))
  Downloading h11-0.16.0-py3-none-any.whl.metadata (8.3 kB)
Collecting starlette<0.51.0,>=0.40.0 (from fastapi->gradio==3.50.2->-r requirements.txt (line 7))
  Downloading starlette-0.50.0-py3-none-any.whl.metadata (6.3 kB)
Collecting annotated-doc>=0.0.2 (from fastapi->gradio==3.50.2->-r requirements.txt (line 7))
  Downloading annotated_doc-0.0.4-py3-none-any.whl.metadata (6.6 kB)
Collecting anyio (from httpx->gradio==3.50.2->-r requirements.txt (line 7))
  Downloading anyio-4.12.0-py3-none-any.whl.metadata (4.3 kB)
Collecting httpcore==1.* (from httpx->gradio==3.50.2->-r requirements.txt (line 7))
  Downloading httpcore-1.0.9-py3-none-any.whl.metadata (21 kB)
Collecting attrs>=22.2.0 (from jsonschema>=3.0->altair<6.0,>=4.2.0->gradio==3.50.2->-r requirements.txt (line 7))
  Downloading attrs-25.4.0-py3-none-any.whl.metadata (10 kB)
Collecting jsonschema-specifications>=2023.03.6 (from jsonschema>=3.0->altair<6.0,>=4.2.0->gradio==3.50.2->-r requirements.txt (line 7))
  Downloading jsonschema_specifications-2025.9.1-py3-none-any.whl.metadata (2.9 kB)
Collecting referencing>=0.28.4 (from jsonschema>=3.0->altair<6.0,>=4.2.0->gradio==3.50.2->-r requirements.txt (line 7))
  Downloading referencing-0.37.0-py3-none-any.whl.metadata (2.8 kB)
Collecting rpds-py>=0.7.1 (from jsonschema>=3.0->altair<6.0,>=4.2.0->gradio==3.50.2->-r requirements.txt (line 7))
  Downloading rpds_py-0.30.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.1 kB)
Downloading pandas-2.2.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (13.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 13.0/13.0 MB 6.4 MB/s eta 0:00:00
Downloading numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.3/18.3 MB 3.3 MB/s eta 0:00:00
Downloading scikit_learn-1.3.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (10.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.9/10.9 MB 4.6 MB/s eta 0:00:00
Downloading joblib-1.3.2-py3-none-any.whl (302 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 302.2/302.2 kB 4.6 MB/s eta 0:00:00
Downloading matplotlib-3.9.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (8.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.3/8.3 MB 5.9 MB/s eta 0:00:00
Downloading seaborn-0.13.2-py3-none-any.whl (294 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 294.9/294.9 kB 7.3 MB/s eta 0:00:00
Downloading gradio-3.50.2-py3-none-any.whl (20.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 20.3/20.3 MB 6.9 MB/s eta 0:00:00
Downloading huggingface_hub-0.20.3-py3-none-any.whl (330 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 330.1/330.1 kB 6.7 MB/s eta 0:00:00
Downloading gradio_client-0.6.1-py3-none-any.whl (299 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 299.2/299.2 kB 8.4 MB/s eta 0:00:00
Downloading aiofiles-23.2.1-py3-none-any.whl (15 kB)
Downloading altair-5.5.0-py3-none-any.whl (731 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 731.2/731.2 kB 7.1 MB/s eta 0:00:00
Downloading contourpy-1.3.3-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (355 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 355.2/355.2 kB 8.5 MB/s eta 0:00:00
Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
Downloading fonttools-4.61.1-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (5.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.0/5.0 MB 5.9 MB/s eta 0:00:00
Downloading fsspec-2025.12.0-py3-none-any.whl (201 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 201.4/201.4 kB 4.1 MB/s eta 0:00:00
Downloading importlib_resources-6.5.2-py3-none-any.whl (37 kB)
Downloading jinja2-3.1.6-py3-none-any.whl (134 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 134.9/134.9 kB 3.2 MB/s eta 0:00:00
Downloading kiwisolver-1.4.9-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.4/1.4 MB 2.6 MB/s eta 0:00:00
Downloading MarkupSafe-2.1.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (28 kB)
Downloading orjson-3.11.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (138 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 138.8/138.8 kB 3.0 MB/s eta 0:00:00
Downloading packaging-25.0-py3-none-any.whl (66 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 66.5/66.5 kB 2.5 MB/s eta 0:00:00
Downloading pillow-10.4.0-cp311-cp311-manylinux_2_28_x86_64.whl (4.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.5/4.5 MB 1.4 MB/s eta 0:00:00
Downloading pydantic-2.12.5-py3-none-any.whl (463 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 463.6/463.6 kB 5.0 MB/s eta 0:00:00
Downloading pydantic_core-2.41.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 5.8 MB/s eta 0:00:00
Downloading pyparsing-3.3.1-py3-none-any.whl (121 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121.8/121.8 kB 8.9 MB/s eta 0:00:00
Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 229.9/229.9 kB 2.2 MB/s eta 0:00:00
Downloading pytz-2025.2-py2.py3-none-any.whl (509 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 509.2/509.2 kB 5.0 MB/s eta 0:00:00
Downloading pyyaml-6.0.3-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (806 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 806.6/806.6 kB 6.3 MB/s eta 0:00:00
Downloading requests-2.32.5-py3-none-any.whl (64 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 64.7/64.7 kB 9.7 MB/s eta 0:00:00
Downloading scipy-1.16.3-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (35.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.9/35.9 MB 3.0 MB/s eta 0:00:00
Downloading semantic_version-2.10.0-py2.py3-none-any.whl (15 kB)
Downloading threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Downloading tqdm-4.67.1-py3-none-any.whl (78 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.5/78.5 kB 6.3 MB/s eta 0:00:00
Downloading typing_extensions-4.15.0-py3-none-any.whl (44 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.6/44.6 kB 8.5 MB/s eta 0:00:00
Downloading tzdata-2025.3-py2.py3-none-any.whl (348 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 348.5/348.5 kB 2.9 MB/s eta 0:00:00
Downloading uvicorn-0.40.0-py3-none-any.whl (68 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 68.5/68.5 kB 2.5 MB/s eta 0:00:00
Downloading websockets-11.0.3-cp311-cp311-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (130 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 130.6/130.6 kB 8.1 MB/s eta 0:00:00
Downloading fastapi-0.128.0-py3-none-any.whl (103 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 103.1/103.1 kB 6.5 MB/s eta 0:00:00
Downloading ffmpy-1.0.0-py3-none-any.whl (5.6 kB)
Downloading filelock-3.20.1-py3-none-any.whl (16 kB)
Downloading httpx-0.28.1-py3-none-any.whl (73 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 73.5/73.5 kB 8.4 MB/s eta 0:00:00
Downloading httpcore-1.0.9-py3-none-any.whl (78 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.8/78.8 kB 7.4 MB/s eta 0:00:00
Downloading pydub-0.25.1-py2.py3-none-any.whl (32 kB)
Downloading python_multipart-0.0.21-py3-none-any.whl (24 kB)
Downloading annotated_doc-0.0.4-py3-none-any.whl (5.3 kB)
Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
Downloading certifi-2025.11.12-py3-none-any.whl (159 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 159.4/159.4 kB 5.7 MB/s eta 0:00:00
Downloading charset_normalizer-3.4.4-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (151 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 151.6/151.6 kB 4.8 MB/s eta 0:00:00
Downloading click-8.3.1-py3-none-any.whl (108 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 108.3/108.3 kB 5.7 MB/s eta 0:00:00
Downloading h11-0.16.0-py3-none-any.whl (37 kB)
Downloading idna-3.11-py3-none-any.whl (71 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 71.0/71.0 kB 9.8 MB/s eta 0:00:00
Downloading jsonschema-4.25.1-py3-none-any.whl (90 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 90.0/90.0 kB 9.5 MB/s eta 0:00:00
Downloading narwhals-2.14.0-py3-none-any.whl (430 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 430.8/430.8 kB 5.5 MB/s eta 0:00:00
Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
Downloading starlette-0.50.0-py3-none-any.whl (74 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 74.0/74.0 kB 10.1 MB/s eta 0:00:00
Downloading anyio-4.12.0-py3-none-any.whl (113 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 113.4/113.4 kB 9.1 MB/s eta 0:00:00
Downloading typing_inspection-0.4.2-py3-none-any.whl (14 kB)
Downloading urllib3-2.6.2-py3-none-any.whl (131 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 131.2/131.2 kB 8.4 MB/s eta 0:00:00
Downloading attrs-25.4.0-py3-none-any.whl (67 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 67.6/67.6 kB 8.7 MB/s eta 0:00:00
Downloading jsonschema_specifications-2025.9.1-py3-none-any.whl (18 kB)
Downloading referencing-0.37.0-py3-none-any.whl (26 kB)
Downloading rpds_py-0.30.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (390 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 390.8/390.8 kB 2.3 MB/s eta 0:00:00
Installing collected packages: pytz, pydub, websockets, urllib3, tzdata, typing-extensions, tqdm, threadpoolctl, six, semantic-version, rpds-py, pyyaml, python-multipart, pyparsing, pillow, packaging, orjson, numpy, narwhals, markupsafe, kiwisolver, joblib, importlib-resources, idna, h11, fsspec, fonttools, filelock, ffmpy, cycler, click, charset_normalizer, certifi, attrs, annotated-types, annotated-doc, aiofiles, uvicorn, typing-inspection, scipy, requests, referencing, python-dateutil, pydantic-core, jinja2, httpcore, contourpy, anyio, starlette, scikit-learn, pydantic, pandas, matplotlib, jsonschema-specifications, huggingface_hub, httpx, seaborn, jsonschema, gradio-client, fastapi, altair, gradio
Successfully installed aiofiles-23.2.1 altair-5.5.0 annotated-doc-0.0.4 annotated-types-0.7.0 anyio-4.12.0 attrs-25.4.0 certifi-2025.11.12 charset_normalizer-3.4.4 click-8.3.1 contourpy-1.3.3 cycler-0.12.1 fastapi-0.128.0 ffmpy-1.0.0 filelock-3.20.1 fonttools-4.61.1 fsspec-2025.12.0 gradio-3.50.2 gradio-client-0.6.1 h11-0.16.0 httpcore-1.0.9 httpx-0.28.1 huggingface_hub-0.20.3 idna-3.11 importlib-resources-6.5.2 jinja2-3.1.6 joblib-1.3.2 jsonschema-4.25.1 jsonschema-specifications-2025.9.1 kiwisolver-1.4.9 markupsafe-2.1.5 matplotlib-3.9.2 narwhals-2.14.0 numpy-1.26.4 orjson-3.11.5 packaging-25.0 pandas-2.2.2 pillow-10.4.0 pydantic-2.12.5 pydantic-core-2.41.5 pydub-0.25.1 pyparsing-3.3.1 python-dateutil-2.9.0.post0 python-multipart-0.0.21 pytz-2025.2 pyyaml-6.0.3 referencing-0.37.0 requests-2.32.5 rpds-py-0.30.0 scikit-learn-1.3.2 scipy-1.16.3 seaborn-0.13.2 semantic-version-2.10.0 six-1.17.0 starlette-0.50.0 threadpoolctl-3.6.0 tqdm-4.67.1 typing-extensions-4.15.0 typing-inspection-0.4.2 tzdata-2025.3 urllib3-2.6.2 uvicorn-0.40.0 websockets-11.0.3
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv

[notice] A new release of pip is available: 24.0 -> 25.3
[notice] To update, run: pip install --upgrade pip
Removing intermediate container abd8e4c291cc
 ---> ac6a50571cb1
Step 5/7 : COPY . .
 ---> 7367bd1b3b8d
Step 6/7 : EXPOSE 7860
 ---> Running in 321c4057eecb
Removing intermediate container 321c4057eecb
 ---> 3ec4e723c3b3
Step 7/7 : CMD ["python", "-u", "app.py"]
 ---> Running in 4c86c387ebcf
Removing intermediate container 4c86c387ebcf
 ---> 16fe42d3ba3f
Successfully built 16fe42d3ba3f
Successfully tagged wholesale-gradio-app:latest
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ docker images
REPOSITORY                    TAG         IMAGE ID       CREATED         SIZE
wholesale-gradio-app          latest      16fe42d3ba3f   6 seconds ago   1.49GB
python                        3.11-slim   955f4ccb5624   2 days ago      124MB
gcr.io/k8s-minikube/kicbase   v0.0.48     c6b5532e987b   3 months ago    1.31GB
ngrok/ngrok                   latest      d0788c3e4266   56 years ago    162MB
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ docker run -p 7860:7860 wholesale-gradio-app
Running on local URL:  http://0.0.0.0:7860
IMPORTANT: You are using gradio version 3.50.2, however version 4.44.1 is available, please upgrade.
--------
Running on public URL: https://f490ed5b577f68b8e8.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)
/usr/local/lib/python3.11/site-packages/sklearn/base.py:465: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
  warnings.warn(
/usr/local/lib/python3.11/site-packages/sklearn/base.py:465: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
  warnings.warn(
/usr/local/lib/python3.11/site-packages/sklearn/base.py:465: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
  warnings.warn(
^CKeyboard interruption in main thread... closing server.
Killing tunnel 0.0.0.0:7860 <> https://f490ed5b577f68b8e8.gradio.live
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ eval $(minikube docker-env)
docker build -t wholesale-gradio-app .
[+] Building 73.2s (10/10) FINISHED                                                                                                                  
 => [internal] load build definition from Dockerfile                                                                                            0.0s
 => => transferring dockerfile: 271B                                                                                                            0.0s
 => [internal] load metadata for docker.io/library/python:3.11-slim                                                                             3.4s
 => [internal] load .dockerignore                                                                                                               0.0s
 => => transferring context: 2B                                                                                                                 0.0s
 => [1/5] FROM docker.io/library/python:3.11-slim@sha256:aa9aac8eacc774817e2881238f52d983a5ea13d7f5a1dee479a1a1d466047951                       7.4s
 => => resolve docker.io/library/python:3.11-slim@sha256:aa9aac8eacc774817e2881238f52d983a5ea13d7f5a1dee479a1a1d466047951                       0.0s
 => => sha256:6930196ac2d4c3a295bac0e7039dcc93b3038bd50f548b3a6a6af39a79cab132 1.75kB / 1.75kB                                                  0.0s
 => => sha256:955f4ccb562435365ab83ac6405b197bcfe93e778301cd0187bfb27d76f1ffa0 5.48kB / 5.48kB                                                  0.0s
 => => sha256:02d7611c4eae219af91448a4720bdba036575d3bc0356cfe12774af85daa6aff 29.78MB / 29.78MB                                                5.4s
 => => sha256:1473863bb010703cc5da49db594d480fa6a17a3a7df033dfacf0c91bf1162390 1.29MB / 1.29MB                                                  4.9s
 => => sha256:7cc13cb22d92060ccb92030698fd1bb23d1e7890328d424a3aa61c498ec1cb57 14.36MB / 14.36MB                                                6.9s
 => => sha256:aa9aac8eacc774817e2881238f52d983a5ea13d7f5a1dee479a1a1d466047951 10.37kB / 10.37kB                                                0.0s
 => => sha256:654a090213c05af530d397771fd09de5fe6e66c2355cd1a752c763b27ff451cf 249B / 249B                                                      5.5s
 => => extracting sha256:02d7611c4eae219af91448a4720bdba036575d3bc0356cfe12774af85daa6aff                                                       1.2s
 => => extracting sha256:1473863bb010703cc5da49db594d480fa6a17a3a7df033dfacf0c91bf1162390                                                       0.1s
 => => extracting sha256:7cc13cb22d92060ccb92030698fd1bb23d1e7890328d424a3aa61c498ec1cb57                                                       0.4s
 => => extracting sha256:654a090213c05af530d397771fd09de5fe6e66c2355cd1a752c763b27ff451cf                                                       0.0s
 => [internal] load build context                                                                                                               6.7s
 => => transferring context: 797.11MB                                                                                                           6.7s
 => [2/5] WORKDIR /app                                                                                                                          0.1s
 => [3/5] COPY requirements.txt .                                                                                                               0.0s
 => [4/5] RUN pip install --no-cache-dir -r requirements.txt                                                                                   53.6s
 => [5/5] COPY . .                                                                                                                              4.0s
 => exporting to image                                                                                                                          4.5s
 => => exporting layers                                                                                                                         4.5s
 => => writing image sha256:6df250cf412bc9591e9f0395de89e3bdbd3057f3772fe4101fb7ef1160418e7d                                                    0.0s
 => => naming to docker.io/library/wholesale-gradio-app:latest                                                                                  0.0s
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
deployment.apps/wholesale-segmentation created
service/wholesale-segmentation-service created
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ kubectl get pods
kubectl get services
NAME                                      READY   STATUS              RESTARTS   AGE
wholesale-segmentation-77dc596c68-vngx8   0/1     ContainerCreating   0          15s
NAME                             TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
kubernetes                       ClusterIP   10.96.0.1       <none>        443/TCP          10m
wholesale-segmentation-service   NodePort    10.111.124.23   <none>        7860:30081/TCP   14s
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ minikube service wholesale-gradio-service

❌  Exiting due to SVC_NOT_FOUND: Service 'wholesale-gradio-service' was not found in 'default' namespace.
You may select another namespace by using 'minikube service wholesale-gradio-service -n <namespace>'. Or list out all the services using 'minikube service list'

karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ minikube service wholesale-segmentation-service
┌───────────┬────────────────────────────────┬─────────────┬───────────────────────────┐
│ NAMESPACE │              NAME              │ TARGET PORT │            URL            │
├───────────┼────────────────────────────────┼─────────────┼───────────────────────────┤
│ default   │ wholesale-segmentation-service │ 7860        │ http://192.168.49.2:30081 │
└───────────┴────────────────────────────────┴─────────────┴───────────────────────────┘
🎉  Opening service default/wholesale-segmentation-service in default browser...
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ Opening in existing browser session.
^C
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ minikube service wholesale-segmentation-service
┌───────────┬────────────────────────────────┬─────────────┬───────────────────────────┐
│ NAMESPACE │              NAME              │ TARGET PORT │            URL            │
├───────────┼────────────────────────────────┼─────────────┼───────────────────────────┤
│ default   │ wholesale-segmentation-service │ 7860        │ http://192.168.49.2:30081 │
└───────────┴────────────────────────────────┴─────────────┴───────────────────────────┘
🎉  Opening service default/wholesale-segmentation-service in default browser...
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ Opening in existing browser session.
^C
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ git init
hint: Using 'master' as the name for the initial branch. This default branch name
hint: is subject to change. To configure the initial branch name to use in all
hint: of your new repositories, which will suppress this warning, call:
hint: 
hint:   git config --global init.defaultBranch <name>
hint: 
hint: Names commonly chosen instead of 'master' are 'main', 'trunk' and
hint: 'development'. The just-created branch can be renamed via this command:
hint: 
hint:   git branch -m <name>
Initialized empty Git repository in /home/karthika/Downloads/customer_segmentation/.git/
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ gid add .
Command 'gid' not found, but can be installed with:
sudo apt install id-utils
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ git  add .
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ git commit -m "first commit"
Author identity unknown

*** Please tell me who you are.

Run

  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"

to set your account's default identity.
Omit --global to set the identity only in this repository.

fatal: unable to auto-detect email address (got 'karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH.(none)')
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ git config --global user.name "KarthikaRajagopal44"
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ git config --global user.email "karthiizz.444@gmail.com"
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ git config --global --list
user.name=KarthikaRajagopal44
user.email=karthiizz.444@gmail.com
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ git commit -m "first commit"
[master (root-commit) feea07e] first commit
 14 files changed, 625 insertions(+)
 create mode 100644 .dockerignore
 create mode 100644 .gitignore
 create mode 100644 Dockerfile
 create mode 100644 README.md
 create mode 100644 app.py
 create mode 100644 deployment.yaml
 create mode 100644 flagged/log.csv
 create mode 100644 k8s.yaml
 create mode 100644 model.py
 create mode 100644 models/kmeans.pkl
 create mode 100644 models/scaler.pkl
 create mode 100644 requirements.txt
 create mode 100644 service.yaml
 create mode 100644 wholesale_customers_data.csv
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ git branch -M main
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ git remote add origin https://github.com/KarthikaRajagopal44/customer_segmentation_kmeans.git
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ git push -u origin main
Enumerating objects: 17, done.
Counting objects: 100% (17/17), done.
Delta compression using up to 8 threads
Compressing objects: 100% (14/14), done.
Writing objects: 100% (17/17), 11.45 KiB | 1.14 MiB/s, done.
Total 17 (delta 1), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (1/1), done.
To https://github.com/KarthikaRajagopal44/customer_segmentation_kmeans.git
 * [new branch]      main -> main
branch 'main' set up to track 'origin/main'.
karthika@karthika-ASUS-TUF-Gaming-F15-FX506LH-FX566LH:~/Downloads/customer_segmentation$ minikube service wholesale-segmentation-service
┌───────────┬────────────────────────────────┬─────────────┬───────────────────────────┐
│ NAMESPACE │              NAME              │ TARGET PORT │            URL            │
├───────────┼────────────────────────────────┼─────────────┼───────────────────────────┤
│ default   │ wholesale-segmentation-service │ 7860        │ http://192.168.49.2:30081 │
└───────────┴────────────────────────────────┴─────────────┴───────────────────────────┘
🎉  Opening service default/wholesale-segmentation-service in default browser...
