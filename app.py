import gradio as gr
import numpy as np
#function to predict cluster
import joblib


km = joblib.load("models/kmeans.pkl")
scaler = joblib.load("models/scaler.pkl")
def predict_cluster(fresh, milk, grocery, frozen, detergents_paper, delicassen):
    # Prepare input
    data= np.array([[fresh, milk, grocery, frozen, detergents_paper, delicassen]])
    data_scaled= scaler.transform(data)
#predict cluster
    cluster= km.predict(data_scaled)
    return f"customer belongs to the Cluster{cluster[0]}"
#gradio interface
inputs= [
    gr.Number(label="Fresh"),
    gr.Number(label="Milk"),
    gr.Number(label="Grocery"),
    gr.Number(label="Frozen"),
    gr.Number(label="Detergents_Paper"),
    gr.Number(label="Delicassen")
]

examples = [
    [12000, 3000, 8000, 2000, 2500, 1000],   # Bulk essentials buyer
    [3000, 8000, 14000, 900, 6000, 2000],    # Luxury gourmet shopper
    [500, 200, 300, 4000, 150, 100],         # Frozen-heavy Horeca client
]


outputs= gr.Textbox(label="Cluster result")
demo= gr.Interface(fn= predict_cluster, inputs=inputs, outputs=outputs, examples=examples, title= "Wholesale Customer Segmentation")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
