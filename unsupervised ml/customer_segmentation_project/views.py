import os
import json
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

class SegmentationEngine:
    def __init__(self):
        self.default_url = "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv"
        self.df = None
        self.X = None
        self.scaler = None
        self.kmeans_model = None
        self.y_clusters = None
        self.feature_indices = [3, 4]  # Annual Income (k$) and Spending Score (1-100)
        self.feature_names = ['Annual Income (k$)', 'Spending Score (1-100)']
        self.custom_df = None
        self.active_df = None
        self.is_custom = False

    def get_df(self):
        if self.active_df is not None:
            return self.active_df
        try:
            # Load default dataset
            self.df = pd.read_csv(self.default_url)
            self.active_df = self.df
            self.is_custom = False
        except Exception as e:
            print("Error loading raw URL dataset, using local fallback generation:", e)
            # Safe local mockup representing standard structure
            data = {
                'CustomerID': range(1, 201),
                'Gender': np.random.choice(['Male', 'Female'], 200),
                'Age': np.random.randint(18, 70, 200),
                'Annual Income (k$)': np.random.randint(15, 137, 200),
                'Spending Score (1-100)': np.random.randint(1, 100, 200)
            }
            self.df = pd.DataFrame(data)
            self.active_df = self.df
            self.is_custom = False
        return self.active_df

    def load_custom_csv(self, file_path):
        try:
            self.custom_df = pd.read_csv(file_path)
            self.active_df = self.custom_df
            self.is_custom = True
            
            # Autodetect numeric features for clustering
            cols = self.active_df.columns.tolist()
            income_idx = -1
            spend_idx = -1
            
            # Find common names
            for idx, col in enumerate(cols):
                col_lower = col.lower()
                if 'income' in col_lower or 'salary' in col_lower or 'earning' in col_lower:
                    income_idx = idx
                if 'spend' in col_lower or 'score' in col_lower or 'rating' in col_lower:
                    spend_idx = idx
            
            # If not found, select first two numerical columns
            num_cols = self.active_df.select_dtypes(include=[np.number]).columns.tolist()
            if income_idx == -1 and len(num_cols) > 0:
                income_idx = cols.index(num_cols[0])
            if spend_idx == -1 and len(num_cols) > 1:
                spend_idx = cols.index(num_cols[1])
            elif spend_idx == -1 and len(num_cols) > 0:
                spend_idx = cols.index(num_cols[0])
                
            if income_idx != -1 and spend_idx != -1 and income_idx != spend_idx:
                self.feature_indices = [income_idx, spend_idx]
            else:
                # Fallback to last two columns
                self.feature_indices = [len(cols)-2, len(cols)-1]
                
            self.feature_names = [cols[self.feature_indices[0]], cols[self.feature_indices[1]]]
            return True, None
        except Exception as e:
            return False, str(e)

    def reset_to_default(self):
        self.custom_df = None
        self.active_df = self.df
        self.feature_indices = [3, 4]
        self.feature_names = ['Annual Income (k$)', 'Spending Score (1-100)']
        self.is_custom = False

    def get_elbow_data(self):
        df = self.get_df()
        cols = df.columns
        f1 = cols[self.feature_indices[0]]
        f2 = cols[self.feature_indices[1]]
        
        X_features = df[[f1, f2]].dropna().values
        if len(X_features) < 10:
            max_k = len(X_features) + 1
        else:
            max_k = 11
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)
        
        wcss = []
        for k in range(1, max_k):
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(float(kmeans.inertia_))
            
        return {
            'wcss': wcss,
            'labels': list(range(1, max_k)),
            'feature_x': f1,
            'feature_y': f2
        }

    def train_kmeans(self, k):
        df = self.get_df()
        cols = df.columns
        f1 = cols[self.feature_indices[0]]
        f2 = cols[self.feature_indices[1]]
        self.feature_names = [f1, f2]
        
        # Drop rows with nulls in features
        df_clean = df.dropna(subset=[f1, f2]).copy()
        
        self.X = df_clean[[f1, f2]].values
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Ensure k is within bounds
        if k >= len(df_clean):
            k = max(2, len(df_clean) - 1)
            
        self.kmeans_model = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        self.y_clusters = self.kmeans_model.fit_predict(X_scaled)
        
        # Calculate silhouette score
        sil = 0.0
        if 2 <= k < len(df_clean):
            sil = float(silhouette_score(X_scaled, self.y_clusters))
            
        centroids = self.scaler.inverse_transform(self.kmeans_model.cluster_centers_)
        
        # Map clusters
        df_clean['Cluster'] = self.y_clusters
        
        cluster_info = []
        for i in range(k):
            sub_df = df_clean[df_clean['Cluster'] == i]
            mean_x = float(sub_df[f1].mean()) if len(sub_df) > 0 else 0.0
            mean_y = float(sub_df[f2].mean()) if len(sub_df) > 0 else 0.0
            
            # Calculate mean age if "Age" column exists
            mean_age = 0.0
            age_col = [c for c in df_clean.columns if 'age' in c.lower()]
            if age_col and len(sub_df) > 0:
                mean_age = float(sub_df[age_col[0]].mean())
                
            # Classify cluster
            min_x, max_x = float(df_clean[f1].min()), float(df_clean[f1].max())
            min_y, max_y = float(df_clean[f2].min()), float(df_clean[f2].max())
            mid_x = min_x + (max_x - min_x) / 2
            mid_y = min_y + (max_y - min_y) / 2
            
            desc = "Standard"
            if mean_x < mid_x and mean_y > mid_y:
                desc = "Low Income, High Spend (Careless)"
            elif mean_x > mid_x and mean_y > mid_y:
                desc = "High Income, High Spend (Target)"
            elif mean_x > mid_x and mean_y < mid_y:
                desc = "High Income, Low Spend (Conservative)"
            elif mean_x < mid_x and mean_y < mid_y:
                desc = "Low Income, Low Spend (Frugal)"
            else:
                desc = "Medium Income, Medium Spend (Standard)"
                
            cluster_info.append({
                'id': i,
                'name': f"Cluster {i}: {desc}",
                'short_name': desc,
                'count': int(len(sub_df)),
                'percentage': float(len(sub_df) / len(df_clean) * 100),
                'avg_x': mean_x,
                'avg_y': mean_y,
                'avg_age': mean_age
            })
            
        return {
            'silhouette_score': sil,
            'centroids': centroids.tolist(),
            'cluster_info': cluster_info,
            'y_clusters': self.y_clusters.tolist(),
            'clean_df_json': json.loads(df_clean.to_json(orient='records'))
        }

    def predict_customer(self, x_val, y_val):
        if self.kmeans_model is None or self.scaler is None:
            return None
        try:
            scaled_vals = self.scaler.transform([[x_val, y_val]])
            pred = int(self.kmeans_model.predict(scaled_vals)[0])
            return pred
        except Exception as e:
            print("Prediction error:", e)
            return None

# Instantiate global engine
engine = SegmentationEngine()

def index_view(request):
    """Renders the single-page application dashboard."""
    return render(request, 'index.html')

def get_data_api(request):
    """API endpoint to get the active dataset and features."""
    df = engine.get_df()
    cols = df.columns.tolist()
    
    # Calculate dataset stats
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats[col] = {
            'mean': float(df[col].mean()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
        }
        
    response_data = {
        'columns': cols,
        'is_custom': engine.is_custom,
        'feature_indices': engine.feature_indices,
        'feature_names': engine.feature_names,
        'stats': stats,
        'data_count': len(df)
    }
    return JsonResponse(response_data)

def get_elbow_api(request):
    """API endpoint to retrieve WCSS for K=1..10."""
    try:
        elbow_data = engine.get_elbow_data()
        return JsonResponse({'status': 'success', 'data': elbow_data})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

@csrf_exempt
def train_api(request):
    """API endpoint to train the K-Means clustering model."""
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            k = int(body.get('k', 5))
        except Exception:
            k = 5
    else:
        k = int(request.GET.get('k', 5))
        
    try:
        results = engine.train_kmeans(k)
        return JsonResponse({'status': 'success', 'data': results})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

@csrf_exempt
def predict_api(request):
    """API endpoint to predict segment for a custom customer."""
    try:
        if request.method == 'POST':
            body = json.loads(request.body)
            x_val = float(body.get('x'))
            y_val = float(body.get('y'))
        else:
            x_val = float(request.GET.get('x'))
            y_val = float(request.GET.get('y'))
            
        pred_cluster = engine.predict_customer(x_val, y_val)
        if pred_cluster is None:
            return JsonResponse({'status': 'error', 'message': 'Model not trained yet'}, status=400)
            
        return JsonResponse({'status': 'success', 'prediction': pred_cluster})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

@csrf_exempt
def upload_api(request):
    """API endpoint to upload custom CSV data."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST requests allowed'}, status=405)
        
    csv_file = request.FILES.get('file')
    if not csv_file:
        return JsonResponse({'status': 'error', 'message': 'No file uploaded'}, status=400)
        
    if not csv_file.name.endswith('.csv'):
        return JsonResponse({'status': 'error', 'message': 'Only CSV files are allowed'}, status=400)
        
    try:
        # Save file temporarily
        path = default_storage.save('tmp_custom_upload.csv', ContentFile(csv_file.read()))
        full_path = default_storage.path(path)
        
        # Load in engine
        success, err = engine.load_custom_csv(full_path)
        
        # Delete temporary file
        if os.path.exists(full_path):
            os.remove(full_path)
            
        if not success:
            return JsonResponse({'status': 'error', 'message': err}, status=400)
            
        # Re-fetch default features and rows
        df = engine.get_df()
        
        return JsonResponse({
            'status': 'success', 
            'message': 'CSV uploaded and parsed successfully!',
            'columns': df.columns.tolist(),
            'feature_names': engine.feature_names,
            'data_count': len(df)
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

@csrf_exempt
def reset_api(request):
    """API endpoint to reset custom uploaded data to default."""
    try:
        engine.reset_to_default()
        df = engine.get_df()
        return JsonResponse({
            'status': 'success', 
            'message': 'Reset to default dataset successful!',
            'columns': df.columns.tolist(),
            'feature_names': engine.feature_names,
            'data_count': len(df)
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
