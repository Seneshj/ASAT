# app.py - Flask server that loads your trained GMM model and predicts attention profiles
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and scaler
gmm_model = None
scaler = None
model_metadata = None

# Feature names in the exact order used during training
FEATURE_NAMES = [
    'CPT_RT_mean',
    'CPT_RT_std', 
    'CPT_CV',
    'CPT_omission',
    'CPT_commission',
    'CPT_dprime',
    'CPT_sustained_slope',
    'Stroop_interference',
    'Flanker_effect'
]

# Attention group interpretations based on cluster analysis from your training
# These should be updated based on your actual cluster characteristics
ATTENTION_GROUPS = {
    0: {
        'name': 'Cluster 0: Slow Processor',
        'description': 'Slower reaction times across all tasks. May benefit from processing speed exercises.',
        'recommendation': 'Engage in timed problem-solving activities and processing speed tasks.',
        'color': '#9C27B0'
    },
    1: {
        'name': 'Cluster 1: High Performer',
        'description': 'Excellent attention control! Fast processing, high sensitivity, and minimal interference.',
        'recommendation': 'Maintain current cognitive habits. You\'re performing optimally!',
        'color': '#2196F3'
    },
    2: {
        'name': 'Cluster 2: Balanced Attender',
        'description': 'Average performance across all metrics. Within normal range for attention control.',
        'recommendation': 'Continue with current attention maintenance strategies.',
        'color': '#4CAF50'
    }
    # Add more clusters based on your optimal_clusters value
}


def load_models():
    """Load pre-trained GMM model and scaler"""
    global gmm_model, scaler, model_metadata
    
    model_path = "gmm_model.pkl"
    scaler_path = "scaler.pkl"
    metadata_path = "model_metadata.pkl"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f" ERROR: Model file not found at {model_path}")
        return False
    
    if not os.path.exists(scaler_path):
        print(f" ERROR: Scaler file not found at {scaler_path}")
        return False
    
    try:
        # Load GMM model
        gmm_model = joblib.load(model_path)
        print(f" GMM Model loaded: {type(gmm_model).__name__}")
        print(f"   Number of clusters: {gmm_model.n_components}")
        
        # Load scaler
        scaler = joblib.load(scaler_path)
        print(f" Scaler loaded: {type(scaler).__name__}")
        
        # Load metadata if available
        if os.path.exists(metadata_path):
            model_metadata = joblib.load(metadata_path)
            print(f" Metadata loaded: {model_metadata.get('optimal_clusters', 'N/A')} clusters")
            
            # Update ATTENTION_GROUPS based on metadata if available
            if 'cluster_profiles' in model_metadata:
                print("   Cluster profiles available in metadata")
        
        # Dynamically add attention groups for all clusters
        n_clusters = gmm_model.n_components
        for i in range(n_clusters):
            if i not in ATTENTION_GROUPS:
                ATTENTION_GROUPS[i] = {
                    'name': f'Cluster {i}',
                    'description': f'Attention profile based on GMM cluster {i}',
                    'recommendation': 'Complete more tasks for personalized recommendations.',
                    'color': '#9E9E9E'
                }
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        traceback.print_exc()
        return False


def validate_features(features_dict):
    """Validate that all required features are present"""
    missing = []
    for feature in FEATURE_NAMES:
        if feature not in features_dict or features_dict[feature] is None:
            missing.append(feature)
    return missing


def predict_attention_group(features_dict):
    """
    Predict attention group using pre-trained GMM model
    """
    global gmm_model, scaler
    
    if gmm_model is None or scaler is None:
        return {
            'success': False,
            'error': 'Model not loaded. Please ensure model files exist.'
        }
    
    try:
        # Validate features
        missing = validate_features(features_dict)
        if missing:
            return {
                'success': False,
                'error': f'Missing features: {missing}'
            }
        
        # Extract features in correct order
        features = []
        for feature in FEATURE_NAMES:
            value = float(features_dict[feature])
            features.append(value)
        
        # Convert to numpy array
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features using pre-trained scaler
        features_scaled = scaler.transform(features_array)
        
        # Predict cluster using GMM
        cluster = gmm_model.predict(features_scaled)[0]
        
        # Get cluster probabilities
        probabilities = gmm_model.predict_proba(features_scaled)[0]
        
        # Get confidence (highest probability)
        confidence = float(max(probabilities))
        
        # Get attention group info
        group_info = ATTENTION_GROUPS.get(cluster, {
            'name': f'Cluster {cluster}',
            'description': 'Attention profile based on your performance.',
            'recommendation': 'Complete more tasks for personalized recommendations.',
            'color': '#9E9E9E'
        })
        
        # Prepare response
        result = {
            'success': True,
            'cluster_id': int(cluster),
            'attention_group': group_info['name'],
            'description': group_info['description'],
            'recommendation': group_info['recommendation'],
            'color': group_info['color'],
            'confidence': round(confidence, 4),
            'probabilities': {
                f'cluster_{i}': round(float(p), 4) 
                for i, p in enumerate(probabilities)
            },
            'metrics_received': {
                name: float(features_dict[name]) 
                for name in FEATURE_NAMES
            }
        }
        
        return result
        
    except Exception as e:
        print(f" Prediction error: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for attention metrics prediction
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        print(f"\n Received metrics:")
        for key in FEATURE_NAMES:
            if key in data:
                print(f"   {key}: {data[key]}")
        
        result = predict_attention_group(data)
        
        if result['success']:
            print(f" Predicted: {result['attention_group']} (confidence: {result['confidence']:.2%})")
        else:
            print(f" Prediction failed: {result.get('error', 'Unknown error')}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f" Request error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if gmm_model and scaler else 'degraded',
        'model_loaded': gmm_model is not None,
        'scaler_loaded': scaler is not None,
        'n_clusters': gmm_model.n_components if gmm_model else None,
        'features_expected': FEATURE_NAMES
    })


@app.route('/groups', methods=['GET'])
def get_groups():
    """Get information about available attention groups"""
    groups_info = []
    for cluster_id, info in ATTENTION_GROUPS.items():
        groups_info.append({
            'cluster_id': cluster_id,
            'name': info['name'],
            'description': info['description'],
            'color': info['color']
        })
    
    return jsonify({
        'success': True,
        'groups': groups_info,
        'n_clusters': gmm_model.n_components if gmm_model else None,
        'features_expected': FEATURE_NAMES
    })


@app.route('/metadata', methods=['GET'])
def get_metadata():
    """Get model metadata"""
    if model_metadata:
        return jsonify({
            'success': True,
            'metadata': model_metadata
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Metadata not available'
        })


if __name__ == '__main__':
    print(" ")
    print(" Attention Profile Prediction Server")
    print(" ")
    
    print("\n Loading trained models...")
    models_loaded = load_models()
    
    if models_loaded:
        print("\n Server ready to receive predictions!")
        print(f"   GMM has {gmm_model.n_components} attention clusters")
        print(f"   Features expected: {len(FEATURE_NAMES)} metrics")
    else:
        print("\n Could not load models. Please ensure:")
        print("   1. 'gmm_model.pkl' exists in the current directory")
        print("   2. 'scaler.pkl' exists in the current directory")
        print("   3. Both files were saved correctly during training")
    
    print(" ")
    print(" Starting Flask server...")
    print(f" Prediction endpoint: http://localhost:5000/predict")
    print(f" Health check: http://localhost:5000/health")
    print(f" Groups info: http://localhost:5000/groups")
    print(" ")
    print("\n Ready! Open your HTML file and click 'GET ATTENTION PROFILE'\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
