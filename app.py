from flask import Flask, render_template, request, jsonify
from datetime import datetime
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io
import os
import mysql.connector
from mysql.connector import Error

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
yolo_model = YOLO('best.pt')
mobilenet_model = tf.keras.models.load_model('mobilenet_best_model.h5')

# Database config
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Divsroch@6988',
    'database': 'recipe_db'
}

# Standardized class name mapping
CLASS_NAME_MAPPING = {
    'avocado': 'avocado',
    'beans': 'beans',
    'beet': 'beetroot',
    'bell pepper': 'bell pepper',
    'broccoli': 'broccoli',
    'brus capusta': 'brussels sprouts',
    'cabbage': 'cabbage',
    'carrot': 'carrot',
    'cayliflower': 'cauliflower',
    'celery': 'celery',
    'corn': 'corn',
    'cucumber': 'cucumber',
    'eggplant': 'eggplant',
    'fasol': 'green beans',
    'garlic': 'garlic',
    'hot pepper': 'chili pepper',
    'onion': 'onion',
    'peas': 'peas',
    'potato': 'potato',
    'pumpkin': 'pumpkin',
    'rediska': 'radish',
    'redka': 'radish',
    'salad': 'lettuce',
    'squash-patisson': 'pattypan squash',
    'tomato': 'tomato'
}

# MobileNet class labels
MOBILENET_CLASSES = [
    'freshapples', 'freshbanana', 'freshbittergroud', 'freshcapsicum', 
    'freshcucumber', 'freshokra', 'freshoranges', 'freshpotato', 'freshtomato',
    'rottenapples', 'rottenbanana', 'rottenbittergroud', 'rottencapsicum',
    'rottencucumber', 'rottenokra', 'rottenoranges', 'rottenpotato', 'rottentomato'
]

# Recipe search mapping (simple names)
RECIPE_NAME_MAP = {
    'avocado': 'avocado',
    'beans': 'beans',
    'beetroot': 'beet',
    'bell pepper': 'capsicum',
    'broccoli': 'broccoli',
    'brussels sprouts': 'brussels sprouts',
    'cabbage': 'cabbage',
    'carrot': 'carrot',
    'cauliflower': 'cauliflower',
    'celery': 'celery',
    'corn': 'corn',
    'cucumber': 'cucumber',
    'eggplant': 'eggplant',
    'green beans': 'green beans',
    'garlic': 'garlic',
    'chili pepper': 'chili pepper',
    'onion': 'onion',
    'peas': 'peas',
    'potato': 'potato',
    'pumpkin': 'pumpkin',
    'radish': 'radish',
    'lettuce': 'lettuce',
    'pattypan squash': 'squash',
    'tomato': 'tomato'
}

# Composting category mapping
COMPOSTING_NAME_MAP = {
    'avocado': 'Fruits with Pits',
    'beans': 'Legumes',
    'beetroot': 'Root Vegetables',
    'bell pepper': 'Nightshades',
    'broccoli': 'Brassicas',
    'brussels sprouts': 'Brassicas',
    'cabbage': 'Brassicas',
    'carrot': 'Root Vegetables',
    'cauliflower': 'Brassicas',
    'celery': 'Fibrous Vegetables',
    'corn': 'Starchy Vegetables',
    'cucumber': 'Watery Vegetables',
    'eggplant': 'Nightshades',
    'green beans': 'Legumes',
    'garlic': 'Allium Family',
    'chili pepper': 'Nightshades',
    'onion': 'Allium Family',
    'peas': 'Legumes',
    'potato': 'Root Vegetables',
    'pumpkin': 'Starchy Vegetables',
    'radish': 'Root Vegetables',
    'lettuce': 'Leafy Greens',
    'pattypan squash': 'Starchy Vegetables',
    'tomato': 'Nightshades'
}

def create_db_connection():
    """Create and return a database connection."""
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        print(f"Database connection error: {e}")
        return None
def get_recipes_by_ingredient(vegetable):
    """Fetch recipes containing the specified vegetable."""
    recipe_search_term = RECIPE_NAME_MAP.get(vegetable.lower(), vegetable.lower())
    print(f"Searching recipes for: {recipe_search_term}")
    connection = create_db_connection()
    if not connection:
        print("Failed to connect to database")
        return []
    try:
        cursor = connection.cursor(dictionary=True)
        # 1. Recipes where vegetable is in the recipe name
        query_name_match = """
            SELECT DISTINCT recipe_name, image_url, recipe_link, ingredients 
            FROM recipes_hq 
            WHERE LOWER(recipe_name) LIKE %s
        """
        cursor.execute(query_name_match, (f'%{recipe_search_term}%',))
        name_matches = cursor.fetchall()

        # 2. Recipes where vegetable is in ingredients but NOT in recipe name
        query_ingredient_match = """
            SELECT DISTINCT recipe_name, image_url, recipe_link, ingredients 
            FROM recipes_hq 
            WHERE LOWER(ingredients) LIKE %s 
            AND LOWER(recipe_name) NOT LIKE %s
        """
        cursor.execute(query_ingredient_match, (f'%{recipe_search_term}%', f'%{recipe_search_term}%'))
        ingredient_matches = cursor.fetchall()

        # Combine results: recipe_name matches first, then ingredient matches
        recipes = name_matches + ingredient_matches

        print(f"Found {len(recipes)} recipes for {recipe_search_term}")
        return recipes
    except Error as e:
        print(f"Recipe fetch error: {e}")
        return []
    finally:
        connection.close()


def get_composting_info(vegetable):
    """Fetch composting information for the specified vegetable."""
    composting_category = COMPOSTING_NAME_MAP.get(vegetable.lower(), vegetable.lower())
    print(f"Searching composting info for category: {composting_category}")

    connection = create_db_connection()
    if not connection:
        return None

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT recommended_method, reason 
            FROM vegetable_composting 
            WHERE LOWER(vegetable_type) LIKE %s
        """, (f"%{composting_category.lower()}%",))  # Add wildcards for partial match

        return cursor.fetchone()
    except Error as e:
        print(f"Composting info fetch error: {e}")
        return None
    finally:
        connection.close()


def preprocess_image_for_mobilenet(img, target_size=(128, 128)):
    #Preprocess image for MobileNet model.
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis=0) / 255.0

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html', now=datetime.now())

@app.route('/recipe-finder', methods=['GET', 'POST'])
def recipe_finder():
    """Handle image upload and return results."""
    if request.method == 'GET':
        return render_template('recipe_finder.html', now=datetime.now())

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Process image
        img = Image.open(io.BytesIO(file.read()))
        
        # YOLO object detection
        yolo_results = yolo_model(img)
        detected_vegetables = set()
        
        for result in yolo_results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                veg_name = yolo_model.names[class_id]
                standardized_name = CLASS_NAME_MAPPING.get(veg_name, veg_name)
                detected_vegetables.add(standardized_name)

        detected_vegetables = list(detected_vegetables)
        print("Detected vegetables:", detected_vegetables)

        # MobileNet freshness prediction
        processed_img = preprocess_image_for_mobilenet(img)
        predictions = mobilenet_model.predict(processed_img)[0]
        max_index = np.argmax(predictions)
        freshness_label = MOBILENET_CLASSES[max_index]
        is_fresh = freshness_label.startswith('fresh')

        # Prepare results
        results = {
            'is_fresh': is_fresh,
            'detected_vegetables': detected_vegetables,
            'freshness_info': {
                'freshness_level': freshness_label,
                'confidence': float(predictions[max_index])
            }
        }

        # Get recommendations based on freshness
        if is_fresh and detected_vegetables:
            recipes = []
            for veg in detected_vegetables:
                veg_recipes = get_recipes_by_ingredient(veg)
                recipes.extend(veg_recipes)
            results['recommendations'] = recipes
            
        elif detected_vegetables:
            composting_info = {}
            for veg in detected_vegetables:
                info = get_composting_info(veg)
                if info:
                    composting_info[veg] = info
            results['composting_info'] = composting_info

        return render_template('results.html', results=results, now=datetime.now())

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': f'Image processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)