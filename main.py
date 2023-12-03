from flask import Flask, request, jsonify
import random
import pandas as pd
import re

app = Flask(__name__)

# Your existing code

# Parsing 'Food Items'
def parse_food_items(food_items_str):
    items = food_items_str.split(',')
    parsed_items = []
    for item in items:
        match = re.search(r'([a-zA-Z\s]+)\((\d+)\s+(gm|ml|piece)\)', item.strip())
        if match:
            food_name = match.group(1).strip()
            quantity = int(match.group(2).strip())
            unit = match.group(3).strip()
            parsed_items.append((food_name, quantity, unit))
    return parsed_items

def aggregate_quantities(parsed_items):
    # Example conversion factors (these are approximations and should be refined)
    average_weight_per_piece = 40  # grams per piece, as an average estimate
    ml_to_grams_conversion = 0.7      # 1 ml approximately equals 1 gram
    total_grams = 0
    for item in parsed_items:
        quantity, unit = item[1], item[2]
        if unit == 'gm':
            total_grams += quantity
        elif unit == 'ml':
            # Convert milliliters to grams (assuming 1 ml = 0.7 gram)
            total_grams += quantity * ml_to_grams_conversion
        elif unit == 'piece':
            # Convert pieces to grams using the average weight per piece
            total_grams += quantity * average_weight_per_piece
    return total_grams

def parse_user_input(food_category, food_items_str, pre_meal_blood_sugar, feature_names):
    parsed_items = parse_food_items(food_items_str)
    total_food_quantity = aggregate_quantities(parsed_items)

    # Initialize a data point with default values for all features
    data_point = {feature: 0 for feature in feature_names}

    # Update the relevant features based on user input
    data_point['Total Food Quantity'] = total_food_quantity
    data_point['Pre Meal Blood Sugar'] = pre_meal_blood_sugar
    meal_type_feature = f'Meal Type_{food_category}'
    if meal_type_feature in data_point:
        data_point[meal_type_feature] = 1

    return data_point

def prepare_meal_data_point(meal, pre_meal_blood_sugar, feature_names):
    # Initialize a data point with default values for all features
    meal_data_point = {feature: 0 for feature in feature_names}

    # Update the pre-meal blood sugar level
    meal_data_point['Pre Meal Blood Sugar'] = pre_meal_blood_sugar

    # Check the format of 'meal' and process accordingly
    if isinstance(meal, str):
        # If 'meal' is a string, parse it into a list of tuples (food_item, quantity, unit)
        parsed_meal = parse_food_items(meal)  # Assuming parse_food_items can handle this format
    else:
        # If 'meal' is already in the expected format (list of tuples)
        parsed_meal = meal

    # Update the total food quantity based on the meal
    total_quantity = sum([quantity for _, quantity, _ in parsed_meal])
    meal_data_point['Total Food Quantity'] = total_quantity

    # Additional logic for other features (if necessary)

    return meal_data_point


def adjust_quantity_of_food_items(meal, adjustment_factor):
    # Split the meal string into individual food items
    food_items = meal.split(', ')

    # Adjust the quantity of each food item
    adjusted_food_items = []
    for food_item in food_items:
        # Split the food item into name and quantity
        parts = food_item.rsplit(' ', 1)
        name = parts[0]
        
        if len(parts) == 2:
            quantity_str = parts[1]
            if 'piece' in quantity_str.lower():
                # Leave the quantity unchanged for items with "piece"
                adjusted_food_item = food_item
            else:
                # Clean up the quantity string
                quantity_str = quantity_str.replace('gm', '').replace('ml', '').replace(')', '')

                # Try converting to float after cleaning up
                try:
                    quantity = float(quantity_str)
                    # Adjust the quantity based on the factor
                    adjusted_quantity = quantity * adjustment_factor
                    # Add the adjusted food item to the list
                    adjusted_food_item = f"{name} {adjusted_quantity:.2f} gm"
                except ValueError:
                    # Handle the case where conversion to float fails
                    adjusted_food_item = food_item
        else:
            # No quantity specified, leave it unchanged
            adjusted_food_item = food_item

        # Add the adjusted food item to the list
        adjusted_food_items.append(adjusted_food_item)

    # Join the adjusted food items back into a string
    adjusted_meal = ', '.join(adjusted_food_items)

    return adjusted_meal

def get_recommendation_and_suggest_next_meal(food_category, food_items_str, pre_meal_blood_sugar, model, healthy_range, meal_database, feature_names):
    # Prepare the current meal data point
    current_meal_data_point = parse_user_input(food_category, food_items_str, pre_meal_blood_sugar, feature_names)

    # Predict post-meal blood sugar level for the current meal
    current_meal_data_df = pd.DataFrame([current_meal_data_point])
    predicted_post_meal_blood_sugar = model.predict(current_meal_data_df)[0]

    # Determine the next meal type based on the current meal category
    if food_category == "Breakfast":
        next_meal_type = "Lunch"
    elif food_category == "Lunch":
        next_meal_type = "Snacks"
    elif food_category == "Snacks":
        next_meal_type = "Dinner"
    else:  # Assuming current meal is Dinner
        next_meal_type = "Breakfast"

    # Get all potential meals for the next meal type
    potential_next_meals = meal_database[next_meal_type]

    # Generate meal suggestions for the next meal
    meal_suggestions = []
    for meal in potential_next_meals:
        # Prepare data point for the suggested meal
        suggested_meal_data_point = prepare_meal_data_point(meal, pre_meal_blood_sugar, feature_names)
        suggested_meal_data_df = pd.DataFrame([suggested_meal_data_point])

        # Predict blood sugar level for the suggested meal
        predicted_blood_sugar_next_meal = model.predict(suggested_meal_data_df)[0]

        # Check if the predicted level is within the healthy range
        if healthy_range[0] <= predicted_blood_sugar_next_meal <= healthy_range[1]:
            meal_suggestions.append((meal, predicted_blood_sugar_next_meal))

    # Categorize meal suggestions based on predicted blood sugar levels
    low_sugar_meals = [(meal, sugar) for meal, sugar in meal_suggestions if sugar < healthy_range[0] + 5]  # Adjust threshold as needed
    moderate_sugar_meals = [(meal, sugar) for meal, sugar in meal_suggestions if
                             healthy_range[0] + 5 <= sugar <= healthy_range[1] - 10]
    high_sugar_meals = [(meal, sugar) for meal, sugar in meal_suggestions if sugar > healthy_range[1] - 10]  # Adjust threshold as needed

    # Randomly select one menu from each category for display
    selected_low_sugar_meal = random.choice(low_sugar_meals) if low_sugar_meals else None

    # Adjust quantity for moderate sugar meal
    selected_moderate_sugar_meals = random.sample(moderate_sugar_meals, min(2, len(moderate_sugar_meals)))
    adjusted_moderate_sugar_meals = []

    for meal, sugar in selected_moderate_sugar_meals:
        # Reduce the quantity of food items to fall under the low blood sugar level category
        adjusted_quantity = 0.6  # You can adjust this factor based on your preference
        adjusted_meal = adjust_quantity_of_food_items(meal, adjusted_quantity)  # Implement this function
        adjusted_moderate_sugar_meals.append((adjusted_meal, sugar))

    # Adjust quantity for high sugar meal
    selected_high_sugar_meal = random.choice(high_sugar_meals) if high_sugar_meals else None

    if selected_high_sugar_meal:
        # Increase the quantity of food items to fall under the high blood sugar level category
        adjusted_quantity = 1.2  # You can adjust this factor based on your preference
        adjusted_high_sugar_meal = adjust_quantity_of_food_items(selected_high_sugar_meal[0], adjusted_quantity)  # Implement this function
    else:
        adjusted_high_sugar_meal = None

    # Print the results
    output_str =""
    # print(f"Predicted Post-Meal Blood Sugar: {predicted_post_meal_blood_sugar}")
    output_str = "Predicted Post-Meal Blood Sugar: "+  str(predicted_post_meal_blood_sugar) +" "+ '\n'
    # print("Next Meal Suggestions:")
    output_str = output_str + " Next Meal Suggestions: " + "\n"
    if selected_low_sugar_meal:
        # print(f"Significantly Low Sugar Meal: {next_meal_type}, ({selected_low_sugar_meal[0]} | Predicted Blood Sugar: {selected_low_sugar_meal[1]})")
        output_str = output_str + "Significantly Low Sugar Meal: " + str(next_meal_type) + " "+ str(selected_low_sugar_meal[0]) + " "+ "Predicted Blood Sugar: "+" "+ str(selected_low_sugar_meal[1]) + "\n"
    for meal, sugar in adjusted_moderate_sugar_meals:
        # print(f"Moderate Sugar Meal: {next_meal_type}, ({meal} | Predicted Blood Sugar: {sugar})")
        output_str = output_str + "Moderate Sugar Meal: " +str(next_meal_type)+ " "+ str(meal) + " "+ "| "+ "Predicted Blood Sugar: "+ str(sugar)+" "+"\n"

    if adjusted_high_sugar_meal:
        # print(f"High Sugar Meal: {next_meal_type}, ({adjusted_high_sugar_meal} | Predicted Blood Sugar: {selected_high_sugar_meal[1]})")
        output_str = output_str + "High Sugar Meal: "+ str(next_meal_type)+ " , " +str(adjusted_high_sugar_meal)+ " "+ "| "+ "Predicted Blood Sugar: "+ str(selected_high_sugar_meal[1])

    return output_str
def extract_unique_meals(meal_data):
    meal_database = {}
    for meal_type in ['Breakfast', 'Lunch', 'Snacks', 'Dinner']:
        unique_meals = meal_data[meal_data['Meal Type'] == meal_type]['Food Items'].unique()
        meal_database[meal_type] = unique_meals
    return meal_database


# API endpoint
@app.route('/predict_meal', methods=['POST'])
def predict_meal():
    import joblib

    # Load the model from the serialized file
    model_file_path = 'best_random_forest_model.joblib'  # Update with the actual path
    with open(model_file_path, 'rb') as file:
        loaded_model = joblib.load(file)
    food_data = pd.read_csv("food_data.csv")
    # Creating the meal database
    meal_database = extract_unique_meals(food_data)
    data = request.json

    # Example user input
    food_category = data.get('food_category', 'Breakfast')
    food_items_str = data.get('food_items_str', 'Poha (200 gm), Tea (100 ml)')
    pre_meal_blood_sugar = data.get('pre_meal_blood_sugar', 110)
    healthy_blood_sugar_range = data.get('healthy_blood_sugar_range', [90, 130])

        # Your specified values
    feature_type_values = ['Pre Meal Blood Sugar', 'Meal Type_Breakfast', 'Meal Type_Dinner',
                            'Meal Type_Lunch', 'Meal Type_Snacks', 'Total Food Quantity']

    # Create a pandas Index with the specified values
    feature_names = pd.Index(feature_type_values)
    # Test the function
    output_str = get_recommendation_and_suggest_next_meal(
        food_category, food_items_str, pre_meal_blood_sugar, loaded_model, healthy_blood_sugar_range, meal_database, feature_names
    )
    print(output_str)
    return jsonify({"output_str": output_str})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)