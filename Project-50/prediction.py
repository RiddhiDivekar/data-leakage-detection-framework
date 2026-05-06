import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# ------------------ LOAD MODEL ------------------
with open("Random_Forest_Model.pkl", "rb") as f:
    model = pickle.load(f)

# ------------------ LOAD DATASET AND CREATE ENCODERS ------------------
# Read your dataset
df = pd.read_csv("multi_tenant_data_leakage.csv")  



user_data = df['user'].unique()







import numpy as np

# Original array
relationships = np.array(user_data)

# Remove leading or trailing whitespace and single quotes for consistent matching
cleaned_relationships = [
    str(r).strip().strip("'") if isinstance(r, str) else str(r).strip() 
    for r in relationships
    if not (isinstance(r, float) and np.isnan(r))  # skip NaN values
]

sorted_relationships = sorted(set(cleaned_relationships))

label_mapping = {relationship: idx for idx, relationship in enumerate(sorted_relationships)}

def get_relationship_value(chosen_relationship):
    cleaned_input = chosen_relationship.strip().strip("'")
    return label_mapping.get(cleaned_input, "Not found")









# Example usage
chosen_relationship = 'User_0208' 
rel_int = get_relationship_value(chosen_relationship)


















# # # Create label encoders for categorical columns
# # user_encoder = LabelEncoder()
# # pc_encoder = LabelEncoder()
# # authority_encoder = LabelEncoder()
# # external_encoder = LabelEncoder()

# # # Fit encoders with data from your dataset
# # user_encoder.fit(df['user'].dropna())
# # pc_encoder.fit(df['pc'].dropna())
# # authority_encoder.fit(df['Authority'].dropna())
# # external_encoder.fit(df['External Destination'].dropna())

# print("Available User values:", list(user_encoder.classes_)[:20], "...")
# print("Available PC values:", list(pc_encoder.classes_)[:20], "...")
# print("Available Authority values:", list(authority_encoder.classes_))
# print("Available External Destination values:", list(external_encoder.classes_))

# ------------------ USER INPUT WITH BETTER VALIDATION ------------------
def get_valid_input(prompt, valid_options=None, input_type=str, allow_new=False):
    """Get validated user input with better error handling"""
    while True:
        try:
            value = input(prompt).strip()
            
            if input_type == int:
                value = int(value)
                if value not in [0, 1] and '0/1' in prompt:
                    print("Please enter either 0 or 1")
                    continue
                return value
            else:
                # For string inputs
                if valid_options is not None and value not in valid_options:
                    if allow_new:
                        print(f"Warning: '{value}' not in training data. Using new encoding.")
                        return value
                    else:
                        print(f"Invalid input. Please choose from: {list(valid_options)[:20]}...")
                        continue
                return value
                
        except ValueError:
            print("Please enter a valid value")

# ------------------ ENCODING FUNCTION ------------------
def encode_value(encoder, value, default_for_new=None):
    """Encode a value, handling new/unseen values"""
    try:
        return encoder.transform([value])[0]
    except ValueError:
        # Handle new/unseen values
        if default_for_new is not None:
            return default_for_new
        else:
            # Assign next available number for new values
            return len(encoder.classes_)

# Get user inputs
print("\n--- Enter Input Values ---")
user_val = get_valid_input("Enter user: ", valid_options=user_encoder.classes_, allow_new=True)
pc_val = get_valid_input("Enter PC: ", valid_options=pc_encoder.classes_, allow_new=True)
auth_val = get_valid_input("Enter Authority (manager/senior manager/staff): ", valid_options=authority_encoder.classes_, allow_new=True)

through_pwd = get_valid_input("Through password (0/1): ", input_type=int)
through_pin = get_valid_input("Through PIN (0/1): ", input_type=int)
through_mfa = get_valid_input("Through MFA (0/1): ", input_type=int)
data_mod = get_valid_input("Data Modification (0/1): ", input_type=int)
conf_access = get_valid_input("Confidential Data Access (0/1): ", input_type=int)
conf_transfer = get_valid_input("Confidential File Transfer (0/1): ", input_type=int)

ext_dest = get_valid_input("External Destination (internal/external): ", valid_options=external_encoder.classes_, allow_new=True)

container_id = get_valid_input("Container_ID: ", input_type=int)
req_cpu = get_valid_input("Requested_CPU: ", input_type=int)
req_mem = get_valid_input("Requested_Memory_MB: ", input_type=int)
req_storage = get_valid_input("Requested_Storage_GB: ", input_type=int)
exec_start = get_valid_input("Execution_Start_Time_ms: ", input_type=int)
exec_finish = get_valid_input("Execution_Finish_Time_ms: ", input_type=int)
makespan = get_valid_input("Makespan_ms: ", input_type=int)
total_data = get_valid_input("Total_Data_Transferred_MB: ", input_type=int)

# ------------------ AUTO ENCODING ------------------
# Encode categorical variables
user_encoded = encode_value(user_encoder, user_val)
pc_encoded = encode_value(pc_encoder, pc_val)
auth_encoded = encode_value(authority_encoder, auth_val)
ext_dest_encoded = encode_value(external_encoder, ext_dest)

print(f"\n--- Encoded Values ---")
print(f"User: {user_val} -> {user_encoded}")
print(f"PC: {pc_val} -> {pc_encoded}")
print(f"Authority: {auth_val} -> {auth_encoded}")
print(f"External Destination: {ext_dest} -> {ext_dest_encoded}")

# ------------------ CREATE INPUT DATA ------------------
input_dict = {
    'user': user_encoded,
    'pc': pc_encoded,
    'Authority': auth_encoded,
    'Through_pwd': through_pwd,
    'Through_pin': through_pin,
    'Through_MFA': through_mfa,
    'Data Modification': data_mod,
    'Confidential Data Access': conf_access,
    'Confidential File Transfer': conf_transfer,
    'External Destination': ext_dest_encoded,
    'Container_ID': container_id,
    'Requested_CPU': req_cpu,
    'Requested_Memory_MB': req_mem,
    'Requested_Storage_GB': req_storage,
    'Execution_Start_Time_ms': exec_start,
    'Execution_Finish_Time_ms': exec_finish,
    'Makespan_ms': makespan,
    'Total_Data_Transferred_MB': total_data
}

input_df = pd.DataFrame([input_dict])

# ------------------ PREDICTION ------------------
pred = model.predict(input_df)[0]
result_text = "Data Leakage DETECTED!" if pred == 1 else "No Data Leakage"
print(f"\n--- Prediction Result ---")
print("Result:", result_text)