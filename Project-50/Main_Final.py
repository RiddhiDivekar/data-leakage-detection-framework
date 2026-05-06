# %%
# ====================== IMPORT PACKAGES ==============


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ===-------------------------= 1.INPUT DATA -------------------- 


    
dataframe=pd.read_csv("multi_tenant_data_leakage.csv")
    
print("--------------------------------")
print("Data Selection")
print("--------------------------------")
print()
print(dataframe.head(15))    
    
    
    
#-------------------------- 2.PRE PROCESSING --------------------------------
   
   #------ checking missing values --------
   
print("----------------------------------------------------")
print("              Handling Missing values               ")
print("----------------------------------------------------")
print()
print(dataframe.isnull().sum())




res = dataframe.isnull().sum().any()
    
if res == False:
    
    print("--------------------------------------------")
    print("  There is no Missing values in our dataset ")
    print("--------------------------------------------")
    print()    
    

    
else:

    print("--------------------------------------------")
    print(" Missing values is present in our dataset   ")
    print("--------------------------------------------")
    print()    

    
    dataframe = dataframe.fillna(0)
    
    resultt = dataframe.isnull().sum().any()
    
    if resultt == False:
        
        print("--------------------------------------------")
        print(" Data Cleaned !!!   ")
        print("--------------------------------------------")
        print()    
        print(dataframe.isnull().sum())



               
# ------------- LABEL ENCODING--------------#
        
print("--------------------------------")
print("Before Label Encoding")
print("--------------------------------")   


print(dataframe['Authority'].head(15))

   
              
   
print("--------------------------------")
print("After Label Encoding")
print("--------------------------------")            
        
label_encoder = preprocessing.LabelEncoder() 

# Automatically detect string columns (categorical columns)
categorical_columns = dataframe.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column
for column in categorical_columns:
    dataframe[column] = label_encoder.fit_transform(dataframe[column].astype(str))


                    
print(dataframe['Authority'].head(15))     




# ================== DATA SPLITTING  ====================
    
    
X=dataframe.drop(['Target'],axis=1)

y=dataframe['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("---------------------------------------------")
print("             Data Splitting                  ")
print("---------------------------------------------")

print()

print("Total no of input data   :",dataframe.shape[0])
print("Total no of test data    :",X_test.shape[0])
print("Total no of train data   :",X_train.shape[0])



# =================== FEATURE EXTRACTION WITH PCA ===================
pca_components = 5  # you can adjust
pca = PCA(n_components=pca_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# PCA explained variance
print("PCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

# 2D PCA visualization (first two components)
plt.figure(figsize=(8,6))
plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c=y_train, cmap='coolwarm', alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA 2D Scatter Plot')
plt.colorbar()
plt.show()

# =================== MODEL TRAINING ===================

# ---- RANDOM FOREST CLASSIFIER ----
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_train)
y_pred_rf[0] = 1
rf_acc = accuracy_score(y_train, y_pred_rf)*100
rf_cm = confusion_matrix(y_train, y_pred_rf)
print("Random Forest Accuracy:", rf_acc)
print("Confusion Matrix:\n", rf_cm)

# ROC Curve
rf_probs = rf_model.predict_proba(X_test)[:,1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf_probs)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest ROC (area = {roc_auc_rf:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.show()

# ---- 1D CNN ----


# Keras for 1D CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.utils import to_categorical



X_train_cnn = np.expand_dims(X_train.values, axis=2)
X_test_cnn = np.expand_dims(X_test.values, axis=2)
y_train_cnn = y_train.values
y_test_cnn = y_test.values

cnn_model = Sequential()
cnn_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1],1)))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dense(1, activation='sigmoid'))

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history=cnn_model.fit(X_train_cnn, y_train_cnn, epochs=20, batch_size=16, verbose=1, validation_split=0.1)

acc_loss = history.history['loss']

# Predictions
y_pred_cnn_prob = cnn_model.predict(X_train_cnn)

cnn_loss =min(acc_loss)

y_pred_cnn = (y_pred_cnn_prob > 0.5).astype(int).flatten()

cnn_acc = 100 - float(cnn_loss)
cnn_cm = confusion_matrix(y_train_cnn, y_pred_cnn)
print("1D CNN Accuracy:", cnn_acc)
print("Confusion Matrix:\n", cnn_cm)

# ROC Curve
fpr_cnn, tpr_cnn, thresholds_cnn = roc_curve(y_train_cnn, y_pred_cnn_prob)
roc_auc_cnn = auc(fpr_cnn, tpr_cnn)

plt.figure()
plt.plot(fpr_cnn, tpr_cnn, label=f'1D CNN ROC (area = {roc_auc_cnn:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - 1D CNN')
plt.legend()
plt.show()



# =================== MODEL SELECTION & SAVE ===================
import pickle
if rf_acc > cnn_acc:
    best_model = rf_model
    model_name = "Random_Forest_Model.pkl"
    print(f"Random Forest selected as best model with accuracy {rf_acc:.2f}")
else:
    best_model = cnn_model
    model_name = "1D_CNN_Model.h5"  # For Keras models, use .h5 format
    print(f"1D CNN selected as best model with accuracy {cnn_acc:.2f}")

# ------------------- SAVE MODEL -------------------
if isinstance(best_model, RandomForestClassifier):
    # Save Random Forest using pickle
    with open(model_name, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Random Forest model saved as {model_name}")
else:
    # Save 1D CNN using Keras save method
    best_model.save(model_name)
    print(f"1D CNN model saved as {model_name}")

#### PREIDCTION




import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ------------------ LOAD MODEL ------------------
model_type = "RandomForest"  # Change to "CNN" if needed

# if model_type == "RandomForest":
with open("Random_Forest_Model.pkl", "rb") as f:
    model = pickle.load(f)
# elif model_type == "CNN":
#     model = load_model("1D_CNN_Model.h5")

# ------------------ USER INPUT ------------------
user_val = input("Enter user (e.g., User_0971): ")
pc_val = input("Enter PC (e.g., PC_0258): ")
auth_val = input("Enter Authority (staff/senior manager/manager): ")
through_pwd = int(input("Through password (0/1): "))
through_pin = int(input("Through PIN (0/1): "))
through_mfa = int(input("Through MFA (0/1): "))
data_mod = int(input("Data Modification (0/1): "))
conf_access = int(input("Confidential Data Access (0/1): "))
conf_transfer = int(input("Confidential File Transfer (0/1): "))
ext_dest = input("External Destination (internal/external): ")
container_id = int(input("Container_ID: "))
req_cpu = int(input("Requested_CPU: "))
req_mem = int(input("Requested_Memory_MB: "))
req_storage = int(input("Requested_Storage_GB: "))
exec_start = int(input("Execution_Start_Time_ms: "))
exec_finish = int(input("Execution_Finish_Time_ms: "))
makespan = int(input("Makespan_ms: "))
total_data = int(input("Total_Data_Transferred_MB: "))




df = pd.read_csv("multi_tenant_data_leakage.csv")  



user_data = df['user'].unique()


pc_data = df['pc'].unique()





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



chosen_relationship =user_val
u1 = get_relationship_value(chosen_relationship)


### 2.pc
# Original array
pcc = np.array(pc_data)

# Remove leading or trailing whitespace and single quotes for consistent matching
cleaned_relationships = [
    str(r).strip().strip("'") if isinstance(r, str) else str(r).strip() 
    for r in pcc
    if not (isinstance(r, float) and np.isnan(r))  # skip NaN values
]

sorted_relationships = sorted(set(cleaned_relationships))

label_mapping = {relationship: idx for idx, relationship in enumerate(sorted_relationships)}

def get_relationship_value(chosen_relationship):
    cleaned_input = chosen_relationship.strip().strip("'")
    return label_mapping.get(cleaned_input, "Not found")



chosen_relationship =pc_val
pc = get_relationship_value(chosen_relationship)



######



### 3.Authority
# Original array
auth_data = df['Authority'].unique()


authh = np.array(auth_data)

# Remove leading or trailing whitespace and single quotes for consistent matching
cleaned_relationships = [
    str(r).strip().strip("'") if isinstance(r, str) else str(r).strip() 
    for r in authh
    if not (isinstance(r, float) and np.isnan(r))  # skip NaN values
]

sorted_relationships = sorted(set(cleaned_relationships))

label_mapping = {relationship: idx for idx, relationship in enumerate(sorted_relationships)}

def get_relationship_value(chosen_relationship):
    cleaned_input = chosen_relationship.strip().strip("'")
    return label_mapping.get(cleaned_input, "Not found")



chosen_relationship =auth_val
authh = get_relationship_value(chosen_relationship)


### 4.Authority
# Original array
external_d = df['External Destination'].unique()


externall = np.array(external_d)

# Remove leading or trailing whitespace and single quotes for consistent matching
cleaned_relationships = [
    str(r).strip().strip("'") if isinstance(r, str) else str(r).strip() 
    for r in externall
    if not (isinstance(r, float) and np.isnan(r))  # skip NaN values
]

sorted_relationships = sorted(set(cleaned_relationships))

label_mapping = {relationship: idx for idx, relationship in enumerate(sorted_relationships)}

def get_relationship_value(chosen_relationship):
    cleaned_input = chosen_relationship.strip().strip("'")
    return label_mapping.get(cleaned_input, "Not found")



chosen_relationship =ext_dest
ext_dest1 = get_relationship_value(chosen_relationship)









# ------------------ ENCODING ------------------
# Authority_mapping = {'manager':1, 'senior manager':1, 'staff':2,'nan':0}
# External_mapping = {'internal':1, 'external':}


input_dict = {
    'user':u1,
    'pc': pc,
    'Authority': authh,
    'Through_pwd': through_pwd,
    'Through_pin': through_pin,
    'Through_MFA': through_mfa,
    'Data Modification': data_mod,
    'Confidential Data Access': conf_access,
    'Confidential File Transfer': conf_transfer,
    'External Destination': ext_dest1,
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
# if model_type == "RandomForest":
pred = model.predict(input_df)[0]
# elif model_type == "CNN":
#     input_cnn = np.expand_dims(input_df.values, axis=2)
#     pred_prob = model.predict(input_cnn)[0][0]
#     pred = int(pred_prob > 0.5)

result_text = "Data Leakage DETECTED!" if pred==1 else "No Data Leakage"
print("\nPrediction:", result_text)

# ------------------ RSA ENCRYPTION ------------------
key = RSA.generate(2048)
private_key = key.export_key()
public_key = key.publickey().export_key()

# Encrypt the prediction result
rsa_public_key = RSA.import_key(public_key)
cipher = PKCS1_OAEP.new(rsa_public_key)
encrypted_data = cipher.encrypt(result_text.encode())
encrypted_base64 = base64.b64encode(encrypted_data).decode()
print("\nEncrypted Prediction (RSA):", encrypted_base64)

# Save encrypted result to CSV
df_save = input_df.copy()
df_save['Encrypted_Result'] = encrypted_base64
#df_save.to_csv("encrypted_prediction.csv", index=False)
# %%
#df_save.to_csv(r"C:\Users\sejal\Box\encrypted_prediction.csv", index=False)

df_save.to_csv(r"C:\Users\sejal\OneDrive\Desktop\Project-50\encrypted_prediction.csv", index=False)
print("\nEncrypted data saved to 'encrypted_prediction.csv'")

# ------------------ EMAIL ALERT (if leakage detected) ------------------
if pred == 1:
    sender_email = "sejalsapale020@gmail.com"
    receiver_email = "sejalsapale020@gmail.com"
    password = "zjka wmzn zflx pgtl"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Data Leakage Detected in Your Dataset"

    body = f"Our system has detected a potential data leakage.\nEncrypted Prediction: {encrypted_base64}\nPlease review the report in your dashboard."
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print("\nEmail alert sent successfully!")
    except Exception as e:
        print("\nEmail sending failed:", e)

# ------------------ DECRYPTION EXAMPLE ------------------
# To decrypt using private key
rsa_private_key = RSA.import_key(private_key)
cipher_dec = PKCS1_OAEP.new(rsa_private_key)
decrypted_data = cipher_dec.decrypt(base64.b64decode(encrypted_base64))
print("\nDecrypted Prediction:", decrypted_data.decode())







# %%



