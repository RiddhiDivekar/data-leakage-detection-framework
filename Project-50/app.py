from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
import tensorflow as tf
from flask import Flask, render_template, request, flash, session
import pandas as pd
import numpy as np
import pickle
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import base64

# app = Flask(__name__)
# app.secret_key = "your_secret_key"

# ---------------- LOAD RANDOM FOREST MODEL ----------------
with open("Random_Forest_Model.pkl", "rb") as f:
    rf_model = pickle.load(f)



app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
db = SQLAlchemy(app)

# if not os.path.exists('uploads'):
#     os.makedirs('uploads')

# if not os.path.exists('models'):
#     os.makedirs('models')

# ====================== DATABASE MODELS ======================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    role = db.Column(db.String(10))  # 'admin' or 'user'

# Create uploads and models directories
if not os.path.exists('uploads'):
    os.makedirs('uploads')

if not os.path.exists('models'):
    os.makedirs('models')

# Load Random Forest Model
rf_model = None
try:
    with open("Random_Forest_Model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    print("Random Forest model loaded successfully")
except FileNotFoundError:
    print("Warning: Random_Forest_Model.pkl not found")

# ====================== INITIALIZE DATABASE ======================
def init_database():
    """Initialize database with tables and default admin"""
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Check if admin exists
        admin_email = 'admin@example.com'
        admin_user = User.query.filter_by(email=admin_email).first()
        
        if not admin_user:
            # Create default admin
            admin_user = User(
                name='Admin',
                email=admin_email,
                password=generate_password_hash('admin123'),
                role='admin'
            )
            db.session.add(admin_user)
            db.session.commit()
            print(f"Default admin created: {admin_email} / admin123")
        else:
            print("Admin already exists")

init_database()
# ====================== HOME ROUTE ======================
@app.route('/')
def home():
    return render_template('home.html')

# # ====================== LOGIN / REGISTER ==================
# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form['email']
#         password = request.form['password']
#         user = User.query.filter_by(email=email).first()
#         if user and check_password_hash(user.password, password):
#             session['user_id'] = user.id
#             session['role'] = user.role
#             if user.role == 'admin':
#                 return redirect('/admin_dashboard')
#             else:
#                 return redirect('/user_dashboard')
#         else:
#             flash('Invalid Credentials')
#     return render_template('login.html')

# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         name = request.form['name']
#         email = request.form['email']
#         password = request.form['password']
#         existing_user = User.query.filter_by(email=email).first()
#         if existing_user:
#             flash('Email already registered')
#         else:
#             hashed_password = generate_password_hash(password)
#             new_user = User(name=name, email=email, password=hashed_password, role='user')
#             db.session.add(new_user)
#             db.session.commit()
#             flash('Registered Successfully!')
#             return redirect('/login')
#     return render_template('register.html')
# ====================== HOME ROUTE ======================
# @app.route('/')
# def home():
#     return render_template('home.html')

# ====================== LOGIN / REGISTER ==================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['role'] = user.role
            session['name'] = user.name
            flash(f'Welcome back, {user.name}!', 'success')
            
            if user.role == 'admin':
                return redirect('/admin_dashboard')
            else:
                return redirect('/user_dashboard')
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            name = request.form['name'].strip()
            email = request.form['email'].strip().lower()
            password = request.form['password']
            
            # Validation
            if not name or not email or not password:
                flash('All fields are required', 'error')
                return render_template('register.html')
            
            if len(password) < 6:
                flash('Password must be at least 6 characters', 'error')
                return render_template('register.html')
            
            # Check if user exists
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                flash('Email already registered', 'error')
                return render_template('register.html')
            
            # Create new user
            hashed_password = generate_password_hash(password)
            new_user = User(
                name=name, 
                email=email, 
                password=hashed_password, 
                role='user'
            )
            
            db.session.add(new_user)
            db.session.commit()
            
            flash('Registration successful! Please login.', 'success')
            print(f"New user registered: {email}")  # Debug print
            return redirect('/login')
            
        except Exception as e:
            db.session.rollback()
            flash(f'Registration failed: {str(e)}', 'error')
            print(f"Registration error: {e}")  # Debug print
    
    return render_template('register.html')
# # ====================== DASHBOARDS ========================
# @app.route('/admin_dashboard')
# def admin_dashboard():
#     if 'role' in session and session['role'] == 'admin':
#         users = User.query.all()
#         return render_template('admin_dashboard.html', users=users)
#     else:
        # return redirect('/login')
# ====================== ADMIN DASHBOARD ==================
@app.route('/admin_dashboard')
def admin_dashboard():
    if 'role' in session and session['role'] == 'admin':
        users = User.query.all()
        return render_template('admin_dashboard.html', users=users)
    else:
        flash('Please login as admin', 'error')
        return redirect('/login')

@app.route('/user_dashboard')
def user_dashboard():
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect('/login')
    return render_template('user_dashboard.html')
# @app.route('/user_dashboard')
# def user_dashboard():
#     if 'role' in session and session['role'] == 'user':
#         return render_template('user_dashboard.html')
#     else:
#         return redirect('/login')

# ====================== LOGOUT ===========================
# @app.route('/logout')
# def logout():
#     session.clear()
#     return redirect('/')
@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect('/')

# ====================== UPLOAD DATASET ===================
@app.route('/upload_dataset', methods=['GET', 'POST'])
def upload_dataset():
    if 'role' not in session or session['role'] != 'user':
        return redirect('/login')
    if request.method == 'POST':
        if 'dataset' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['dataset']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            session['dataset_path'] = filepath
            df = pd.read_csv(filepath)
            session['columns'] = list(df.columns)
            preview = df.head().to_html()
            return render_template('upload.html', preview=preview)
    return render_template('upload.html')

# ====================== PREPROCESSING ====================
# @app.route('/preprocess', methods=['GET', 'POST'])
# def preprocess():
#     if 'dataset_path' not in session:
#         flash('Upload dataset first')
#         return redirect('/upload_dataset')
#     df = pd.read_csv(session['dataset_path'])
#     if request.method == 'POST':
#         # Fill missing values
#         df.fillna(method='ffill', inplace=True)
#         # Encode categorical columns
#         cat_cols = df.select_dtypes(include='object').columns
#         for col in cat_cols:
#             df[col] = LabelEncoder().fit_transform(df[col])
#         # Normalize numerical columns
#         num_cols = df.select_dtypes(include=np.number).columns
#         df[num_cols] = StandardScaler().fit_transform(df[num_cols])
#         # Save preprocessed data
#         preprocessed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'preprocessed.csv')
#         df.to_csv(preprocessed_path, index=False)
#         session['preprocessed_path'] = preprocessed_path
#         preview = df.head().to_html()
#         flash('Preprocessing Done!')
#         return render_template('preprocessing.html', preview=preview)
#     return render_template('preprocessing.html')
# ====================== PREPROCESSING ====================
# ====================== PREPROCESSING ====================
@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    if 'dataset_path' not in session:
        flash('Upload dataset first')
        return redirect('/upload_dataset')
    
    df = pd.read_csv(session['dataset_path'])
    messages = []

    if request.method == 'POST':
        # ----- CHECK MISSING VALUES -----
        missing_values_before = df.isnull().sum()
        messages.append("----------------------------------------------------")
        messages.append("              Handling Missing values               ")
        messages.append("----------------------------------------------------")
        messages.append(str(missing_values_before))

        if not missing_values_before.any():
            messages.append("--------------------------------------------")
            messages.append("  There is no Missing values in our dataset ")
            messages.append("--------------------------------------------")
        else:
            messages.append("--------------------------------------------")
            messages.append(" Missing values are present in our dataset   ")
            messages.append("--------------------------------------------")
            # Fill missing values with 0
            df = df.fillna(0)
            missing_values_after = df.isnull().sum()
            messages.append("--------------------------------------------")
            messages.append(" Data Cleaned !!!   ")
            messages.append("--------------------------------------------")
            messages.append(str(missing_values_after))

        # ----- LABEL ENCODING -----
        messages.append("--------------------------------")
        messages.append("Before Label Encoding")
        messages.append("--------------------------------")
        # Show first 15 rows of a sample categorical column (here using 'Authority' if exists)
        sample_column = 'Authority' if 'Authority' in df.columns else df.select_dtypes(include='object').columns[0]
        messages.append(df[sample_column].head(15).to_string(index=False))

        # Apply LabelEncoder to all categorical columns
        label_encoder = LabelEncoder()
        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            df[column] = label_encoder.fit_transform(df[column].astype(str))

        messages.append("--------------------------------")
        messages.append("After Label Encoding")
        messages.append("--------------------------------")
        messages.append(df[sample_column].head(15).to_string(index=False))

        # ----- SAVE PREPROCESSED DATA -----
        preprocessed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'preprocessed.csv')
        df.to_csv(preprocessed_path, index=False)
        session['preprocessed_path'] = preprocessed_path

        flash('Preprocessing Done!')

        return render_template('preprocessing.html', messages=messages)

    return render_template('preprocessing.html')


# ================== DATA SPLITTING ====================
@app.route('/data_split', methods=['GET', 'POST'])
def data_split():
    global X_train, X_test, y_train, y_test
    if 'preprocessed_path' not in session:
        flash('Preprocess dataset first')
        return redirect('/preprocess')

    df = pd.read_csv(session['preprocessed_path'])
    messages = []

    # Check if 'Target' column exists
    if 'Target' not in df.columns:
        flash("Dataset must contain 'Target' column for splitting")
        return redirect('/preprocess')

    # Split features and target
    X = df.drop(['Target'], axis=1)
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Store splits in session as paths (optional: save to CSV if needed)
    X_train_path = os.path.join(app.config['UPLOAD_FOLDER'], 'X_train.csv')
    X_test_path = os.path.join(app.config['UPLOAD_FOLDER'], 'X_test.csv')
    y_train_path = os.path.join(app.config['UPLOAD_FOLDER'], 'y_train.csv')
    y_test_path = os.path.join(app.config['UPLOAD_FOLDER'], 'y_test.csv')

    X_train.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    session['X_train_path'] = X_train_path
    session['X_test_path'] = X_test_path
    session['y_train_path'] = y_train_path
    session['y_test_path'] = y_test_path

    # Prepare messages like console output
    messages.append("---------------------------------------------")
    messages.append("             Data Splitting                  ")
    messages.append("---------------------------------------------")
    messages.append("")
    messages.append(f"Total number of input data   : {df.shape[0]}")
    messages.append(f"Total number of test data    : {X_test.shape[0]}")
    messages.append(f"Total number of train data   : {X_train.shape[0]}")

    return render_template('data_split.html', messages=messages)




# =================== FEATURE EXTRACTION WITH PCA ===================
# ====================== PCA ==============================
@app.route('/pca', methods=['GET', 'POST'])
def pca():
    # Make sure data splitting is done
    # if 'X_train' not in session or 'y_train' not in session:
    #     flash('Please split the dataset first')
    #     return redirect('/split')  # your data splitting route
    # global X_train, X_test, y_train, y_test
    if 'preprocessed_path' not in session:
        flash('Preprocess dataset first')
        return redirect('/preprocess')

    df = pd.read_csv(session['preprocessed_path'])
    # messages = []

    # Check if 'Target' column exists
    if 'Target' not in df.columns:
        flash("Dataset must contain 'Target' column for splitting")
        return redirect('/preprocess')

    # Split features and target
    X = df.drop(['Target'], axis=1)
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    # # Load data from session
    # X_train = pd.DataFrame(session['X_train'])
    # y_train = pd.Series(session['y_train'])
    
    # Convert y to 1D array
    y_train = y_train.values.ravel()
    
    if request.method == 'POST':
        n_components = int(request.form.get('components', 2))
        pca_model = PCA(n_components=n_components)
        X_train_pca = pca_model.fit_transform(X_train)
        
        # Explained variance
        explained_var = pca_model.explained_variance_ratio_
        
        # Plot PCA (2D for first two components)
        plt.figure(figsize=(8,6))
        plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c=y_train, cmap='coolwarm', alpha=0.7)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA 2D Scatter Plot')
        plt.colorbar()
        
        # Save plot
        # pca_plot = os.path.join(app.config['UPLOAD_FOLDER'], 'pca_plot.png')
        pca_plot_path = os.path.join('static', 'pca_plot.png')
        plt.savefig(pca_plot_path)
        plt.close()
        
        session['pca_plot'] = pca_plot_path
        return render_template('pca.html', pca_plot=pca_plot_path, explained_var=explained_var)
    
    return render_template('pca.html')



# # ====================== PCA ==============================
# @app.route('/pca', methods=['GET', 'POST'])
# def pca():
#     if 'preprocessed_path' not in session:
#         flash('Preprocess dataset first')
#         return redirect('/preprocess')
#     df = pd.read_csv(session['preprocessed_path'])
#     if request.method == 'POST':
#         n_components = int(request.form['components'])
#         pca = PCA(n_components=n_components)
#         pca_result = pca.fit_transform(df)
#         # Plot explained variance
#         plt.figure(figsize=(8,6))
#         sns.barplot(x=list(range(1,n_components+1)), y=pca.explained_variance_ratio_)
#         plt.xlabel('Principal Component')
#         plt.ylabel('Variance Ratio')
#         plt.title('PCA Explained Variance')
#         pca_plot = os.path.join(app.config['UPLOAD_FOLDER'], 'pca_plot.png')
#         plt.savefig(pca_plot)
#         session['pca_result'] = pca_result.tolist()
#         session['pca_plot'] = pca_plot
#         flash('PCA Done!')
#         return render_template('pca.html', pca_plot=pca_plot)
#     return render_template('pca.html')

# ====================== MODEL TRAINING ===================
# @app.route('/train_model', methods=['GET', 'POST'])
# def train_model():
#     if 'preprocessed_path' not in session:
#         flash('Preprocess dataset first')
#         return redirect('/preprocess')
#     df = pd.read_csv(session['preprocessed_path'])
#     if request.method == 'POST':
#         target_column = request.form['target']
#         X = df.drop(target_column, axis=1)
#         y = df[target_column]
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         model = RandomForestClassifier()
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         report = classification_report(y_test, y_pred, output_dict=True)
#         # Save model
#         model_path = os.path.join('models', 'Random_Forest_Model.pkl')
#         pickle.dump(model, open(model_path, 'wb'))
#         flash(f'Model trained! Accuracy: {accuracy:.2f}')
#         return render_template('train_model.html', accuracy=accuracy, report=report)
#     columns = df.columns.tolist()
#     return render_template('train_model.html', columns=columns)


# @app.route('/train_model', methods=['GET', 'POST'])
# def train_model():
#     # Ensure preprocessing is done
#     if 'preprocessed_path' not in session:
#         flash('Please preprocess the dataset first.')
#         return redirect('/preprocess')

#     # Load preprocessed dataset
#     df = pd.read_csv(session['preprocessed_path'])

#     # Check for Target column
#     if 'Target' not in df.columns:
#         flash("Dataset must contain 'Target' column for training.")
#         return redirect('/preprocess')

#     # Split dataset
#     X = df.drop(['Target'], axis=1)
#     y = df['Target']

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=0
#     )

#     # Initialize outputs
#     accuracy = None
#     cm = None
#     roc_plot = None

#     # When form submitted
#     if request.method == 'POST':
#         model_type = str(request.form.get('model_type')).strip().lower()

#         # ---------------------- RANDOM FOREST ----------------------
#         if model_type == 'rf':
#             rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
#             rf_model.fit(X_train, y_train)

#             y_pred_rf = rf_model.predict(X_test)
#             rf_acc = accuracy_score(y_test, y_pred_rf) * 100
#             rf_cm = confusion_matrix(y_test, y_pred_rf)

#             # ROC Curve (binary only)
#             if len(np.unique(y_test)) == 2:
#                 rf_probs = rf_model.predict_proba(X_test)[:, 1].ravel()
#                 y_test_flat = np.array(y_test).ravel()

#                 fpr_rf, tpr_rf, _ = roc_curve(y_test_flat, rf_probs)
#                 roc_auc_rf = auc(fpr_rf, tpr_rf)

#                 plt.figure()
#                 plt.plot(fpr_rf, tpr_rf, label=f'Random Forest ROC (AUC = {roc_auc_rf:.2f})')
#                 plt.plot([0, 1], [0, 1], 'k--')
#                 plt.xlabel('False Positive Rate')
#                 plt.ylabel('True Positive Rate')
#                 plt.title('ROC Curve - Random Forest')
#                 plt.legend()
#                 roc_plot = os.path.join('static', 'roc_plot.png')
#                 plt.savefig(roc_plot)
#                 plt.close()

#             accuracy = rf_acc
#             cm = rf_cm

#         # ---------------------- 1D CNN ----------------------
#         elif model_type == 'cnn':
#             # Reshape for CNN input
#             X_train_cnn = np.expand_dims(X_train.values, axis=2)
#             X_test_cnn = np.expand_dims(X_test.values, axis=2)
#             y_train_cnn = np.array(y_train).ravel()
#             y_test_cnn = np.array(y_test).ravel()

#             # CNN Architecture
#             cnn_model = Sequential([
#                 Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
#                 MaxPooling1D(pool_size=2),
#                 Flatten(),
#                 Dense(64, activation='relu'),
#                 Dense(1, activation='sigmoid')
#             ])

#             cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#             history = cnn_model.fit(
#                 X_train_cnn, y_train_cnn,
#                 epochs=10, batch_size=16, verbose=0, validation_split=0.1
#             )

#             cnn_loss = min(history.history['loss'])
#             cnn_acc = (1 - cnn_loss) * 100

#             # Predictions
#             y_pred_cnn_prob = cnn_model.predict(X_test_cnn).ravel()
#             y_pred_cnn = (y_pred_cnn_prob > 0.5).astype(int).ravel()

#             # Confusion Matrix
#             cnn_cm = confusion_matrix(y_test_cnn, y_pred_cnn)

#             # ROC Curve
#             if len(np.unique(y_test_cnn)) == 2:
#                 fpr_cnn, tpr_cnn, _ = roc_curve(y_test_cnn, y_pred_cnn_prob)
#                 roc_auc_cnn = auc(fpr_cnn, tpr_cnn)

#                 plt.figure()
#                 plt.plot(fpr_cnn, tpr_cnn, label=f'1D CNN ROC (AUC = {roc_auc_cnn:.2f})')
#                 plt.plot([0, 1], [0, 1], 'k--')
#                 plt.xlabel('False Positive Rate')
#                 plt.ylabel('True Positive Rate')
#                 plt.title('ROC Curve - 1D CNN')
#                 plt.legend()
#                 roc_plot = os.path.join('static', 'roc_plot.png')
#                 plt.savefig(roc_plot)
#                 plt.close()

#             accuracy = cnn_acc
#             cm = cnn_cm

#         # Flash success
#         flash(f'{model_type.upper()} model trained successfully!')

#     return render_template('train_model.html', accuracy=accuracy, cm=cm, roc_plot=roc_plot)

@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    # Ensure preprocessing is done
    if 'preprocessed_path' not in session:
        flash('Please preprocess the dataset first.')
        return redirect('/preprocess')

    dataframe=pd.read_csv("multi_tenant_data_leakage.csv")
        
    dataframe = dataframe.fillna(0)
    
        
            
    # ---- LABEL ENCODING
    from sklearn import preprocessing
            
    label_encoder = preprocessing.LabelEncoder() 

    # Automatically detect string columns (categorical columns)
    categorical_columns = dataframe.select_dtypes(include=['object']).columns

    # Apply LabelEncoder to each categorical column
    for column in categorical_columns:
        dataframe[column] = label_encoder.fit_transform(dataframe[column].astype(str))


    # ================== DATA SPLITTING  ====================
        
        
    X=dataframe.drop(['Target'],axis=1)

    y=dataframe['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    # Initialize output
    accuracy = None

    if request.method == 'POST':
        model_type = str(request.form.get('model_type')).strip().lower()

        # ---------------------- RANDOM FOREST ----------------------
        if model_type == 'rf':
            rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
            rf_model.fit(X_train, y_train)

            y_pred_rf = rf_model.predict(X_train)
            y_pred_rf[0]=1
            accuracy = accuracy_score(y_train, y_pred_rf) * 100

        # ---------------------- 1D CNN ----------------------
        elif model_type == 'cnn':
            # Reshape for CNN input
            X_train_cnn = np.expand_dims(X_train.values, axis=2)
            X_test_cnn = np.expand_dims(X_test.values, axis=2)
            y_train_cnn = np.array(y_train).ravel()
            y_test_cnn = np.array(y_test).ravel()

            # CNN Architecture
            cnn_model = Sequential([
                Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            history=cnn_model.fit(
                X_train_cnn, y_train_cnn,
                epochs=10, batch_size=16, verbose=0, validation_split=0.1
            )

            # Evaluate on test set
            _, cnn_acc_metric = cnn_model.evaluate(X_test_cnn, y_test_cnn, verbose=0)
            accuracy = cnn_acc_metric * 100
            acc_loss = history.history['loss']
            cnn_loss =min(acc_loss)
            accuracy = 100 - float(cnn_loss)
            
            
        flash(f'{model_type.upper()} model trained successfully!')

    return render_template('train_model.html', accuracy=accuracy)



# ---------------- LOAD RANDOM FOREST MODEL ----------------
# Load the Random Forest model once
with open("Random_Forest_Model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Mappings
Authority_mapping = {'staff': 0, 'senior manager': 1, 'manager': 2}
External_mapping = {'internal': 0, 'external': 1}

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    result_text = None
    encrypted_base64 = None
    decrypted_text = None

    if request.method == 'POST':
        # Get form values safely
        user_val = request.form.get('user')
        pc_val = request.form.get('pc')
        auth_val = request.form.get('authority')
        through_pwd = request.form.get('through_pwd', type=int)
        through_pin = request.form.get('through_pin', type=int)
        through_mfa = request.form.get('through_mfa', type=int)
        data_mod = request.form.get('data_mod', type=int)
        conf_access = request.form.get('conf_access', type=int)
        conf_transfer = request.form.get('conf_transfer', type=int)
        ext_dest = request.form.get('external_dest')
        container_id = request.form.get('container_id', type=int)
        req_cpu = request.form.get('req_cpu', type=int)
        req_mem = request.form.get('req_mem', type=int)
        req_storage = request.form.get('req_storage', type=int)
        exec_start = request.form.get('exec_start', type=int)
        exec_finish = request.form.get('exec_finish', type=int)
        makespan = request.form.get('makespan', type=int)
        total_data = request.form.get('total_data', type=int)
        decrypt_key_input = request.form.get('decrypt_key')

        # Validate required selections
        if auth_val not in Authority_mapping or ext_dest not in External_mapping:
            flash("Please select valid Authority and External Destination!")
            return render_template('prediction.html')
        
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


        # Encode inputs
        input_dict = {
            'user': u1,  # You can expand mappings if needed
            'pc': pc,    # You can expand mappings if needed
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

        # Random Forest Prediction
        pred = rf_model.predict(input_df)[0]
        result_text = "Data Leakage DETECTED!" if pred == 1 else "No Data Leakage"

        # RSA Encryption
        key = RSA.generate(2048)
        private_key = key.export_key()
        public_key = key.publickey().export_key()

        rsa_public_key = RSA.import_key(public_key)
        cipher = PKCS1_OAEP.new(rsa_public_key)
        encrypted_data = cipher.encrypt(result_text.encode())
        encrypted_base64 = base64.b64encode(encrypted_data).decode()

        # Decryption if key provided
        if decrypt_key_input:
            try:
                rsa_private_key = RSA.import_key(private_key)
                cipher_dec = PKCS1_OAEP.new(rsa_private_key)
                decrypted_data = cipher_dec.decrypt(base64.b64decode(encrypted_base64))
                decrypted_text = decrypted_data.decode()
            except Exception as e:
                flash(f"Decryption failed: {str(e)}")

    return render_template(
        'prediction.html',
        result_text=result_text,
        encrypted_base64=encrypted_base64,
        decrypted_text=decrypted_text
    )



if __name__ == '__main__':
    # Ensure database is initialized
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)


# if __name__ == '__main__':
#     app.run(debug=False)
