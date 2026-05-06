from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify
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
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import base64
import sqlite3
from pathlib import Path
from flask_mail import Mail, Message
import random
import string
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
from sqlalchemy import desc, func, text, inspect
import logging
import io
import base64
from matplotlib.figure import Figure

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get the absolute path to the current directory
BASE_DIR = Path(__file__).parent.absolute()
DATABASE_PATH = BASE_DIR / "database.db"

print(f"Current directory: {BASE_DIR}")
print(f"Database will be created at: {DATABASE_PATH}")

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Email Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('GMAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('GMAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('GMAIL_USERNAME')

mail = Mail(app)

# Use absolute path for database
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DATABASE_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
db = SQLAlchemy(app)

# ====================== DATABASE MODELS ======================
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(10), default='user')
    status = db.Column(db.String(10), default='active')  # active, suspended, deleted
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    reset_token = db.Column(db.String(100))
    reset_token_expiry = db.Column(db.DateTime)
    
    # Relationships
    activities = db.relationship('ActivityLog', backref='user', lazy=True)
    predictions = db.relationship('PredictionLog', backref='user', lazy=True)

class PasswordResetOTP(db.Model):
    __tablename__ = 'password_reset_otp'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), nullable=False)
    otp = db.Column(db.String(6), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)
    attempts = db.Column(db.Integer, default=0)
    is_verified = db.Column(db.Boolean, default=False)

class ActivityLog(db.Model):
    __tablename__ = 'activity_log'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    activity_type = db.Column(db.String(50), nullable=False)  # login, logout, upload, predict, etc.
    description = db.Column(db.Text, nullable=False)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Index for faster queries
    __table_args__ = (db.Index('idx_activity_user_time', 'user_id', 'timestamp'),)

class PredictionLog(db.Model):
    __tablename__ = 'prediction_log'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    input_data = db.Column(db.Text, nullable=False)  # JSON string of input
    prediction_result = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float)
    is_leakage = db.Column(db.Boolean, default=False)
    encrypted_result = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Index for analytics
    __table_args__ = (db.Index('idx_prediction_user_time', 'user_id', 'timestamp'),)

class SystemMetrics(db.Model):
    __tablename__ = 'system_metrics'
    id = db.Column(db.Integer, primary_key=True)
    metric_date = db.Column(db.Date, nullable=False, unique=True)
    total_users = db.Column(db.Integer, default=0)
    active_users = db.Column(db.Integer, default=0)
    total_predictions = db.Column(db.Integer, default=0)
    leakage_predictions = db.Column(db.Integer, default=0)
    avg_confidence = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create directories
for directory in ['uploads', 'models', 'static', 'static/charts']:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load Random Forest Model
rf_model = None
try:
    with open("Random_Forest_Model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    print("Random Forest model loaded successfully")
except FileNotFoundError:
    print("Warning: Random_Forest_Model.pkl not found")

# ====================== HELPER FUNCTIONS ======================
def save_plot_to_base64(fig):
    """Save matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_str

def log_activity(user_id, activity_type, description, ip_address=None, user_agent=None):
    """Log user activity"""
    try:
        activity = ActivityLog(
            user_id=user_id,
            activity_type=activity_type,
            description=description,
            ip_address=ip_address,
            user_agent=user_agent
        )
        db.session.add(activity)
        db.session.commit()
        return True
    except Exception as e:
        logger.error(f"Error logging activity: {e}")
        db.session.rollback()
        return False

def generate_otp(length=6):
    """Generate a random OTP"""
    return ''.join(random.choices(string.digits, k=length))

def migrate_database():
    """Migrate database to new schema"""
    try:
        with app.app_context():
            print("\n" + "=" * 60)
            print("DATABASE MIGRATION")
            print("=" * 60)
            
            # Create all tables
            db.create_all()
            print("✓ All tables created/updated")
            
            # Create default admin if not exists
            admin = User.query.filter_by(email='admin@example.com').first()
            if not admin:
                admin = User(
                    name='Admin',
                    email='admin@example.com',
                    password=generate_password_hash('admin123'),
                    role='admin',
                    status='active'
                )
                db.session.add(admin)
                db.session.commit()
                print("✓ Default admin created")
            
            return True
            
    except Exception as e:
        print(f"✗ Error migrating database: {e}")
        import traceback
        traceback.print_exc()
        return False

# ====================== ROUTES ======================

# ====================== HOME ROUTE ======================
@app.route('/')
def home():
    return render_template('home.html')

# ====================== LOGIN / REGISTER ==================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            email = request.form['email'].strip().lower()
            password = request.form['password']
            
            user = User.query.filter_by(email=email).first()
            
            if user and check_password_hash(user.password, password):
                session['user_id'] = user.id
                session['role'] = user.role
                session['name'] = user.name
                
                # Update last login
                user.last_login = datetime.utcnow()
                db.session.commit()
                
                log_activity(user.id, 'login', 'User logged in', 
                           request.remote_addr, request.user_agent.string)
                
                flash(f'Welcome back, {user.name}!', 'success')
                
                if user.role == 'admin':
                    return redirect('/admin_dashboard')
                else:
                    return redirect('/user_dashboard')
            else:
                flash('Invalid email or password', 'error')
                
        except Exception as e:
            print(f"Login error: {e}")
            flash('Login failed. Please try again.', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            name = request.form['name'].strip()
            email = request.form['email'].strip().lower()
            password = request.form['password']
            
            if len(password) < 8:
                flash('Password must be at least 8 characters', 'error')
                return render_template('register.html')
            
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                flash('Email already registered', 'error')
                return render_template('register.html')
            
            new_user = User(
                name=name, 
                email=email, 
                password=generate_password_hash(password), 
                role='user',
                status='active'
            )
            
            db.session.add(new_user)
            db.session.commit()
            
            log_activity(new_user.id, 'registration', 'New user registered',
                        request.remote_addr, request.user_agent.string)
            
            flash('Registration successful! Please login.', 'success')
            return redirect('/login')
            
        except Exception as e:
            db.session.rollback()
            flash(f'Registration failed: {str(e)}', 'error')
    
    return render_template('register.html')

# ====================== ADMIN DASHBOARD ==================
@app.route('/admin_dashboard')
def admin_dashboard():
    if 'role' not in session or session['role'] != 'admin':
        flash('Please login as admin', 'error')
        return redirect('/login')
    
    users = User.query.all()
    return render_template('admin_dashboard.html', users=users)

# ====================== USER DASHBOARD ==================
@app.route('/user_dashboard')
def user_dashboard():
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect('/login')
    return render_template('user_dashboard.html')

# ====================== LOGOUT ===========================
@app.route('/logout')
def logout():
    if 'user_id' in session:
        log_activity(session['user_id'], 'logout', 'User logged out',
                    request.remote_addr, request.user_agent.string)
    
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect('/')

# ====================== UPLOAD DATASET ===================
@app.route('/upload_dataset', methods=['GET', 'POST'])
def upload_dataset():
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect('/login')
    
    if request.method == 'POST':
        if 'dataset' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)
        
        file = request.files['dataset']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            session['dataset_path'] = filepath
            
            # Read and display sample
            try:
                df = pd.read_csv(filepath)
                preview_html = df.head(10).to_html(classes='table table-striped', index=False)
                session['preview_html'] = preview_html
                session['dataset_columns'] = list(df.columns)
                session['dataset_shape'] = df.shape
                
                log_activity(session['user_id'], 'upload', f'Uploaded dataset: {file.filename}',
                           request.remote_addr, request.user_agent.string)
                
                flash(f'Dataset uploaded successfully! Shape: {df.shape}', 'success')
                return render_template('upload.html', preview=preview_html, 
                                      shape=df.shape, columns=df.columns.tolist())
            except Exception as e:
                flash(f'Error reading CSV file: {str(e)}', 'error')
        else:
            flash('Please upload a CSV file', 'error')
    
    return render_template('upload.html')

# ====================== PREPROCESSING ====================
@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    if 'dataset_path' not in session:
        flash('Please upload a dataset first', 'error')
        return redirect('/upload_dataset')
    
    messages = []
    charts = []
    
    try:
        df = pd.read_csv(session['dataset_path'])
        
        if request.method == 'POST':
            # Store original info
            original_shape = df.shape
            messages.append(f"Original dataset shape: {original_shape}")
            
            # 1. Handle Missing Values
            messages.append("\n" + "="*50)
            messages.append("STEP 1: HANDLING MISSING VALUES")
            messages.append("="*50)
            
            missing_before = df.isnull().sum().sum()
            messages.append(f"Total missing values before: {missing_before}")
            
            if missing_before > 0:
                # Create missing values chart
                missing_cols = df.isnull().sum()
                missing_cols = missing_cols[missing_cols > 0]
                
                if len(missing_cols) > 0:
                    fig1 = Figure(figsize=(10, 6))
                    ax1 = fig1.add_subplot(111)
                    missing_cols.plot(kind='bar', ax=ax1, color='coral')
                    ax1.set_title('Missing Values by Column')
                    ax1.set_xlabel('Columns')
                    ax1.set_ylabel('Missing Count')
                    ax1.tick_params(axis='x', rotation=45)
                    fig1.tight_layout()
                    
                    missing_chart = save_plot_to_base64(fig1)
                    charts.append({
                        'title': 'Missing Values Analysis',
                        'image': missing_chart
                    })
                
                # Fill missing values
                df = df.fillna(0)
                missing_after = df.isnull().sum().sum()
                messages.append(f"Missing values filled with 0")
                messages.append(f"Total missing values after: {missing_after}")
            else:
                messages.append("No missing values found in the dataset")
            
            # 2. Label Encoding
            messages.append("\n" + "="*50)
            messages.append("STEP 2: LABEL ENCODING")
            messages.append("="*50)
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            messages.append(f"Categorical columns found: {len(categorical_cols)}")
            
            if len(categorical_cols) > 0:
                label_encoder = LabelEncoder()
                encoding_results = {}
                
                for col in categorical_cols:
                    original_values = df[col].unique()[:5]
                    df[col] = label_encoder.fit_transform(df[col].astype(str))
                    encoded_values = df[col].unique()[:5]
                    
                    encoding_results[col] = {
                        'original': list(original_values),
                        'encoded': list(encoded_values)
                    }
                
                # Display encoding results
                for col, values in encoding_results.items():
                    messages.append(f"\nColumn: {col}")
                    messages.append(f"  Original values (sample): {values['original']}")
                    messages.append(f"  Encoded values (sample): {values['encoded']}")
                
                # Create categorical distribution chart
                if len(categorical_cols) > 0:
                    sample_col = categorical_cols[0]
                    fig2 = Figure(figsize=(12, 6))
                    ax2 = fig2.add_subplot(111)
                    
                    # Plot distribution before encoding
                    value_counts = pd.Series([str(x) for x in df[sample_col].value_counts().index[:10]])
                    counts = df[sample_col].value_counts().values[:10]
                    
                    bars = ax2.bar(range(len(counts)), counts, color='skyblue', alpha=0.7)
                    ax2.set_title(f'Distribution of {sample_col} (After Encoding)')
                    ax2.set_xlabel('Encoded Values')
                    ax2.set_ylabel('Count')
                    ax2.set_xticks(range(len(counts)))
                    ax2.set_xticklabels(value_counts, rotation=45)
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}', ha='center', va='bottom')
                    
                    fig2.tight_layout()
                    encoding_chart = save_plot_to_base64(fig2)
                    charts.append({
                        'title': f'Distribution of {sample_col}',
                        'image': encoding_chart
                    })
            else:
                messages.append("No categorical columns found for encoding")
            
            # 3. Feature Scaling (Optional)
            messages.append("\n" + "="*50)
            messages.append("STEP 3: FEATURE SCALING")
            messages.append("="*50)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                scaler = StandardScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                messages.append(f"Applied StandardScaler to {len(numeric_cols)} numeric columns")
                
                # Create before/after scaling comparison
                if len(numeric_cols) > 0:
                    sample_num_col = numeric_cols[0]
                    fig3 = Figure(figsize=(12, 6))
                    
                    # Before scaling
                    ax3a = fig3.add_subplot(121)
                    original_df = pd.read_csv(session['dataset_path'])
                    if sample_num_col in original_df.columns:
                        ax3a.hist(original_df[sample_num_col].dropna(), bins=30, 
                                 color='lightcoral', alpha=0.7, edgecolor='black')
                        ax3a.set_title(f'{sample_num_col} - Before Scaling')
                        ax3a.set_xlabel('Value')
                        ax3a.set_ylabel('Frequency')
                    
                    # After scaling
                    ax3b = fig3.add_subplot(122)
                    ax3b.hist(df[sample_num_col], bins=30, color='lightgreen', 
                             alpha=0.7, edgecolor='black')
                    ax3b.set_title(f'{sample_num_col} - After Scaling')
                    ax3b.set_xlabel('Scaled Value')
                    ax3b.set_ylabel('Frequency')
                    
                    fig3.tight_layout()
                    scaling_chart = save_plot_to_base64(fig3)
                    charts.append({
                        'title': 'Feature Scaling Comparison',
                        'image': scaling_chart
                    })
            
            # 4. Data Overview after preprocessing
            messages.append("\n" + "="*50)
            messages.append("STEP 4: DATA OVERVIEW")
            messages.append("="*50)
            
            messages.append(f"\nFinal dataset shape: {df.shape}")
            messages.append(f"Number of features: {df.shape[1]}")
            
            # Save preprocessed data
            preprocessed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'preprocessed.csv')
            df.to_csv(preprocessed_path, index=False)
            session['preprocessed_path'] = preprocessed_path
            
            # Create final data distribution chart
            fig4 = Figure(figsize=(10, 6))
            ax4 = fig4.add_subplot(111)
            
            # Check if there's a target column
            target_col = None
            for col in ['Target', 'target', 'Class', 'class', 'Label', 'label']:
                if col in df.columns:
                    target_col = col
                    break
            
            if target_col:
                target_counts = df[target_col].value_counts()
                bars = ax4.bar(target_counts.index.astype(str), target_counts.values, 
                             color=['lightgreen', 'lightcoral'], alpha=0.7)
                ax4.set_title('Target Variable Distribution')
                ax4.set_xlabel('Class')
                ax4.set_ylabel('Count')
                
                for bar in bars:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom')
            else:
                # Plot correlation matrix for first 10 numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    im = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
                    ax4.set_title('Feature Correlation Matrix')
                    ax4.set_xticks(range(len(numeric_cols)))
                    ax4.set_yticks(range(len(numeric_cols)))
                    ax4.set_xticklabels(numeric_cols, rotation=45, ha='right')
                    ax4.set_yticklabels(numeric_cols)
                    fig4.colorbar(im, ax=ax4)
            
            fig4.tight_layout()
            overview_chart = save_plot_to_base64(fig4)
            charts.append({
                'title': 'Final Data Overview',
                'image': overview_chart
            })
            
            log_activity(session['user_id'], 'preprocess', 'Dataset preprocessing completed',
                        request.remote_addr, request.user_agent.string)
            
            flash('Preprocessing completed successfully!', 'success')
            return render_template('preprocessing.html', 
                                 messages=messages, 
                                 charts=charts,
                                 completed=True)
        
        else:
            # GET request - show dataset overview
            messages.append("Dataset Overview:")
            messages.append(f"Shape: {df.shape}")
            messages.append(f"Columns: {', '.join(df.columns.tolist()[:10])}")
            if len(df.columns) > 10:
                messages.append(f"... and {len(df.columns) - 10} more columns")
            
            # Data types
            messages.append("\nData Types:")
            for dtype, count in df.dtypes.value_counts().items():
                messages.append(f"  {dtype}: {count} columns")
            
            # Missing values summary
            missing_total = df.isnull().sum().sum()
            messages.append(f"\nMissing Values: {missing_total}")
            
            if missing_total > 0:
                missing_cols = df.columns[df.isnull().any()].tolist()
                messages.append(f"Columns with missing values: {', '.join(missing_cols)}")
            
            return render_template('preprocessing.html', 
                                 messages=messages, 
                                 charts=charts,
                                 completed=False)
            
    except Exception as e:
        print(f"Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        flash(f'Error during preprocessing: {str(e)}', 'error')
        return redirect('/upload_dataset')

# ====================== DATA SPLITTING ====================
@app.route('/data_split', methods=['GET', 'POST'])
def data_split():
    if 'preprocessed_path' not in session:
        flash('Please preprocess the dataset first', 'error')
        return redirect('/preprocess')
    
    messages = []
    charts = []
    
    try:
        df = pd.read_csv(session['preprocessed_path'])
        
        # Check for target column
        target_col = None
        for col in ['Target', 'target', 'Class', 'class', 'Label', 'label']:
            if col in df.columns:
                target_col = col
                break
        
        if not target_col:
            flash("No target column found. Please ensure your dataset has a target variable named 'Target', 'target', 'Class', or 'Label'.", 'error')
            return redirect('/preprocess')
        
        if request.method == 'POST':
            # Get split ratio from form
            test_size = float(request.form.get('test_size', 0.2))
            random_state = int(request.form.get('random_state', 42))
            
            messages.append("\n" + "="*50)
            messages.append("DATA SPLITTING")
            messages.append("="*50)
            
            # Split features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            messages.append(f"Features shape: {X.shape}")
            messages.append(f"Target shape: {y.shape}")
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            messages.append(f"\nSplit Parameters:")
            messages.append(f"Test size: {test_size} ({int(test_size * 100)}%)")
            messages.append(f"Train size: {1-test_size} ({int((1-test_size) * 100)}%)")
            messages.append(f"Random state: {random_state}")
            
            messages.append(f"\nSplit Results:")
            messages.append(f"Training set: {X_train.shape[0]} samples")
            messages.append(f"Testing set: {X_test.shape[0]} samples")
            
            # Store in session
            session['X_train'] = X_train.to_json()
            session['X_test'] = X_test.to_json()
            session['y_train'] = y_train.to_json()
            session['y_test'] = y_test.to_json()
            session['test_size'] = test_size
            session['random_state'] = random_state
            
            # Create split visualization
            fig = Figure(figsize=(12, 8))
            
            # Plot 1: Class distribution in train/test
            ax1 = fig.add_subplot(221)
            train_counts = y_train.value_counts().sort_index()
            test_counts = y_test.value_counts().sort_index()
            
            x = np.arange(len(train_counts))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, train_counts.values, width, 
                          label='Train', color='skyblue', alpha=0.8)
            bars2 = ax1.bar(x + width/2, test_counts.values, width, 
                          label='Test', color='lightcoral', alpha=0.8)
            
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Count')
            ax1.set_title('Class Distribution in Train/Test Sets')
            ax1.set_xticks(x)
            ax1.set_xticklabels([f'Class {i}' for i in train_counts.index])
            ax1.legend()
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom')
            
            # Plot 2: Train-test split ratio
            ax2 = fig.add_subplot(222)
            sizes = [len(X_train), len(X_test)]
            labels = [f'Train\n{len(X_train)} samples\n({(1-test_size)*100:.1f}%)', 
                     f'Test\n{len(X_test)} samples\n({test_size*100:.1f}%)']
            colors = ['lightgreen', 'lightcoral']
            
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors,
                                              autopct='%1.1f%%', startangle=90)
            
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
            
            ax2.set_title('Train-Test Split Ratio')
            
            # Plot 3: Feature importance (simplified)
            ax3 = fig.add_subplot(223)
            
            # Calculate correlation with target for top 10 features
            correlations = X.corrwith(y).abs().sort_values(ascending=False)[:10]
            
            if len(correlations) > 0:
                y_pos = np.arange(len(correlations))
                bars = ax3.barh(y_pos, correlations.values, color='teal', alpha=0.7)
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels(correlations.index)
                ax3.set_xlabel('Absolute Correlation')
                ax3.set_title('Top 10 Feature Correlations with Target')
                
                # Add correlation values
                for i, (bar, corr) in enumerate(zip(bars, correlations.values)):
                    width = bar.get_width()
                    ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{corr:.3f}', ha='left', va='center')
            
            # Plot 4: Data statistics
            ax4 = fig.add_subplot(224)
            ax4.axis('off')
            
            stats_text = f"""
            Dataset Statistics:
            ------------------
            Total Samples: {len(df)}
            Features: {X.shape[1]}
            Classes: {len(y.unique())}
            
            Training Set:
            - Samples: {len(X_train)}
            - Features: {X_train.shape[1]}
            
            Testing Set:
            - Samples: {len(X_test)}
            - Features: {X_test.shape[1]}
            
            Class Balance (Train):
            """
            
            for cls, count in train_counts.items():
                stats_text += f"\n- Class {cls}: {count} ({count/len(X_train)*100:.1f}%)"
            
            ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            fig.tight_layout()
            split_chart = save_plot_to_base64(fig)
            charts.append({
                'title': 'Data Splitting Analysis',
                'image': split_chart
            })
            
            # Save split data to files
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
            
            log_activity(session['user_id'], 'data_split', 
                        f'Data split completed: {len(X_train)} train, {len(X_test)} test samples',
                        request.remote_addr, request.user_agent.string)
            
            flash('Data splitting completed successfully!', 'success')
            return render_template('data_split.html', 
                                 messages=messages, 
                                 charts=charts,
                                 completed=True,
                                 test_size=test_size,
                                 train_size=1-test_size)
        
        else:
            # GET request - show data overview
            messages.append("Dataset Overview for Splitting:")
            messages.append(f"Total samples: {len(df)}")
            messages.append(f"Number of features: {len(df.columns) - 1}")
            messages.append(f"Target column: {target_col}")
            
            # Show class distribution
            class_dist = df[target_col].value_counts()
            messages.append(f"\nClass Distribution:")
            for cls, count in class_dist.items():
                percentage = (count / len(df)) * 100
                messages.append(f"  Class {cls}: {count} samples ({percentage:.1f}%)")
            
            return render_template('data_split.html', 
                                 messages=messages, 
                                 charts=charts,
                                 completed=False)
            
    except Exception as e:
        print(f"Data splitting error: {e}")
        import traceback
        traceback.print_exc()
        flash(f'Error during data splitting: {str(e)}', 'error')
        return redirect('/preprocess')

# ====================== PCA ==============================
@app.route('/pca', methods=['GET', 'POST'])
def pca():
    if 'X_train_path' not in session:
        flash('Please split the data first', 'error')
        return redirect('/data_split')
    
    messages = []
    charts = []
    
    try:
        X_train = pd.read_csv(session['X_train_path'])
        
        if request.method == 'POST':
            n_components = int(request.form.get('components', 2))
            
            messages.append("\n" + "="*50)
            messages.append("PRINCIPAL COMPONENT ANALYSIS (PCA)")
            messages.append("="*50)
            
            # Apply PCA
            pca = PCA(n_components=n_components, random_state=42)
            X_pca = pca.fit_transform(X_train)
            
            messages.append(f"Original shape: {X_train.shape}")
            messages.append(f"PCA shape: {X_pca.shape}")
            messages.append(f"Number of components: {n_components}")
            
            # Explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            messages.append(f"\nExplained Variance:")
            for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
                messages.append(f"  PC{i+1}: {var*100:.2f}% (Cumulative: {cum_var*100:.2f}%)")
            
            # Store PCA results in session
            session['pca_model'] = pickle.dumps(pca)
            session['X_pca'] = pd.DataFrame(X_pca).to_json()
            session['n_components'] = n_components
            session['explained_variance'] = explained_variance.tolist()
            
            # Create PCA visualizations
            fig = Figure(figsize=(15, 10))
            
            # Plot 1: Scree plot
            ax1 = fig.add_subplot(231)
            components = range(1, n_components + 1)
            ax1.bar(components, explained_variance * 100, color='skyblue', alpha=0.8)
            ax1.plot(components, cumulative_variance * 100, 'ro-', linewidth=2)
            ax1.set_xlabel('Principal Components')
            ax1.set_ylabel('Explained Variance (%)')
            ax1.set_title('Scree Plot')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
                ax1.text(i + 1, var * 100 + 1, f'{var*100:.1f}%', 
                        ha='center', va='bottom', fontsize=9)
                if i == n_components - 1:
                    ax1.text(i + 1, cum_var * 100 + 1, f'Total: {cum_var*100:.1f}%', 
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Plot 2: 2D PCA scatter plot
            ax2 = fig.add_subplot(232)
            if n_components >= 2:
                scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, 
                                     c=np.arange(len(X_pca)), cmap='viridis')
                ax2.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)')
                ax2.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)')
                ax2.set_title('2D PCA Projection')
                ax2.grid(True, alpha=0.3)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax2)
                cbar.set_label('Sample Index')
            
            # Plot 3: 3D PCA scatter plot (if 3+ components)
            if n_components >= 3:
                ax3 = fig.add_subplot(233, projection='3d')
                scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                                     alpha=0.6, c=np.arange(len(X_pca)), cmap='viridis')
                ax3.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)')
                ax3.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)')
                ax3.set_zlabel(f'PC3 ({explained_variance[2]*100:.1f}%)')
                ax3.set_title('3D PCA Projection')
            
            # Plot 4: Cumulative explained variance
            ax4 = fig.add_subplot(234)
            ax4.plot(range(1, n_components + 1), cumulative_variance * 100, 
                    'bo-', linewidth=2, markersize=8)
            ax4.axhline(y=95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
            ax4.set_xlabel('Number of Components')
            ax4.set_ylabel('Cumulative Explained Variance (%)')
            ax4.set_title('Cumulative Explained Variance')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            # Find components needed for 95% variance
            components_for_95 = np.where(cumulative_variance >= 0.95)[0]
            if len(components_for_95) > 0:
                n_95 = components_for_95[0] + 1
                ax4.axvline(x=n_95, color='g', linestyle='--', alpha=0.7, 
                          label=f'{n_95} components for 95%')
                messages.append(f"\nComponents needed for 95% variance: {n_95}")
            
            # Plot 5: PCA loadings (first component)
            ax5 = fig.add_subplot(235)
            loadings = pca.components_[0]
            top_features = 10
            
            # Get top positive and negative loadings
            feature_names = X_train.columns
            idx_pos = np.argsort(loadings)[-top_features:]
            idx_neg = np.argsort(loadings)[:top_features]
            
            all_idx = np.concatenate([idx_pos, idx_neg])
            all_loadings = loadings[all_idx]
            all_features = feature_names[all_idx]
            
            colors = ['green' if x > 0 else 'red' for x in all_loadings]
            bars = ax5.barh(range(len(all_loadings)), all_loadings, color=colors, alpha=0.7)
            ax5.set_yticks(range(len(all_loadings)))
            ax5.set_yticklabels(all_features)
            ax5.set_xlabel('Loading Value')
            ax5.set_title(f'Top {top_features*2} Features in PC1')
            ax5.grid(True, alpha=0.3, axis='x')
            
            # Plot 6: Biplot (if 2D)
            if n_components >= 2:
                ax6 = fig.add_subplot(236)
                
                # Scatter points
                scatter = ax6.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, c='gray')
                
                # Plot feature vectors
                scale = 5
                for i, feature in enumerate(feature_names[:10]):  # Show top 10 features
                    ax6.arrow(0, 0, pca.components_[0, i] * scale, 
                             pca.components_[1, i] * scale, 
                             head_width=0.1, head_length=0.1, fc='red', ec='red')
                    ax6.text(pca.components_[0, i] * scale * 1.1, 
                            pca.components_[1, i] * scale * 1.1, 
                            feature, color='red', fontsize=9)
                
                ax6.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)')
                ax6.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)')
                ax6.set_title('Biplot (Feature Vectors)')
                ax6.grid(True, alpha=0.3)
                ax6.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax6.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            fig.tight_layout()
            pca_chart = save_plot_to_base64(fig)
            charts.append({
                'title': 'PCA Analysis Results',
                'image': pca_chart
            })
            
            log_activity(session['user_id'], 'pca', 
                        f'PCA applied: {n_components} components, {cumulative_variance[-1]*100:.1f}% variance explained',
                        request.remote_addr, request.user_agent.string)
            
            flash('PCA analysis completed successfully!', 'success')
            return render_template('pca.html', 
                                 messages=messages, 
                                 charts=charts,
                                 completed=True,
                                 n_components=n_components,
                                 explained_variance=explained_variance,
                                 cumulative_variance=cumulative_variance)
        
        else:
            # GET request - show PCA options
            messages.append("PCA Configuration:")
            messages.append(f"Number of features: {X_train.shape[1]}")
            messages.append(f"Recommended maximum components: {min(10, X_train.shape[1])}")
            messages.append("\nNote: PCA will reduce dimensionality while preserving variance")
            
            return render_template('pca.html', 
                                 messages=messages, 
                                 charts=charts,
                                 completed=False)
            
    except Exception as e:
        print(f"PCA error: {e}")
        import traceback
        traceback.print_exc()
        flash(f'Error during PCA: {str(e)}', 'error')
        return redirect('/data_split')

# ====================== TRAIN MODEL ======================
@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    if 'X_train_path' not in session or 'y_train_path' not in session:
        flash('Please split the data first', 'error')
        return redirect('/data_split')
    
    messages = []
    charts = []
    results = {}
    
    try:
        # Load data
        X_train = pd.read_csv(session['X_train_path'])
        X_test = pd.read_csv(session['X_test_path'])
        y_train = pd.read_csv(session['y_train_path']).squeeze()
        y_test = pd.read_csv(session['y_test_path']).squeeze()
        
        if request.method == 'POST':
            model_type = request.form.get('model_type', 'rf')
            
            messages.append("\n" + "="*50)
            messages.append("MODEL TRAINING")
            messages.append("="*50)
            messages.append(f"Model type: {model_type.upper()}")
            
            if model_type == 'rf':
                # Random Forest
                messages.append("\nTraining Random Forest Classifier...")
                
                n_estimators = int(request.form.get('rf_n_estimators', 100))
                max_depth = request.form.get('rf_max_depth')
                max_depth = int(max_depth) if max_depth else None
                
                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Train model
                start_time = time.time()
                rf.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Make predictions
                y_train_pred = rf.predict(X_train)
                y_test_pred = rf.predict(X_test)
                
                # Calculate metrics
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                train_report = classification_report(y_train, y_train_pred, output_dict=True)
                test_report = classification_report(y_test, y_test_pred, output_dict=True)
                
                messages.append(f"\nTraining Parameters:")
                messages.append(f"  Number of trees: {n_estimators}")
                messages.append(f"  Max depth: {max_depth if max_depth else 'Unlimited'}")
                messages.append(f"  Training time: {training_time:.2f} seconds")
                
                messages.append(f"\nPerformance Metrics:")
                messages.append(f"  Training Accuracy: {train_accuracy*100:.2f}%")
                messages.append(f"  Testing Accuracy: {test_accuracy*100:.2f}%")
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=False)
                
                messages.append(f"\nTop 5 Important Features:")
                for i, row in feature_importance.head().iterrows():
                    messages.append(f"  {row['feature']}: {row['importance']:.4f}")
                
                # Store results
                results = {
                    'model_type': 'Random Forest',
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'training_time': training_time,
                    'feature_importance': feature_importance.to_dict('records'),
                    'train_report': train_report,
                    'test_report': test_report
                }
                
                # Create visualizations
                fig = Figure(figsize=(15, 10))
                
                # Plot 1: Confusion Matrix
                ax1 = fig.add_subplot(231)
                cm = confusion_matrix(y_test, y_test_pred)
                im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax1.set_title('Confusion Matrix (Test Set)')
                ax1.set_xlabel('Predicted')
                ax1.set_ylabel('Actual')
                
                # Add text annotations
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax1.text(j, i, format(cm[i, j], 'd'),
                                ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black")
                
                # Plot 2: Feature Importance
                ax2 = fig.add_subplot(232)
                top_features = feature_importance.head(10)
                y_pos = np.arange(len(top_features))
                ax2.barh(y_pos, top_features['importance'], color='skyblue')
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(top_features['feature'])
                ax2.invert_yaxis()
                ax2.set_xlabel('Importance')
                ax2.set_title('Top 10 Feature Importance')
                
                # Plot 3: ROC Curve
                ax3 = fig.add_subplot(233)
                y_pred_proba = rf.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                ax3.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax3.set_xlim([0.0, 1.0])
                ax3.set_ylim([0.0, 1.05])
                ax3.set_xlabel('False Positive Rate')
                ax3.set_ylabel('True Positive Rate')
                ax3.set_title('Receiver Operating Characteristic')
                ax3.legend(loc="lower right")
                ax3.grid(True, alpha=0.3)
                
                # Plot 4: Accuracy Comparison
                ax4 = fig.add_subplot(234)
                categories = ['Training', 'Testing']
                accuracy_values = [train_accuracy * 100, test_accuracy * 100]
                bars = ax4.bar(categories, accuracy_values, 
                             color=['lightgreen', 'lightcoral'])
                ax4.set_ylabel('Accuracy (%)')
                ax4.set_title('Training vs Testing Accuracy')
                ax4.set_ylim([0, 100])
                
                # Add value labels
                for bar, val in zip(bars, accuracy_values):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{val:.2f}%', ha='center', va='bottom')
                
                # Plot 5: Learning Curve (simplified)
                ax5 = fig.add_subplot(235)
                train_sizes = np.linspace(0.1, 1.0, 10)
                train_scores = []
                test_scores = []
                
                for size in train_sizes:
                    n_samples = int(len(X_train) * size)
                    rf_temp = RandomForestClassifier(n_estimators=50, random_state=42)
                    rf_temp.fit(X_train[:n_samples], y_train[:n_samples])
                    train_scores.append(rf_temp.score(X_train[:n_samples], y_train[:n_samples]))
                    test_scores.append(rf_temp.score(X_test, y_test))
                
                ax5.plot(train_sizes * 100, train_scores, 'o-', color='blue', label='Training')
                ax5.plot(train_sizes * 100, test_scores, 'o-', color='green', label='Testing')
                ax5.set_xlabel('Training Set Size (%)')
                ax5.set_ylabel('Accuracy')
                ax5.set_title('Learning Curve')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
                
                # Plot 6: Classification Report Heatmap
                ax6 = fig.add_subplot(236)
                report_df = pd.DataFrame(test_report).transpose().iloc[:-3, :-1]
                im = ax6.imshow(report_df.values, cmap='YlOrRd', aspect='auto')
                ax6.set_xticks(range(len(report_df.columns)))
                ax6.set_yticks(range(len(report_df.index)))
                ax6.set_xticklabels(report_df.columns)
                ax6.set_yticklabels(report_df.index)
                ax6.set_title('Classification Report (Test Set)')
                
                # Add text annotations
                for i in range(len(report_df.index)):
                    for j in range(len(report_df.columns)):
                        text = ax6.text(j, i, f'{report_df.iloc[i, j]:.2f}',
                                       ha="center", va="center", color="black")
                
                fig.tight_layout()
                model_chart = save_plot_to_base64(fig)
                charts.append({
                    'title': 'Random Forest Model Analysis',
                    'image': model_chart
                })
                
                # Save model
                model_path = os.path.join('models', 'random_forest_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(rf, f)
                session['model_path'] = model_path
                
                flash('Random Forest model trained successfully!', 'success')
                
            elif model_type == 'cnn':
                # CNN Model
                messages.append("\nTraining 1D CNN Model...")
                
                # Reshape data for CNN
                X_train_cnn = np.expand_dims(X_train.values, axis=2)
                X_test_cnn = np.expand_dims(X_test.values, axis=2)
                y_train_cnn = y_train.values
                y_test_cnn = y_test.values
                
                # Build CNN model
                model = Sequential([
                    Conv1D(filters=32, kernel_size=3, activation='relu', 
                          input_shape=(X_train_cnn.shape[1], 1)),
                    MaxPooling1D(pool_size=2),
                    Conv1D(filters=64, kernel_size=3, activation='relu'),
                    MaxPooling1D(pool_size=2),
                    Flatten(),
                    Dense(64, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
                
                model.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy'])
                
                # Train model
                epochs = int(request.form.get('cnn_epochs', 10))
                batch_size = int(request.form.get('cnn_batch_size', 32))
                
                start_time = time.time()
                history = model.fit(
                    X_train_cnn, y_train_cnn,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    verbose=0
                )
                training_time = time.time() - start_time
                
                # Evaluate model
                train_loss, train_accuracy = model.evaluate(X_train_cnn, y_train_cnn, verbose=0)
                test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test_cnn, verbose=0)
                
                messages.append(f"\nTraining Parameters:")
                messages.append(f"  Epochs: {epochs}")
                messages.append(f"  Batch size: {batch_size}")
                messages.append(f"  Training time: {training_time:.2f} seconds")
                
                messages.append(f"\nPerformance Metrics:")
                messages.append(f"  Training Accuracy: {train_accuracy*100:.2f}%")
                messages.append(f"  Testing Accuracy: {test_accuracy*100:.2f}%")
                messages.append(f"  Training Loss: {train_loss:.4f}")
                messages.append(f"  Testing Loss: {test_loss:.4f}")
                
                # Store results
                results = {
                    'model_type': '1D CNN',
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'training_time': training_time,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'history': history.history
                }
                
                # Create visualizations
                fig = Figure(figsize=(15, 10))
                
                # Plot 1: Training History
                ax1 = fig.add_subplot(231)
                ax1.plot(history.history['accuracy'], label='Training Accuracy')
                ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Accuracy')
                ax1.set_title('Model Accuracy')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2 = fig.add_subplot(232)
                ax2.plot(history.history['loss'], label='Training Loss')
                ax2.plot(history.history['val_loss'], label='Validation Loss')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.set_title('Model Loss')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Plot 2: Confusion Matrix
                ax3 = fig.add_subplot(233)
                y_pred = (model.predict(X_test_cnn) > 0.5).astype(int)
                cm = confusion_matrix(y_test_cnn, y_pred)
                im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax3.set_title('Confusion Matrix (Test Set)')
                ax3.set_xlabel('Predicted')
                ax3.set_ylabel('Actual')
                
                # Add text annotations
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax3.text(j, i, format(cm[i, j], 'd'),
                                ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black")
                
                # Plot 3: ROC Curve
                ax4 = fig.add_subplot(234)
                y_pred_proba = model.predict(X_test_cnn)
                fpr, tpr, _ = roc_curve(y_test_cnn, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                ax4.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax4.set_xlim([0.0, 1.0])
                ax4.set_ylim([0.0, 1.05])
                ax4.set_xlabel('False Positive Rate')
                ax4.set_ylabel('True Positive Rate')
                ax4.set_title('Receiver Operating Characteristic')
                ax4.legend(loc="lower right")
                ax4.grid(True, alpha=0.3)
                
                # Plot 4: Accuracy Comparison
                ax5 = fig.add_subplot(235)
                categories = ['Training', 'Testing']
                accuracy_values = [train_accuracy * 100, test_accuracy * 100]
                bars = ax5.bar(categories, accuracy_values, 
                             color=['lightgreen', 'lightcoral'])
                ax5.set_ylabel('Accuracy (%)')
                ax5.set_title('Training vs Testing Accuracy')
                ax5.set_ylim([0, 100])
                
                # Add value labels
                for bar, val in zip(bars, accuracy_values):
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{val:.2f}%', ha='center', va='bottom')
                
                # Plot 5: Model Architecture (simplified)
                ax6 = fig.add_subplot(236)
                ax6.axis('off')
                model_summary = []
                model.summary(print_fn=lambda x: model_summary.append(x))
                summary_text = "\n".join(model_summary[:20])
                ax6.text(0.1, 0.9, "Model Architecture:", transform=ax6.transAxes,
                        fontsize=12, fontweight='bold')
                ax6.text(0.1, 0.8, summary_text, transform=ax6.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                fig.tight_layout()
                model_chart = save_plot_to_base64(fig)
                charts.append({
                    'title': 'CNN Model Analysis',
                    'image': model_chart
                })
                
                # Save model
                model_path = os.path.join('models', 'cnn_model.h5')
                model.save(model_path)
                session['model_path'] = model_path
                
                flash('CNN model trained successfully!', 'success')
            
            log_activity(session['user_id'], 'train_model', 
                        f'{model_type.upper()} model trained: {results["test_accuracy"]*100:.2f}% accuracy',
                        request.remote_addr, request.user_agent.string)
            
            return render_template('train_model.html', 
                                 messages=messages, 
                                 charts=charts,
                                 results=results,
                                 completed=True,
                                 model_type=model_type)
        
        else:
            # GET request - show training options
            messages.append("Model Training Configuration:")
            messages.append(f"Training samples: {len(X_train)}")
            messages.append(f"Testing samples: {len(X_test)}")
            messages.append(f"Number of features: {X_train.shape[1]}")
            
            # Check class balance
            class_counts = y_train.value_counts()
            messages.append(f"\nClass distribution (Training):")
            for cls, count in class_counts.items():
                percentage = (count / len(y_train)) * 100
                messages.append(f"  Class {cls}: {count} samples ({percentage:.1f}%)")
            
            return render_template('train_model.html', 
                                 messages=messages, 
                                 charts=charts,
                                 completed=False)
            
    except Exception as e:
        print(f"Model training error: {e}")
        import traceback
        traceback.print_exc()
        flash(f'Error during model training: {str(e)}', 'error')
        return redirect('/data_split')

# ====================== PREDICTION ======================
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    result_text = None
    encrypted_base64 = None
    decrypted_text = None
    confidence = None
    prediction_details = None
    
    if request.method == 'POST':
        try:
            # Get form values
            user_val = request.form.get('user')
            pc_val = request.form.get('pc')
            auth_val = request.form.get('authority')
            through_pwd = request.form.get('through_pwd', type=int, default=0)
            through_pin = request.form.get('through_pin', type=int, default=0)
            through_mfa = request.form.get('through_mfa', type=int, default=0)
            data_mod = request.form.get('data_mod', type=int, default=0)
            conf_access = request.form.get('conf_access', type=int, default=0)
            conf_transfer = request.form.get('conf_transfer', type=int, default=0)
            ext_dest = request.form.get('external_dest')
            container_id = request.form.get('container_id', type=int, default=0)
            req_cpu = request.form.get('req_cpu', type=int, default=0)
            req_mem = request.form.get('req_mem', type=int, default=0)
            req_storage = request.form.get('req_storage', type=int, default=0)
            exec_start = request.form.get('exec_start', type=int, default=0)
            exec_finish = request.form.get('exec_finish', type=int, default=0)
            makespan = request.form.get('makespan', type=int, default=0)
            total_data = request.form.get('total_data', type=int, default=0)
            
            # Prepare input data
            input_data = {
                'user': user_val,
                'pc': pc_val,
                'authority': auth_val,
                'through_pwd': through_pwd,
                'through_pin': through_pin,
                'through_mfa': through_mfa,
                'data_mod': data_mod,
                'conf_access': conf_access,
                'conf_transfer': conf_transfer,
                'external_dest': ext_dest,
                'container_id': container_id,
                'req_cpu': req_cpu,
                'req_mem': req_mem,
                'req_storage': req_storage,
                'exec_start': exec_start,
                'exec_finish': exec_finish,
                'makespan': makespan,
                'total_data': total_data
            }
            
            # Encode categorical variables (simplified)
            def encode_categorical(value, categories):
                try:
                    return categories.index(str(value).lower()) if value in categories else 0
                except:
                    return hash(str(value)) % 100
            
            # Create feature vector
            features = [
                encode_categorical(user_val, ['user1', 'user2', 'user3', 'admin']),
                encode_categorical(pc_val, ['pc1', 'pc2', 'pc3', 'server']),
                encode_categorical(auth_val, ['low', 'medium', 'high', 'admin']),
                through_pwd,
                through_pin,
                through_mfa,
                data_mod,
                conf_access,
                conf_transfer,
                encode_categorical(ext_dest, ['internal', 'external', 'cloud']),
                container_id,
                req_cpu,
                req_mem,
                req_storage,
                exec_start,
                exec_finish,
                makespan,
                total_data
            ]
            
            # Make prediction using loaded model
            if rf_model:
                features_array = np.array(features).reshape(1, -1)
                pred = rf_model.predict(features_array)[0]
                pred_proba = rf_model.predict_proba(features_array)[0]
                
                confidence = max(pred_proba) * 100
                is_leakage = pred == 1
                result_text = "🚨 DATA LEAKAGE DETECTED!" if is_leakage else "✅ No Data Leakage"
                
                # Prepare prediction details
                prediction_details = {
                    'input_data': input_data,
                    'prediction': 'Leakage' if is_leakage else 'Normal',
                    'confidence': confidence,
                    'probability_leakage': pred_proba[1] * 100 if len(pred_proba) > 1 else confidence,
                    'probability_normal': pred_proba[0] * 100,
                    'features_used': len(features),
                    'model_type': 'Random Forest'
                }
                
                # RSA Encryption
                key = RSA.generate(2048)
                private_key = key.export_key()
                public_key = key.publickey().export_key()
                
                rsa_public_key = RSA.import_key(public_key)
                cipher = PKCS1_OAEP.new(rsa_public_key)
                encrypted_data = cipher.encrypt(result_text.encode())
                encrypted_base64 = base64.b64encode(encrypted_data).decode()
                
                # Log prediction if user is logged in
                if 'user_id' in session:
                    log_activity(session['user_id'], 'prediction',
                               f'Made prediction: {result_text} ({confidence:.1f}% confidence)',
                               request.remote_addr, request.user_agent.string)
                
                flash('Prediction completed successfully!', 'success')
            else:
                flash('Model not loaded. Please train a model first.', 'error')
            
        except Exception as e:
            print(f"Prediction error: {e}")
            flash(f'Error during prediction: {str(e)}', 'error')
    
    return render_template('prediction.html',
                         result_text=result_text,
                         encrypted_base64=encrypted_base64,
                         decrypted_text=decrypted_text,
                         confidence=confidence,
                         prediction_details=prediction_details)

# ====================== RUN APPLICATION ======================
if __name__ == '__main__':
    # Migrate database
    with app.app_context():
        migrate_database()
    
    print("\n" + "="*60)
    print("SECURELEAK APPLICATION STARTED")
    print("="*60)
    print(f"Database: {DATABASE_PATH}")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Models folder: models/")
    print("\nAccess URLs:")
    print(f"  Home: http://localhost:5000")
    print(f"  Admin Login: admin@example.com / admin123")
    print(f"  Debug DB: http://localhost:5000/debug/db")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)