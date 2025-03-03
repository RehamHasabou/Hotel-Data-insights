
import warnings
warnings.filterwarnings("ignore")
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_excel('Hotel data.xlsx')

print(data.shape)

data.info()

data.columns

data.describe()

data.isna().sum()

data.duplicated().sum()

data=data.drop_duplicates()

data.shape

data.duplicated().sum()

class_counts = data['is_canceled'].value_counts()
print(class_counts)
#Filter the rows where 'is_canceled' is 0
zero_rows = data[data['is_canceled'] == 0]

# Drop the first 47186 rows with 0 in 'is_caneled'
rows_to_drop = zero_rows.index[:40000]
data = data.drop(rows_to_drop)


# Check the new shape of the data to confirm
print(f"New shape of the dataset: {data.shape}")

class_counts = data['is_canceled'].value_counts()
print(class_counts)

plt.figure(figsize=(18, 8))
correlation_matrix = data[['lead_time','arrival_date_year','arrival_date_week_number','arrival_date_day_of_month','stays_in_weekend_nights',
'stays_in_week_nights','adults','children','babies','is_repeated_guest','previous_cancellations','previous_bookings_not_canceled','booking_changes','days_in_waiting_list',
'adr','required_car_parking_spaces','total_of_special_requests','is_canceled']].astype(float).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

correlation_with_target = correlation_matrix['is_canceled'].sort_values(ascending=False)
print(correlation_with_target)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Define columns
numerical_cols = ['lead_time', 'arrival_date_year', 'arrival_date_day_of_month', 'stays_in_weekend_nights',
                  'stays_in_week_nights', 'adults', 'children', 'is_repeated_guest', 'previous_cancellations',
                  'required_car_parking_spaces']
categorical_cols = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment', 'distribution_channel',
                    'reserved_room_type', 'deposit_type', 'customer_type']

scaler = None
trained_models = {}
label_encoder = {}
model_accuracies = {}


def load_data():
    data = pd.read_excel('Hotel data.xlsx')
    data = data.drop_duplicates()
    # Filter the rows where 'is_canceled' is 0
    zero_rows = data[data['is_canceled'] == 0]

    # Drop the first 47186 rows with 0 in 'is_caneled'
    rows_to_drop = zero_rows.index[:40000]
    data = data.drop(rows_to_drop)
    cols_to_drop = ['arrival_date_week_number', 'babies', 'previous_bookings_not_canceled', 'assigned_room_type',
                    'booking_changes', 'agent', 'company', 'days_in_waiting_list','adr', 'total_of_special_requests', 'reservation_status',
                    'reservation_status_date', 'year']

    df_cleaned = data.drop(columns=cols_to_drop)

    global label_encoder, scaler
    scaler = StandardScaler()

    # Encode categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_cleaned[col] = le.fit_transform(df_cleaned[col])
        label_encoder[col] = le  # Store the encoder for later use

    df_cleaned[numerical_cols] = scaler.fit_transform(df_cleaned[numerical_cols])

    X = df_cleaned.drop('is_canceled', axis=1)
    y = df_cleaned['is_canceled']

    return X, y


def train_models(X, y):
    global trained_models, model_accuracies
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    trained_models = {}

    # Train SVM
    svm_model = SVC(kernel='rbf', gamma=0.01, C=10)
    svm_model.fit(X_train_resampled, y_train_resampled)
    trained_models['SVM'] = svm_model

    # Train Logistic Regression
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)
    trained_models['Logistic Regression'] = logistic_model

    # Train Random Forest
    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest_model.fit(X_train_resampled, y_train_resampled)
    trained_models['Random Forest'] = random_forest_model

    # Train XGBoost
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train_resampled, y_train_resampled)
    trained_models['XGBoost'] = xgb_model

    # Calculate accuracies
    model_accuracies = {}
    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[model_name] = accuracy

    return trained_models, model_accuracies


def predict_models(input_data, trained_models, model_accuracies):
    input_df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col in categorical_cols:
        if col in label_encoder:
            if input_df[col].values[0] in label_encoder[col].classes_:
                input_df[col] = label_encoder[col].transform(input_df[col])
            else:
                raise ValueError(f"Unseen label '{input_df[col].values[0]}' in column '{col}'")

    # Scale numerical columns
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    model_prediction = []
    for model_name, model in trained_models.items():
        pred = model.predict(input_df)
        model_prediction.append(pred[0])

    yes_models = [model for i, model in enumerate(trained_models.keys()) if model_prediction[i] == 1]
    no_models = [model for i, model in enumerate(trained_models.keys()) if model_prediction[i] == 0]

    total_yes_accuracy = sum(model_accuracies[model] for model in yes_models)
    total_no_accuracy = sum(model_accuracies[model] for model in no_models)

    avg_yes_accuracy = total_yes_accuracy / len(yes_models) if yes_models else 0
    avg_no_accuracy = total_no_accuracy / len(no_models) if no_models else 0

    prediction = "The customer will be canceled" if avg_yes_accuracy > avg_no_accuracy else "The customer will not be canceled"
    return prediction


test_data = {
        'hotel': 'City Hotel',  # Assuming '1' corresponds to 'Resort Hotel'
        'lead_time': 15,
        'arrival_date_year': 2025,
        'arrival_date_month':'April',
        'arrival_date_day_of_month': 15,
        'stays_in_weekend_nights': 2,
        'stays_in_week_nights': 3,
        'adults': 2,
        'children': 1,
        'meal': 'BB',
        'country': 'USA',
        'market_segment': 'Online TA',
        'distribution_channel': 'TA/TO',
        'is_repeated_guest': 0,
        'previous_cancellations': 0,
        'reserved_room_type':'A',
        'deposit_type': 'No Deposit',
        'customer_type': 'Transient',
        'required_car_parking_spaces': 1
    }
X , y = load_data()
trained_models, model_accuracies = train_models(X, y)
prediction = predict_models(test_data, trained_models, model_accuracies)
print(prediction)
