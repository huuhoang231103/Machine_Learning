import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, mean_squared_error, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings('ignore')

# Đọc dữ liệu từ file CSV
df = pd.read_csv('enemy_data.csv')

# Mã hóa các cột phân loại thành số
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Phân tách dữ liệu đầu vào (X) và đầu ra (y)
X = df.drop(columns=['Hành động của quái vật'])
y = df['Hành động của quái vật']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Hàm huấn luyện mô hình Decision Tree với tham số khác nhau
def train_and_evaluate(max_depth, min_samples_split, min_samples_leaf):
    dt_model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    return f1_micro

# Dãy các tham số cần thử nghiệm
max_depth_values = [5, 10, 15, 20]
min_samples_split_values = [2, 5, 10]
min_samples_leaf_values = [1, 2, 4]

# Tạo danh sách để lưu kết quả
results = []

# Thử nghiệm với các tham số max_depth khác nhau
for max_depth in max_depth_values:
    f1_scores = []
    for min_samples_split in min_samples_split_values:
        for min_samples_leaf in min_samples_leaf_values:
            f1_micro = train_and_evaluate(max_depth, min_samples_split, min_samples_leaf)
            results.append((max_depth, min_samples_split, min_samples_leaf, f1_micro))

# Chuyển kết quả thành DataFrame
results_df = pd.DataFrame(results, columns=['max_depth', 'min_samples_split', 'min_samples_leaf', 'F1 Micro'])

# Khởi tạo và huấn luyện mô hình Decision Tree với các tham số cụ thể
dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Đánh giá mô hình Decision Tree
f1_micro_dt = f1_score(y_test, y_pred_dt, average='micro')
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"F1 Micro: {f1_micro_dt:.4f}")
print(f"MSE: {mse_dt:.4f}, RMSE: {rmse_dt:.4f}")

# Khởi tạo và huấn luyện mô hình Random Forest với các tham số cụ thể
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Đánh giá mô hình Random Forest
f1_micro_rf = f1_score(y_test, y_pred_rf, average='micro')
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"F1 Micro: {f1_micro_rf:.4f}")
print(f"MSE: {mse_rf:.4f}, RMSE: {rmse_rf:.4f}")


# Vẽ biểu đồ confusion matrix
from sklearn.metrics import confusion_matrix

# Tạo confusion matrix
cm = confusion_matrix(y_test, y_pred_dt)



# Tạo heatmap cho confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=label_encoders['Hành động của quái vật'].classes_, 
            yticklabels=label_encoders['Hành động của quái vật'].classes_)
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Vẽ confusion matrix cho Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoders['Hành động của quái vật'].classes_, yticklabels=label_encoders['Hành động của quái vật'].classes_)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Biểu đồ so sánh F1 Micro với tham số max_depth
plt.figure(figsize=(10, 6))
sns.lineplot(data=results_df, x='max_depth', y='F1 Micro', marker='o', hue='min_samples_split', style='min_samples_leaf', markersize=8)
plt.title('So sánh F1 Micro với các tham số của Decision Tree')
plt.xlabel('Max Depth')
plt.ylabel('F1 Micro')
plt.legend(title='Min Samples Split & Min Samples Leaf')
plt.show()

# Nếu muốn vẽ thêm một biểu đồ so sánh F1 Micro với tham số min_samples_split
plt.figure(figsize=(12, 8))
sns.lineplot(data=results_df, x='min_samples_split', y='F1 Micro', marker='o', hue='max_depth', style='min_samples_leaf', markersize=8)
plt.title('So sánh F1 Micro với tham số Min Samples Split')
plt.xlabel('Min Samples Split')
plt.ylabel('F1 Micro')
plt.legend(title='Max Depth & Min Samples Leaf', loc='best')
plt.grid(True)
plt.show()



# Tạo pivot table từ kết quả để hiển thị heatmap
heatmap_data = results_df.pivot_table(
    index='max_depth',
    columns='min_samples_split',
    values='F1 Micro'
)

# Vẽ heatmap so sánh F1 Micro với max_depth và min_samples_split
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="coolwarm", cbar_kws={'label': 'F1 Micro'})
plt.title("Heatmap - So sánh F1 Micro theo max_depth và min_samples_split")
plt.xlabel("Min Samples Split")
plt.ylabel("Max Depth")
plt.show()

# Vẽ biểu đồ cây quyết định cho mô hình Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(dt_model, 
          feature_names=X.columns, 
          class_names=label_encoders['Hành động của quái vật'].classes_, 
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("Decision Tree Visualization")
plt.show()

# Vẽ biểu đồ tương quan ma trận cho dữ liệu đầu vào
plt.figure(figsize=(12, 10))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Correlation Matrix Heatmap")
plt.show()

# Hiển thị một số kết quả dự đoán mẫu
sample_data = X_test.iloc[:5]
sample_labels = y_test.iloc[:5]
sample_preds_dt = dt_model.predict(sample_data)

for i in range(len(sample_data)):
    input_data = sample_data.iloc[i]
    actual_label = sample_labels.iloc[i]
    predicted_label_dt = sample_preds_dt[i]

    print(f"\nInput: {input_data.to_dict()}")
    print(f"Actual Action: {label_encoders['Hành động của quái vật'].inverse_transform([actual_label])[0]}")
    print(f"Predicted Action - Decision Tree: {label_encoders['Hành động của quái vật'].inverse_transform([predicted_label_dt])[0]}")
