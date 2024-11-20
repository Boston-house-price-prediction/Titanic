import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 讀取資料
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 印出前10筆資料
print("Train Data (前10筆):")
print(train_df.head(10))
print("\nTest Data (前10筆):")
print(test_df.head(10))

# 資料前處理
def preprocess_data(df):
    # 填補缺失值
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # 類別標籤轉換
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    
    # 刪除不必要的欄位
    df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
    
    return df

train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

# 分割特徵和標籤
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

# 標準化
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(test_df)

# 分割訓練集和測試集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練羅吉斯回歸模型
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_val)

# 訓練決策樹模型
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_val)

# 評估模型
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# 評估羅吉斯回歸模型
log_reg_metrics = evaluate_model(y_val, y_pred_log_reg)
print("\nLogistic Regression Model Evaluation:")
print(f"Accuracy: {log_reg_metrics[0]:.4f}")
print(f"Precision: {log_reg_metrics[1]:.4f}")
print(f"Recall: {log_reg_metrics[2]:.4f}")
print(f"F1-score: {log_reg_metrics[3]:.4f}")

# 評估決策樹模型
tree_metrics = evaluate_model(y_val, y_pred_tree)
print("\nDecision Tree Model Evaluation:")
print(f"Accuracy: {tree_metrics[0]:.4f}")
print(f"Precision: {tree_metrics[1]:.4f}")
print(f"Recall: {tree_metrics[2]:.4f}")
print(f"F1-score: {tree_metrics[3]:.4f}")

# 繪製混淆矩陣圖
def plot_confusion_matrix(y_true, y_pred, title, cmap):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

# 繪製羅吉斯回歸模型的混淆矩陣（藍色）
plot_confusion_matrix(y_val, y_pred_log_reg, 'Logistic Regression Confusion Matrix', 'Blues')

# 繪製決策樹模型的混淆矩陣（綠色）
plot_confusion_matrix(y_val, y_pred_tree, 'Decision Tree Confusion Matrix', 'Greens')