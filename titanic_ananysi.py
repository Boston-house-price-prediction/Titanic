import pandas as pd

# 載入 Titanic 資料
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 印出前10筆資料
print(train_data.head(10))

# 檢查缺失值
print(train_data.isnull().sum())

# 填補 Age 缺失值
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)

# 填補 Embarked 缺失值
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# 填補 Fare 缺失值（針對測試集）
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

# 將 Sex 類別標籤轉換
train_data['Sex'] = train_data['Sex'].map({'male': 1, 'female': 0})
test_data['Sex'] = test_data['Sex'].map({'male': 1, 'female': 0})

# 對 Embarked 做 One-Hot Encoding
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Embarked'], drop_first=True)

from sklearn.preprocessing import StandardScaler

# 初始化標準化工具
scaler = StandardScaler()

# 對 Age 和 Fare 進行標準化
train_data[['Age', 'Fare']] = scaler.fit_transform(train_data[['Age', 'Fare']])
test_data[['Age', 'Fare']] = scaler.transform(test_data[['Age', 'Fare']])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 分割資料
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S']
X = train_data[features]
y = train_data['Survived']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 羅吉斯回歸
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_valid)

# 決策樹
tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_valid)

# 評估函數
def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, prec, rec, f1

# 羅吉斯回歸評估
log_acc, log_prec, log_rec, log_f1 = evaluate_model(y_valid, y_pred_log)

# 決策樹評估
tree_acc, tree_prec, tree_rec, tree_f1 = evaluate_model(y_valid, y_pred_tree)

# 印出結果
print(f"Logistic Regression: Accuracy={log_acc:.2f}, Precision={log_prec:.2f}, Recall={log_rec:.2f}, F1-Score={log_f1:.2f}")
print(f"Decision Tree: Accuracy={tree_acc:.2f}, Precision={tree_prec:.2f}, Recall={tree_rec:.2f}, F1-Score={tree_f1:.2f}")

# 繪製混淆矩陣
conf_matrix_log = confusion_matrix(y_valid, y_pred_log)
conf_matrix_tree = confusion_matrix(y_valid, y_pred_tree)

# 羅吉斯回歸混淆矩陣
sns.heatmap(conf_matrix_log, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# 決策樹混淆矩陣
sns.heatmap(conf_matrix_tree, annot=True, fmt='d', cmap='Greens')
plt.title('Decision Tree Confusion Matrix')
plt.show()

# 羅吉斯回歸測試集預測
test_predictions_log = log_reg.predict(test_data[features])

# 決策樹測試集預測
test_predictions_tree = tree_clf.predict(test_data[features])

# 儲存預測結果
output_log = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': test_predictions_log})
output_tree = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': test_predictions_tree})

output_log.to_csv('logistic_regression_predictions.csv', index=False)
output_tree.to_csv('decision_tree_predictions.csv', index=False)

