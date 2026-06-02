# cell 1: импорты и настройки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score,
                             accuracy_score, f1_score, precision_score, recall_score)

# cell 2: загрузка данных
df = pd.read_csv('adult.csv')
df.head()
# cell 3: первичный осмотр
print(df.shape)
print(df.columns.tolist())
df.info()
df['Exited'].value_counts(dropna=False)
# cell 4: базовые визуализации
print('Доля ушедших:', df['Exited'].mean())
# Распределение по возрасту / балансу / кредитному рейтингу
fig, axes = plt.subplots(1, 3, figsize=(18,4))
sns.histplot(df['Age'], bins=30, ax=axes[0])
axes[0].set_title('Age')
sns.histplot(df['Balance'], bins=30, ax=axes[1])
axes[1].set_title('Balance')
sns.histplot(df['CreditScore'], bins=30, ax=axes[2])
axes[2].set_title('CreditScore')
plt.tight_layout()
plt.show()

# Сравнение доли ушедших по Geography и Gender
fig, axes = plt.subplots(1,2,figsize=(12,4))
sns.barplot(x='Geography', y='Exited', data=df, ax=axes[0])
axes[0].set_title('Частота ухода по Geography')
sns.barplot(x='Gender', y='Exited', data=df, ax=axes[1])
axes[1].set_title('Частота ухода по Gender')
plt.show()

# cell 5: подготовка фич и таргета
# Уберём идентификаторы и фамилии — они не должны влиять на модель
df_model = df.copy()
drop_cols = ['RowNumber', 'CustomerId', 'Surname']
df_model = df_model.drop(columns=[c for c in drop_cols if c in df_model.columns])

X = df_model.drop(columns=['Exited'])
y = df_model['Exited']

# явный список числовых/категориальных признаков (по описанию датасета)
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProduct', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
categorical_features = ['Geography', 'Gender']

# cell 6: создаём трансформеры
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),   # если есть пропуски
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))  # drop='first' сокращает мультиколлинеарность
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# cell 7: train-test split (стратифицированно по y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print('Train:', X_train.shape, 'Test:', X_test.shape)
print('Доля ушедших в train:', y_train.mean(), 'в test:', y_test.mean())

# cell 8: pipeline + gridsearch
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(solver='liblinear', max_iter=1000))
])

param_grid = {
    'clf__penalty': ['l2'],            # для простоты: L2-регуляризация
    'clf__C': [0.01, 0.1, 1, 10],      # обратная сила регуляризации
    'clf__class_weight': [None, 'balanced']  # учитываем/не учитываем дисбаланс
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print('Best params:', grid.best_params_)
print('Best CV AUC:', grid.best_score_)

# cell 9: оценки на тесте
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:,1]

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1:', f1_score(y_test, y_pred))
print('ROC AUC:', roc_auc_score(y_test, y_proba))
print('\nClassification report:\n', classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed','Exited'], yticklabels=['Stayed','Exited'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
plt.show()

# cell 10: ROC и PR curves
fpr, tpr, _ = roc_curve(y_test, y_proba)
precision, recall, _ = precision_recall_curve(y_test, y_proba)
avg_prec = average_precision_score(y_test, y_proba)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test,y_proba):.3f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend()

plt.subplot(1,2,2)
plt.plot(recall, precision, label=f'AP = {avg_prec:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()

plt.tight_layout()
plt.show()

# cell 11: извлечение имен признаков и коэффициентов
preproc = best_model.named_steps['preprocessor']
num_cols = numeric_features

# имена one-hot-кодуемых категориальных признаков
ohe = preproc.named_transformers_['cat'].named_steps['onehot']
cat_ohe_names = ohe.get_feature_names_out(categorical_features)

feature_names = np.concatenate([num_cols, cat_ohe_names])
coefs = best_model.named_steps['clf'].coef_[0]

coef_df = pd.DataFrame({'feature': feature_names, 'coef': coefs})
coef_df['odds_ratio'] = np.exp(coef_df['coef'])
coef_df = coef_df.sort_values(by='coef', ascending=False)
coef_df.reset_index(drop=True, inplace=True)
coef_df

# cell 12: подбор порога по метрике (например, максимальный F1)
thresholds = np.linspace(0,1,101)
f1_scores = [f1_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]
print('Best threshold by F1:', best_thresh, 'F1:', f1_scores[best_idx])

plt.figure(figsize=(6,4))
plt.plot(thresholds, f1_scores)
plt.axvline(best_thresh, color='red', linestyle='--', label=f'best={best_thresh:.2f}')
plt.xlabel('Threshold')
plt.ylabel('F1 score')
plt.title('F1 vs threshold')
plt.legend()
plt.show()

# применяем выбранный порог
y_pred_thresh = (y_proba >= best_thresh).astype(int)
print('Precision:', precision_score(y_test, y_pred_thresh))
print('Recall:', recall_score(y_test, y_pred_thresh))
print('F1:', f1_score(y_test, y_pred_thresh))

# cell 13: предсказанные вероятности для теста (и соединение с id, если нужно)
probas_df = X_test.copy()
probas_df['y_true'] = y_test
probas_df['proba_exit'] = y_proba
probas_df_sorted = probas_df.sort_values('proba_exit', ascending=False)
probas_df_sorted.head(20)  # топ-20 клиентов с наибольшей вероятностью ухода

# cell 14: сохранить пайплайн
joblib.dump(best_model, 'logreg_churn_pipeline.joblib')
print('Model saved as logreg_churn_pipeline.joblib')
