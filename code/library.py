import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin

# corr이 0.8 이상인 경우 변수 제거
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.final_cols = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        corr_matrix = X_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.threshold)]
        self.final_cols = [col for col in X_df.columns if col not in to_drop]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df[self.final_cols]

class NumericFiltering(BaseEstimator, TransformerMixin):
    def __init__(self, check_const_col=True, check_id_col=True):
        self.check_const_col = check_const_col
        self.check_id_col = check_id_col

    def fit(self, X, y=None):
        X = np.array(X)
        if self.check_const_col:
            self.constant_col = [i for i in range(X.shape[1]) if X[:, i].std() == 0]
        else:
            self.constant_col = []

        if self.check_id_col:
            self.id_col = [i for i in range(X.shape[1]) if len(np.unique(np.diff(X[:, i]))) == 1]
        else:
            self.id_col = []

        self.rm_cols = self.constant_col + self.id_col
        self.final_cols = [i for i in range(X.shape[1]) if i not in self.rm_cols]
        return self

    def transform(self, X):
        X = np.array(X)
        result = X[:, self.final_cols]
        return result

class CategoricalFiltering(BaseEstimator, TransformerMixin):
    def __init__(self, check_const_col=True, check_id_col=True, check_cardinality=True):
        self.check_const_col = check_const_col
        self.check_id_col = check_id_col
        self.check_cardinality = check_cardinality

    def fit(self, X, y=None):
        X = np.array(X)
        if self.check_const_col:
            self.constant_col = [i for i in range(X.shape[1]) if len(np.unique(X[:, i])) == 1]
        else:
            self.constant_col = []

        if self.check_id_col:
            self.id_col = [i for i in range(X.shape[1]) if len(np.unique(X[:, i])) == X.shape[0]]
        else:
            self.id_col = []

        if self.check_cardinality:
            self.cardinality = [i for i in range(X.shape[1]) if len(np.unique(X[:, i])) > 50]
        else:
            self.cardinality = []

        self.rm_cols = self.constant_col + self.id_col + self.cardinality
        self.final_cols = [i for i in range(X.shape[1]) if i not in self.rm_cols]
        return self

    def transform(self, X):
        X = np.array(X)
        result = X[:, self.final_cols]
        return result

class EnsemblePipeline:
    def __init__(self):
        pipe1 = Pipeline([
            ('step1', SimpleImputer(strategy="mean")),
            ('step2', NumericFiltering()),
            ('step3', StandardScaler()),
            ('step4', CorrelationFilter(threshold=0.8)),
        ])

        pipe2 = Pipeline([
            ('step1', SimpleImputer(strategy="most_frequent")),
            ('step2', CategoricalFiltering()),
            ('step3', OneHotEncoder()),
        ])

        transform = ColumnTransformer([
            ('num', pipe1, make_column_selector(dtype_include=np.number)),
            ('cat', pipe2, make_column_selector(dtype_exclude=np.number)),
        ])

        self.models = {
            'RF': RandomForestClassifier(),
            'LGBM': LGBMClassifier(),
            'XGBM': XGBClassifier()
        }

        ensemble_models = []
        for model in self.models.keys():
            pipe0 = ImbPipeline([
                ('transform', transform),
                ('smote', SMOTE()),
                ('model', self.models[model])
            ])
            ensemble_models.append((model, pipe0))

        self.ensemble_pipe = StackingClassifier(estimators=ensemble_models, final_estimator=XGBClassifier())

    def fit(self, X, y):
        self.columns = X.columns.tolist()
        self.ensemble_pipe.fit(X, y)
        self.feature_names = self._get_final_feature_names(X)
        self.imp = permutation_importance(estimator=self.ensemble_pipe,
                                          X=X, y=y,
                                          scoring="accuracy",
                                          n_repeats=5)
        self.feature_importances_ = pd.DataFrame(self.imp['importances_mean'], index=self.columns, columns=['importance']).sort_values('importance', ascending=False)
        return self

    def _get_final_feature_names(self, X):
        # Get feature names from the ColumnTransformer
        num_features = X.select_dtypes(include=np.number).columns
        cat_features = X.select_dtypes(exclude=np.number).columns
        
        numeric_transformer = self.ensemble_pipe.named_estimators_['RF'].named_steps['transform'].named_transformers_['num']
        categorical_transformer = self.ensemble_pipe.named_estimators_['RF'].named_steps['transform'].named_transformers_['cat']
        
        num_features_final = num_features[numeric_transformer.named_steps['step4'].final_cols]
        cat_features_final = categorical_transformer.named_steps['step3'].get_feature_names_out(cat_features[categorical_transformer.named_steps['step2'].final_cols])
        
        return list(num_features_final) + list(cat_features_final)
        
    def predict(self, X):
      pred = pd.DataFrame()
      for model in self.models.keys():
        pred[model] = self.ensemble_pipe.named_estimators_[model].predict(X)
      # 각 분류기의 예측값을 평균하여 앙상블 예측값 반환
      pred['ensemble'] = pred.mean(axis=1)
      return pred

    def predict_proba(self, X):
        pred = pd.DataFrame(self.ensemble_pipe.predict_proba(X)[:, 1], columns=['ensemble'])
        for model in self.models.keys():
            pred[model] = self.ensemble_pipe.named_estimators_[model].predict_proba(X)[:, 1]
        return pred

    def feature_importances(self):
        return self.feature_importances_

    

    def get_feature_names(self):
        return self.feature_names
