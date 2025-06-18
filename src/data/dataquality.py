import pandas as pd
import numpy as np
from typing import Dict, Any

class DataQualityAssessment:
    """
    Essential data quality checks for weather data
    """
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'Lượng mưa'):
        self.df = df.copy()
        self.target_col = target_col
        self.n_rows = len(df)
        self.n_cols = len(df.columns)
    
    def assess_quality(self) -> Dict[str, Any]:
        """
        Run essential quality checks
        
        Returns:
            Dictionary with quality assessment results
        """
        print(f"🔍 DATA QUALITY ASSESSMENT")
        print(f"📊 Dataset: {self.df.shape}")
        
        results = {
            'basic_info': self._basic_info(),
            'missing_data': self._check_missing_data(),
            'duplicates': self._check_duplicates(),
            'data_types': self._check_data_types(),
            'target_variable': self._check_target_variable()
        }
        
        # Calculate overall score
        score = self._calculate_score(results)
        results['overall_score'] = score
        
        print(f"🏆 Quality Score: {score}/100")
        
        return results
    
    def _basic_info(self) -> Dict[str, Any]:
        """Basic dataset information"""
        return {
            'shape': self.df.shape,
            'memory_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'columns': list(self.df.columns)
        }
    
    def _check_missing_data(self) -> Dict[str, Any]:
        """Check for missing values"""
        missing_counts = self.df.isnull().sum()
        total_missing = missing_counts.sum()
        missing_pct = (total_missing / (self.n_rows * self.n_cols)) * 100
        
        cols_with_missing = missing_counts[missing_counts > 0].to_dict()
        
        print(f"❓ Missing data: {missing_pct:.2f}%")
        
        return {
            'total_missing': int(total_missing),
            'missing_percentage': round(missing_pct, 2),
            'columns_with_missing': cols_with_missing
        }
    
    def _check_duplicates(self) -> Dict[str, Any]:
        """Check for duplicate rows"""
        duplicates = self.df.duplicated().sum()
        duplicate_pct = (duplicates / self.n_rows) * 100
        
        print(f"🔄 Duplicates: {duplicates} rows ({duplicate_pct:.2f}%)")
        
        return {
            'duplicate_rows': int(duplicates),
            'duplicate_percentage': round(duplicate_pct, 2)
        }
    
    def _check_data_types(self) -> Dict[str, Any]:
        """Check data types"""
        dtypes = self.df.dtypes.value_counts().to_dict()
        return {str(k): int(v) for k, v in dtypes.items()}
    
    def _check_target_variable(self) -> Dict[str, Any]:
        """Check target variable quality"""
        if self.target_col not in self.df.columns:
            return {'exists': False}
        
        target_data = self.df[self.target_col]
        
        return {
            'exists': True,
            'missing_count': int(target_data.isnull().sum()),
            'missing_percentage': round((target_data.isnull().sum() / len(target_data)) * 100, 2),
            'min_value': float(target_data.min()),
            'max_value': float(target_data.max()),
            'mean_value': round(float(target_data.mean()), 2),
            'zero_days': int((target_data == 0).sum()),
            'zero_percentage': round(((target_data == 0).sum() / len(target_data)) * 100, 2)
        }
    
    def _calculate_score(self, results: Dict[str, Any]) -> int:
        """Calculate overall quality score"""
        score = 100
        
        # Deduct for missing data
        missing_pct = results['missing_data']['missing_percentage']
        score -= min(missing_pct * 2, 30)
        
        # Deduct for duplicates
        duplicate_pct = results['duplicates']['duplicate_percentage']
        score -= min(duplicate_pct * 3, 20)
        
        # Deduct for target variable issues
        target = results['target_variable']
        if target['exists']:
            target_missing = target['missing_percentage']
            score -= min(target_missing * 2, 15)
        else:
            score -= 50  # No target variable
        
        return max(int(score), 0)

def assess_data_quality(df: pd.DataFrame, target_col: str = 'Lượng mưa') -> Dict[str, Any]:
    """
    Convenience function to assess data quality
    
    Args:
        df: DataFrame to assess
        target_col: Target variable column name
        
    Returns:
        Quality assessment results
    """
    assessor = DataQualityAssessment(df, target_col)
    return assessor.assess_quality() 