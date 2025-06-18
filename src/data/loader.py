import pandas as pd
import os
from typing import Optional
from pathlib import Path

from ..config.constants import Config

class DataLoader:
    """
    Simple data loader with basic preprocessing
    """
    
    def __init__(self):
        self.config = Config()
        self.data_dir = Config.DATA_DIR
    
    def load_data(self, 
                  start_date: str = None, 
                  end_date: str = None,
                  rename_columns: bool = True) -> Optional[pd.DataFrame]:
        """
        Load weather data from CSV file
        
        Args:
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format  
            rename_columns: Whether to rename columns to Vietnamese
            
        Returns:
            DataFrame or None if loading fails
        """
        # Use defaults
        start_date = start_date or Config.DEFAULT_START_DATE
        end_date = end_date or Config.DEFAULT_END_DATE
        
        print(f"📂 LOADING WEATHER DATA")
        
        # Construct file path
        filename = f"hcmc_weather_data_{start_date}_{end_date}.csv"
        filepath = Path(self.data_dir) / filename
        
        try:
            if not filepath.exists():
                print(f"❌ File not found: {filename}")
                return None
            
            # Load data
            df = pd.read_csv(filepath)
            print(f"✅ Data loaded: {df.shape}")
            
            # Rename columns if requested
            if rename_columns:
                df = df.rename(columns=Config.COLUMN_MAPPING)
                print(f"📝 Columns renamed to Vietnamese")
            
            # Convert date column
            df = self._convert_date_column(df)
            
            return df
            
        except Exception as e:
            print(f"❌ Load failed: {str(e)}")
            return None
    
    def _convert_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert date column to datetime format"""
        date_col = 'Ngày' if 'Ngày' in df.columns else 'DATE'
        
        if date_col not in df.columns:
            return df
        
        try:
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col])
                print(f"📅 Date column converted")
            
        except Exception as e:
            print(f"⚠️ Date conversion failed: {str(e)}")
        
        return df 