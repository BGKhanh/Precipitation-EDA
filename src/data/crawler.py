import requests
import pandas as pd
import os
from datetime import datetime
from typing import Optional

from ..config.constants import Config

class NASAPowerCrawler:
    """
    NASA POWER API Crawler - Focused on data collection only
    """
    
    def __init__(self):
        self.base_url = Config.NASA_API_BASE_URL
        self.parameters = Config.NASA_PARAMETERS
        self.coordinates = Config.HCMC_COORDINATES
        self.output_dir = Config.DATA_DIR
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def crawl_data(self, 
                   start_date: str = None, 
                   end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Crawl weather data from NASA POWER API
        
        Args:
            start_date: Start date in YYYYMMDD format (optional)
            end_date: End date in YYYYMMDD format (optional)
            
        Returns:
            DataFrame with weather data or None if failed
        """
        # Use defaults if not provided
        start_date = start_date or Config.DEFAULT_START_DATE
        end_date = end_date or Config.DEFAULT_END_DATE
        
        print(f"🚀 CRAWLING NASA POWER DATA")
        print(f"📅 Period: {start_date} → {end_date}")
        print(f"📍 Location: {self.coordinates}")
        
        # Validate parameters count
        if len(self.parameters) > Config.NASA_API_MAX_PARAMS:
            print(f"❌ Too many parameters: {len(self.parameters)}")
            return None
        
        # Setup API request
        params = {
            "parameters": ",".join(self.parameters),
            "community": "RE",
            "longitude": self.coordinates["lon"],
            "latitude": self.coordinates["lat"],
            "start": start_date,
            "end": end_date,
            "format": "JSON"
        }
        
        try:
            # Make API call
            print("🌐 Making API request...")
            response = requests.get(
                self.base_url, 
                params=params, 
                timeout=Config.NASA_API_TIMEOUT
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if "messages" in data and data["messages"]:
                print(f"❌ API Error: {data['messages']}")
                return None
            
            # Parse response
            df = self._parse_response(data)
            if df is None:
                return None
                
            print(f"✅ Data crawled: {df.shape}")
            
            # Save data
            saved_df = self.save_data(df, start_date, end_date)
            return saved_df
            
        except Exception as e:
            print(f"❌ Crawl failed: {str(e)}")
            return None
    
    def _parse_response(self, data: dict) -> Optional[pd.DataFrame]:
        """Parse API response to DataFrame"""
        try:
            properties = data.get("properties", {})
            parameters_data = properties.get("parameter", {})
            
            if not parameters_data:
                print("❌ No data in response")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(parameters_data)
            
            # Add metadata
            df["DATE"] = pd.to_datetime(df.index)
            df["LATITUDE"] = self.coordinates["lat"]
            df["LONGITUDE"] = self.coordinates["lon"]
            
            return df
            
        except Exception as e:
            print(f"❌ Parse error: {str(e)}")
            return None
    
    def save_data(self, 
                  df: pd.DataFrame, 
                  start_date: str, 
                  end_date: str) -> Optional[pd.DataFrame]:
        """
        Save DataFrame to CSV file
        
        Args:
            df: DataFrame to save
            start_date: Start date string
            end_date: End date string
            
        Returns:
            Saved DataFrame or None if failed
        """
        try:
            filename = f"hcmc_weather_data_{start_date}_{end_date}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save to CSV
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            print(f"💾 Data saved: {filename}")
            print(f"📊 Shape: {df.shape}")
            
            return df
            
        except Exception as e:
            print(f"❌ Save failed: {str(e)}")
            return None