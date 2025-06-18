# =============================================================================
# COMPONENT 2: EXTREME EVENTS ANALYZER (REFACTORED)
# =============================================================================

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from ..config.constants import Config

warnings.filterwarnings('ignore')


class ExtremeEventsAnalyzer:
    """
    Component 2: Extreme Events Analysis
    ✅ REFACTORED: Loại bỏ BaseAnalyzer, standalone module
    """

    def __init__(self, df: pd.DataFrame, target_col: str = 'Lượng mưa', date_col: str = 'Ngày'):
        """
        Initialize ExtremeEventsAnalyzer with configurable parameters
        
        Args:
            df: DataFrame containing time series data
            target_col: Target column name for extreme event analysis
            date_col: Date column name
        """
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col
        
        # Prepare data
        self._prepare_data()

    def _prepare_data(self) -> None:
        """Prepare data for extreme events analysis"""
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_col]):
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors='coerce')
        
        # Add time features if not present
        if 'Month' not in self.df.columns:
            self.df['Month'] = self.df[self.date_col].dt.month
        if 'Year' not in self.df.columns:
            self.df['Year'] = self.df[self.date_col].dt.year
        
        # Sort by date
        self.df = self.df.sort_values(self.date_col).reset_index(drop=True)

    def analyze(self) -> Dict[str, Any]:
        """
        Run comprehensive extreme events analysis
        
        Returns:
            Dict containing all extreme events analysis results
        """
        print("\n" + "="*70)
        print("⛈️ COMPONENT 2: EXTREME EVENTS ANALYSIS")
        print("="*70)

        try:
            # Run extreme events analysis
            extreme_definition = self._define_extreme_events()
            seasonal_patterns = self._seasonal_extreme_patterns()

            # Combine results
            results = {
                'extreme_definition': extreme_definition,
                'seasonal_patterns': seasonal_patterns,
                'component_name': 'ExtremeEventsAnalyzer_Refactored',
                'success': True
            }

            print(f"\n✅ EXTREME EVENTS ANALYSIS COMPLETED")
            return results

        except Exception as e:
            print(f"\n❌ EXTREME EVENTS ANALYSIS FAILED: {e}")
            return {
                'success': False,
                'error': str(e),
                'component_name': 'ExtremeEventsAnalyzer_Refactored'
            }

    def _define_extreme_events(self) -> Dict[str, Any]:
        """
        2.1 Define extreme events and thresholds
        
        Returns:
            Dict containing extreme event thresholds and counts
        """
        print("🔍 2.1 Extreme Events Definition")
        print("-" * 50)

        try:
            # Validate target column
            if self.target_col not in self.df.columns:
                raise ValueError(f"Target column '{self.target_col}' not found in DataFrame")

            # Remove NaN values for quantile calculation
            clean_data = self.df[self.target_col].dropna()
            
            if len(clean_data) == 0:
                raise ValueError("No valid data available for extreme events analysis")

            # Define thresholds
            p95 = clean_data.quantile(0.95)
            p99 = clean_data.quantile(0.99)
            p99_9 = clean_data.quantile(0.999)

            print(f"   📊 Extreme Event Thresholds:")
            print(f"      - 95th percentile: {p95:.2f}mm")
            print(f"      - 99th percentile: {p99:.2f}mm")
            print(f"      - 99.9th percentile: {p99_9:.2f}mm")

            # Count extreme events
            extreme_95 = (clean_data > p95).sum()
            extreme_99 = (clean_data > p99).sum()
            extreme_99_9 = (clean_data > p99_9).sum()

            total_count = len(clean_data)
            
            print(f"\n   📈 Extreme Event Counts:")
            print(f"      - > 95th percentile: {extreme_95} events ({extreme_95/total_count*100:.2f}%)")
            print(f"      - > 99th percentile: {extreme_99} events ({extreme_99/total_count*100:.2f}%)")
            print(f"      - > 99.9th percentile: {extreme_99_9} events ({extreme_99_9/total_count*100:.2f}%)")

            return {
                'thresholds': {
                    'p95': float(p95), 
                    'p99': float(p99), 
                    'p99_9': float(p99_9)
                },
                'extreme_counts': {
                    'p95': int(extreme_95), 
                    'p99': int(extreme_99), 
                    'p99_9': int(extreme_99_9)
                },
                'total_records': int(total_count),
                'success': True
            }

        except Exception as e:
            print(f"   ❌ Error in extreme events definition: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _seasonal_extreme_patterns(self) -> Dict[str, Any]:
        """
        2.2 Seasonal patterns of extreme events
        
        Returns:
            Dict containing seasonal extreme events analysis results
        """
        print(f"\n🌪️ 2.2 Seasonal Patterns of Extreme Events")
        print("-" * 50)

        try:
            # Use 95th percentile as extreme threshold
            clean_data = self.df[self.target_col].dropna()
            p95 = clean_data.quantile(0.95)
            
            # Filter extreme events
            extreme_mask = self.df[self.target_col] > p95
            extreme_events = self.df[extreme_mask].copy()
            
            if len(extreme_events) == 0:
                print(f"   ⚠️ No extreme events found above {p95:.2f}mm threshold")
                return {
                    'success': False,
                    'error': 'No extreme events found'
                }

            # Group by month
            extreme_by_month = extreme_events.groupby('Month').size()

            # Create comprehensive visualization
            self._create_extreme_events_visualization(extreme_events, p95, extreme_by_month)

            # Calculate return periods
            sorted_values = self.df[self.target_col].sort_values(ascending=False)
            return_periods = len(self.df) / (np.arange(1, len(sorted_values) + 1))

            # Peak extreme month
            peak_month = extreme_by_month.idxmax() if len(extreme_by_month) > 0 else None
            if peak_month:
                print(f"   📅 Peak extreme events month: {peak_month}")

            return {
                'extreme_by_month': extreme_by_month.to_dict(),
                'extreme_events_data': extreme_events,
                'peak_month': int(peak_month) if peak_month else None,
                'return_periods': return_periods[:100].tolist(),
                'sorted_values': sorted_values.iloc[:100].tolist(),
                'threshold_p95': float(p95),
                'total_extreme_events': len(extreme_events),
                'success': True
            }

        except Exception as e:
            print(f"   ❌ Error in seasonal patterns analysis: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _create_extreme_events_visualization(self, 
                                           extreme_events: pd.DataFrame,
                                           p95: float,
                                           extreme_by_month: pd.Series) -> None:
        """
        Create comprehensive extreme events visualization
        
        Args:
            extreme_events: DataFrame containing extreme events
            p95: 95th percentile threshold
            extreme_by_month: Monthly extreme events count
        """
        try:
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle('Extreme Events Analysis', fontsize=16, fontweight='bold')

            # 1. Monthly extreme events
            if len(extreme_by_month) > 0:
                axes[0, 0].bar(extreme_by_month.index, extreme_by_month.values, 
                              color='red', alpha=0.7)
                axes[0, 0].set_title(f'Extreme Events by Month (> {p95:.1f}mm)')
                axes[0, 0].set_xlabel('Month')
                axes[0, 0].set_ylabel('Number of Events')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].set_xticks(range(1, 13))
            else:
                axes[0, 0].text(0.5, 0.5, 'No extreme events found', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)

            # 2. Extreme events time series
            if len(extreme_events) > 0:
                axes[0, 1].scatter(extreme_events[self.date_col], extreme_events[self.target_col],
                                  color='red', alpha=0.7, s=30)
                axes[0, 1].set_title('Extreme Events Over Time')
                axes[0, 1].set_xlabel('Date')
                axes[0, 1].set_ylabel('Precipitation (mm)')
                axes[0, 1].grid(True, alpha=0.3)
                # Rotate x-axis labels for better readability
                axes[0, 1].tick_params(axis='x', rotation=45)
            else:
                axes[0, 1].text(0.5, 0.5, 'No extreme events to plot', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)

            # 3. Distribution of extreme events
            if len(extreme_events) > 0:
                axes[1, 0].hist(extreme_events[self.target_col], bins=20, 
                               color='orange', alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Distribution of Extreme Events')
                axes[1, 0].set_xlabel('Precipitation (mm)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No extreme events to plot', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)

            # 4. Return period analysis
            if len(self.df) > 100:  # Only if sufficient data
                sorted_values = self.df[self.target_col].sort_values(ascending=False)
                return_periods = len(self.df) / (np.arange(1, len(sorted_values) + 1))
                
                # Plot top 100 values
                axes[1, 1].loglog(return_periods[:100], sorted_values.iloc[:100], 
                                 'bo-', markersize=4)
                axes[1, 1].set_title('Return Period Analysis')
                axes[1, 1].set_xlabel('Return Period (days)')
                axes[1, 1].set_ylabel('Precipitation (mm)')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient data for return period analysis', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"   ⚠️ Visualization error: {e}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_extreme_events(df: pd.DataFrame,
                          target_col: str = 'Lượng mưa',
                          date_col: str = 'Ngày') -> Dict[str, Any]:
    """
    Convenience function for complete extreme events analysis
    
    Args:
        df: DataFrame with time series data
        target_col: Target column name
        date_col: Date column name
        
    Returns:
        Dict containing all extreme events analysis results
    """
    analyzer = ExtremeEventsAnalyzer(df, target_col, date_col)
    return analyzer.analyze()

def get_extreme_thresholds(df: pd.DataFrame,
                          target_col: str = 'Lượng mưa',
                          percentiles: List[float] = [0.95, 0.99, 0.999]) -> Dict[str, float]:
    """
    Get extreme event thresholds for specified percentiles
    
    Args:
        df: DataFrame with precipitation data
        target_col: Target column name
        percentiles: List of percentiles to calculate
        
    Returns:
        Dict mapping percentile names to threshold values
    """
    clean_data = df[target_col].dropna()
    
    if len(clean_data) == 0:
        raise ValueError("No valid data available for threshold calculation")
    
    thresholds = {}
    for p in percentiles:
        if p == 0.95:
            thresholds['p95'] = float(clean_data.quantile(p))
        elif p == 0.99:
            thresholds['p99'] = float(clean_data.quantile(p))
        elif p == 0.999:
            thresholds['p99_9'] = float(clean_data.quantile(p))
        else:
            thresholds[f'p{int(p*1000)}'] = float(clean_data.quantile(p))
    
    return thresholds

def count_extreme_events(df: pd.DataFrame,
                        target_col: str = 'Lượng mưa',
                        threshold: float = None,
                        percentile: float = 0.95) -> int:
    """
    Count extreme events above a threshold or percentile
    
    Args:
        df: DataFrame with precipitation data
        target_col: Target column name
        threshold: Absolute threshold value (if None, use percentile)
        percentile: Percentile threshold if absolute threshold not provided
        
    Returns:
        Number of extreme events
    """
    clean_data = df[target_col].dropna()
    
    if len(clean_data) == 0:
        return 0
    
    if threshold is None:
        threshold = clean_data.quantile(percentile)
    
    return int((clean_data > threshold).sum())
