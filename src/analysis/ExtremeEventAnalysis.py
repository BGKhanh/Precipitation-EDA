# =============================================================================
# COMPONENT 2: EXTREME EVENTS ANALYZER
# =============================================================================

class ExtremeEventsAnalyzer(BaseAnalyzer):
    """
    Component 2: Extreme Events Analysis
    """

    def analyze(self):
        """Run comprehensive extreme events analysis"""
        print("\n" + "="*70)
        print("⛈️ COMPONENT 2: EXTREME EVENTS ANALYSIS")
        print("="*70)

        # Run extreme events analysis
        extreme_definition = self._define_extreme_events()
        seasonal_patterns = self._seasonal_extreme_patterns()

        # Combine results
        results = {
            'extreme_definition': extreme_definition,
            'seasonal_patterns': seasonal_patterns,
            'component_name': 'ExtremeEventsAnalyzer'
        }

        return results

    def _define_extreme_events(self):
        """2.1 Define extreme events and thresholds"""
        print("🔍 2.1 Extreme Events Definition")
        print("-" * 50)

        # Define thresholds
        p95 = self.df[self.target_col].quantile(0.95)
        p99 = self.df[self.target_col].quantile(0.99)
        p99_9 = self.df[self.target_col].quantile(0.999)

        print(f"   📊 Extreme Event Thresholds:")
        print(f"      - 95th percentile: {p95:.2f}mm")
        print(f"      - 99th percentile: {p99:.2f}mm")
        print(f"      - 99.9th percentile: {p99_9:.2f}mm")

        # Count extreme events
        extreme_95 = (self.df[self.target_col] > p95).sum()
        extreme_99 = (self.df[self.target_col] > p99).sum()
        extreme_99_9 = (self.df[self.target_col] > p99_9).sum()

        print(f"\n   📈 Extreme Event Counts:")
        print(f"      - > 95th percentile: {extreme_95} events ({extreme_95/len(self.df)*100:.2f}%)")
        print(f"      - > 99th percentile: {extreme_99} events ({extreme_99/len(self.df)*100:.2f}%)")
        print(f"      - > 99.9th percentile: {extreme_99_9} events ({extreme_99_9/len(self.df)*100:.2f}%)")

        return {
            'thresholds': {'p95': p95, 'p99': p99, 'p99_9': p99_9},
            'extreme_counts': {'p95': extreme_95, 'p99': extreme_99, 'p99_9': extreme_99_9}
        }

    def _seasonal_extreme_patterns(self):
        """2.2 Seasonal patterns of extreme events"""
        print(f"\n🌪️ 2.2 Seasonal Patterns of Extreme Events")
        print("-" * 50)

        # Use 95th percentile as extreme threshold
        p95 = self.df[self.target_col].quantile(0.95)
        extreme_by_month = self.df[self.df[self.target_col] > p95].groupby('Month').size()

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Extreme Events Analysis', fontsize=16, fontweight='bold')

        # Monthly extreme events
        axes[0,0].bar(extreme_by_month.index, extreme_by_month.values, color='red', alpha=0.7)
        axes[0,0].set_title(f'Extreme Events by Month (> {p95:.1f}mm)')
        axes[0,0].set_xlabel('Month')
        axes[0,0].set_ylabel('Number of Events')
        axes[0,0].grid(True, alpha=0.3)

        # Extreme events time series
        extreme_events = self.df[self.df[self.target_col] > p95]
        if len(extreme_events) > 0:
            axes[0,1].scatter(extreme_events[self.date_col], extreme_events[self.target_col],
                            color='red', alpha=0.7, s=30)
            axes[0,1].set_title('Extreme Events Over Time')
            axes[0,1].set_xlabel('Date')
            axes[0,1].set_ylabel('Precipitation (mm)')
            axes[0,1].grid(True, alpha=0.3)

        # Distribution of extreme events
        if len(extreme_events) > 0:
            axes[1,0].hist(extreme_events[self.target_col], bins=20, color='orange', alpha=0.7)
            axes[1,0].set_title('Distribution of Extreme Events')
            axes[1,0].set_xlabel('Precipitation (mm)')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].grid(True, alpha=0.3)

        # Return period analysis
        sorted_values = self.df[self.target_col].sort_values(ascending=False)
        return_periods = len(self.df) / (np.arange(1, len(sorted_values) + 1))
        axes[1,1].loglog(return_periods[:100], sorted_values.iloc[:100], 'bo-', markersize=4)
        axes[1,1].set_title('Return Period Analysis')
        axes[1,1].set_xlabel('Return Period (days)')
        axes[1,1].set_ylabel('Precipitation (mm)')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Peak extreme month
        peak_month = extreme_by_month.idxmax() if len(extreme_by_month) > 0 else None
        if peak_month:
            print(f"   📅 Peak extreme events month: {peak_month}")

        return {
            'extreme_by_month': extreme_by_month,
            'extreme_events_data': extreme_events,
            'peak_month': peak_month,
            'return_periods': return_periods[:100].tolist(),
            'sorted_values': sorted_values.iloc[:100].tolist()
        }
