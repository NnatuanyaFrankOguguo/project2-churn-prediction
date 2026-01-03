"""
CLV Analysis Module for Customer Churn Prediction System
=========================================================
This module provides CLV-specific analysis and visualization functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple

# Terminal-style color codes for logging
class LogColors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def log_info(message: str):
    print(f"{LogColors.BLUE}[INFO]{LogColors.RESET} {message}")

def log_success(message: str):
    print(f"{LogColors.GREEN}[SUCCESS]{LogColors.RESET} {message}")


def analyze_clv_by_churn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze CLV distribution by churn status
    
    Args:
        df: DataFrame with CLV and Churn columns
        
    Returns:
        Summary DataFrame
    """
    summary = df.groupby('Churn').agg({
        'CLV': ['mean', 'median', 'std', 'count']
    }).round(2)
    summary.columns = ['Mean CLV', 'Median CLV', 'Std CLV', 'Count']
    summary.index = ['Not Churned', 'Churned']
    return summary


def analyze_churn_by_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze churn rate by CLV quartile
    
    Args:
        df: DataFrame with CLV_quartile and Churn columns
        
    Returns:
        Summary DataFrame
    """
    quartile_order = ['Low', 'Medium', 'High', 'Premium']
    
    summary = df.groupby('CLV_quartile').agg({
        'Churn': ['mean', 'sum', 'count'],
        'CLV': ['mean', 'min', 'max']
    }).round(2)
    
    summary.columns = ['Churn Rate', 'Churned Count', 'Total Count', 
                       'Avg CLV', 'Min CLV', 'Max CLV']
    summary = summary.reindex(quartile_order)
    summary['Churn Rate'] = (summary['Churn Rate'] * 100).round(1)
    
    return summary


def calculate_revenue_at_risk(df: pd.DataFrame) -> Dict:
    """
    Calculate revenue at risk from predicted churn
    
    Args:
        df: DataFrame with CLV and Churn columns
        
    Returns:
        Dictionary with revenue metrics
    """
    churned = df[df['Churn'] == 1]
    not_churned = df[df['Churn'] == 0]
    
    return {
        'total_clv': df['CLV'].sum(),
        'churned_clv': churned['CLV'].sum(),
        'retained_clv': not_churned['CLV'].sum(),
        'revenue_at_risk_pct': (churned['CLV'].sum() / df['CLV'].sum()) * 100,
        'avg_churned_clv': churned['CLV'].mean(),
        'avg_retained_clv': not_churned['CLV'].mean()
    }


def create_clv_distribution_plot(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Create CLV distribution histogram
    
    Args:
        df: DataFrame with CLV column
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(df['CLV'], bins=30, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(df['CLV'].mean(), color='red', linestyle='--', 
               label=f'Mean: ${df["CLV"].mean():.2f}')
    ax.axvline(df['CLV'].median(), color='green', linestyle='--',
               label=f'Median: ${df["CLV"].median():.2f}')
    
    ax.set_xlabel('Customer Lifetime Value ($)')
    ax.set_ylabel('Number of Customers')
    ax.set_title('Distribution of Customer Lifetime Value')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        log_success(f"Saved CLV distribution plot to {save_path}")
    
    return fig


def create_churn_by_quartile_plot(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Create bar chart of churn rate by CLV quartile
    
    Args:
        df: DataFrame with CLV_quartile and Churn columns
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    quartile_order = ['Low', 'Medium', 'High', 'Premium']
    churn_by_quartile = df.groupby('CLV_quartile')['Churn'].mean() * 100
    churn_by_quartile = churn_by_quartile.reindex(quartile_order)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#28a745', '#ffc107', '#dc3545', '#17a2b8']
    bars = ax.bar(quartile_order, churn_by_quartile.values, color=colors, 
                  edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, churn_by_quartile.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('CLV Quartile')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_title('Churn Rate by CLV Quartile')
    ax.set_ylim(0, max(churn_by_quartile.values) + 10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        log_success(f"Saved churn by quartile plot to {save_path}")
    
    return fig


def generate_business_insights(df: pd.DataFrame) -> list:
    """
    Generate business insights from CLV × Churn analysis
    
    Args:
        df: DataFrame with CLV_quartile and Churn columns
        
    Returns:
        List of insight strings
    """
    quartile_order = ['Low', 'Medium', 'High', 'Premium']
    churn_by_quartile = df.groupby('CLV_quartile')['Churn'].mean()
    churn_by_quartile = churn_by_quartile.reindex(quartile_order)
    
    insights = []
    
    # Find highest churn quartile
    max_churn_quartile = churn_by_quartile.idxmax()
    max_churn_rate = churn_by_quartile.max() * 100
    
    insights.append(
        f"🚨 CRITICAL: {max_churn_quartile} CLV customers have the highest churn rate "
        f"at {max_churn_rate:.1f}%. These customers are paying premium prices but "
        f"are dissatisfied - prioritize retention efforts here."
    )
    
    # Find lowest churn quartile
    min_churn_quartile = churn_by_quartile.idxmin()
    min_churn_rate = churn_by_quartile.min() * 100
    
    insights.append(
        f"✅ {min_churn_quartile} CLV customers have the lowest churn rate at "
        f"{min_churn_rate:.1f}%. These are typically long-term contract customers - "
        f"contract lock-in is working."
    )
    
    # Revenue at risk
    revenue_metrics = calculate_revenue_at_risk(df)
    insights.append(
        f"💰 {revenue_metrics['revenue_at_risk_pct']:.1f}% of total CLV "
        f"(${revenue_metrics['churned_clv']:,.0f}) is at risk from churning customers."
    )
    
    return insights


if __name__ == "__main__":
    """Run CLV analysis on processed data"""
    DATA_DIR = r"c:\project2-churn-prediction\data\processed"
    
    # Load data
    log_info("Loading processed data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    all_data = pd.concat([train_df, test_df], ignore_index=True)
    
    # Run analysis
    log_info("=" * 70)
    log_info("CLV ANALYSIS REPORT")
    log_info("=" * 70)
    
    # CLV by churn status
    print("\n📊 CLV by Churn Status:")
    print(analyze_clv_by_churn(all_data))
    
    # Churn by quartile
    print("\n📊 Churn Rate by CLV Quartile:")
    print(analyze_churn_by_quartile(all_data))
    
    # Revenue at risk
    print("\n💰 Revenue at Risk:")
    metrics = calculate_revenue_at_risk(all_data)
    for key, value in metrics.items():
        if 'pct' in key:
            print(f"  {key}: {value:.1f}%")
        elif 'clv' in key:
            print(f"  {key}: ${value:,.2f}")
    
    # Business insights
    print("\n💡 Business Insights:")
    for insight in generate_business_insights(all_data):
        print(f"  • {insight}")
    
    log_success("\nCLV Analysis complete!")
