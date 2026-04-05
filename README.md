# Recommender System & Pattern Mining for E-Commerce Analytics

This mini project simulates an e-commerce platform and implements two core analytics approaches:

1. Collaborative Filtering for personalized product recommendations
2. Association Rule Mining for discovering frequent purchase patterns

The project also focuses on visualization, recommendation quality evaluation, and benchmarking for performance and scalability.

## Project Goals

- Recommend products to users based on historical interactions and behavior.
- Identify frequent item combinations from purchase data.
- Visualize key insights from recommendation and pattern-mining outputs.
- Evaluate model quality using ranking/recommendation metrics.
- Benchmark computational performance as data scale increases.

## Problem Statement

Given user interaction and product purchase data, build an analytics pipeline that:

- Learns user-user similarity from interaction patterns.
- Generates top-N personalized recommendations.
- Mines frequent itemsets and association rules.
- Produces actionable visual summaries for decision-making.

## Project Plan

1. Data ingestion and cleaning
2. Feature/table construction (user-item matrix + transaction basket)
3. Collaborative filtering implementation
4. Apriori and rule generation
5. Metrics and benchmarking
6. Visualization notebook/dashboard
7. Advanced EDA extension 
8. Model comparison, hybrid reranking, and rule stability extension 
9. Django integration for interactive management use


## Tech Stack

- Python
- pandas, numpy
- scikit-learn (similarity and evaluation support)
- mlxtend (Apriori and association rules)
- matplotlib / seaborn / plotly (visualization)
- Jupyter Notebook 
- Hybrid reranking and model comparison pipeline 
- Segment and temporal rule-stability analysis utilities
- Django (management web application)

## Evaluation Plan

Recommendation quality will be assessed with ranking metrics and catalog quality metrics:

- Precision@K, Recall@K, MAP
- Coverage (catalog/user coverage)
- Diversity (intra-list diversity)
- Comparative evaluation across Popularity Baseline, Item-CF, User-CF, and Hybrid reranking

Pattern-mining quality will be assessed using:

- Support
- Confidence
- Lift
- Segment-aware rule quality (by user group)
- Rule stability across time windows (monthly support consistency)

Performance/scalability will be tracked by:

- Runtime by dataset size
- Memory usage trends
- Throughput for recommendation/rule generation


## Django Management Application

This project includes a Django-based management app in `django_app/`.

### Django Features

- Dashboard with KPI and model snapshot
- User interaction management form
- Recommendation page (User-CF top-N)
- Global and segment-aware association rule views
- Results page for Notebook 6 model metrics
- Django Admin support with interaction log model (`InteractionLog`)