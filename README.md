# AI-Driven Stock Selection and Portfolio Optimization System

An intelligent, data-driven stock selection and portfolio optimization system that integrates financial indicators, macroeconomic variables, and sentiment features extracted from SEC filings. The system employs XGBoost for stock movement prediction and Markowitz Mean-Variance Optimization for portfolio allocation.

## Features

- **Multi-Factor Analysis**
  - Financial indicators
  - Macroeconomic variables (GDP, CPI, Unemployment Rate)
  - Sentiment analysis from SEC filings (10-K and 10-Q)
  - Loughran-McDonald dictionary-based sentiment scoring

- **Machine Learning Pipeline**
  - XGBoost-based stock movement prediction
  - Confidence threshold-based stock selection
  - Model evaluation and visualization tools

- **Portfolio Optimization**
  - Markowitz Mean-Variance Optimization framework
  - Risk-adjusted return maximization
  - Portfolio rebalancing capabilities

- **Sector Coverage**
  - AI Healthcare
  - Fintech
  - Clean Energy
  - Cloud and Big Data
  - Semiconductor

## Prerequisites

- Python 3.6+
- Required Python packages (see requirements.txt):
  - pandas
  - numpy
  - xgboost
  - scikit-learn
  - cvxpy
  - yfinance
  - requests
  - sec-edgar-downloader

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download required data:
   - Place the `Loughran-McDonald_MasterDictionary_1993-2024.csv` file in the project root directory
   - Configure data paths in `workflow_config.yaml`

## Project Structure

```
├── data/                          # Data storage
├── models/                        # Trained model storage
├── outputs/                       # Analysis outputs
├── results/                       # Backtesting results
├── config/                        # Configuration files
├── features/                      # Feature engineering modules
├── portfolio_optimization/        # Portfolio optimization modules
├── edgar_sentiment_extractor.py   # SEC filing sentiment analysis
├── train_xgb.py                   # XGBoost model training
├── optimizer.py                   # Portfolio optimization
├── predict.py                     # Stock prediction
├── analyze.py                     # Analysis tools
└── fastapi_server.py             # API server
```

## Usage

1. **Data Collection and Processing**
```bash
python edgar_sentiment_extractor.py
```

2. **Model Training**
```bash
python train_xgb.py
```

3. **Portfolio Optimization**
```bash
python optimizer.py
```

4. **Prediction and Analysis**
```bash
python predict.py
```

## Configuration

1. Update the configuration in `workflow_config.yaml`:
   - Data paths
   - Model parameters
   - Optimization constraints
   - API settings

2. Configure SEC EDGAR access in `edgar_sentiment_extractor.py`:
```python
SEC_EMAIL = "your-email@example.com"
```

## Results

The system has been backtested over the 2023-2024 period. While demonstrating promising integration of structured and unstructured data, the current implementation shows areas for improvement in:
- Portfolio diversification
- Return magnitude filtering
- Prediction variance reduction

## Future Enhancements

- Formulation of stock selection as a discrete optimization problem
- Expanded stock coverage
- Advanced textual embeddings
- Reinforcement learning for dynamic allocation
- Enhanced risk management strategies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SEC EDGAR database
- Loughran-McDonald financial sentiment dictionary
- XGBoost development team
- Markowitz portfolio theory