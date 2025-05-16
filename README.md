# SEC EDGAR Financial Sentiment Analysis

A powerful tool for analyzing sentiment in SEC EDGAR financial reports (10-K and 10-Q) across various industry sectors. This project leverages the Loughran-McDonald financial sentiment dictionary to perform comprehensive sentiment analysis on financial disclosures.

## Features

- Automated download of SEC EDGAR filings (10-K and 10-Q)
- Advanced text cleaning and preprocessing pipeline
- Sentiment analysis using the Loughran-McDonald financial dictionary
- Multi-sector analysis support:
  - AI Healthcare
  - Fintech
  - Clean Energy
  - Cloud and Big Data
  - Semiconductor
- Comprehensive sentiment scoring across multiple dimensions:
  - Positive
  - Negative
  - Uncertainty
  - Litigious
  - Constraining
  - Superfluous

## Prerequisites

- Python 3.6+
- Required Python packages:
  - pandas
  - requests
  - sec-edgar-downloader
  - CustomLMClassifier (included in the project)

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

3. Download the Loughran-McDonald dictionary:
Place the `Loughran-McDonald_MasterDictionary_1993-2024.csv` file in the project root directory.

## Configuration

1. Update the SEC EDGAR email in `edgar_sentiment_extractor.py`:
```python
SEC_EMAIL = "your-email@example.com"  # Replace with your email
```

2. Modify industry groups and tickers in the configuration section if needed:
```python
industry_groups = {
    'AI Healthcare': ["GH", "EXAS", "ILMN", "TDOC", "MDT"],
    # ... other sectors
}
```

## Usage

Run the main script:
```bash
python edgar_sentiment_extractor.py
```

The script will:
1. Download SEC filings for all configured companies
2. Process and clean the text
3. Perform sentiment analysis
4. Save results to `data/edgar/edgar_sentiment_scores.csv`

## Output

The analysis results are saved in CSV format with the following columns:
- ticker: Company stock symbol
- sector: Industry sector
- report_type: Type of SEC filing (10-K or 10-Q)
- file: Path to the analyzed file
- positive: Positive sentiment score
- negative: Negative sentiment score
- uncertainty: Uncertainty score
- litigious: Litigious score
- constraining: Constraining score
- superfluous: Superfluous score

## Project Structure

```
├── edgar_sentiment_extractor.py    # Main analysis script
├── custom_lm_classifier.py         # Custom sentiment classifier
├── requirements.txt               # Project dependencies
├── data/
│   └── edgar/                    # Output directory for SEC filings and results
└── README.md                     # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SEC EDGAR database for financial filings
- Loughran-McDonald financial sentiment dictionary
- sec-edgar-downloader package