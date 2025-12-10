# Stock Analysis Suite

A modern, interactive, and educational financial analysis dashboard built with Streamlit, leveraging Python's powerful data science ecosystem (Pandas, NumPy, Scikit-learn) and **Linear Algebra** principles.

This application provides real-time market overviews and advanced quantitative tools for portfolio management and stock factor analysis.

## Features

The application is structured around several distinct analysis pages, accessible via the sidebar navigation:

* **Market Overview:** A live summary of major US indices (S\&P 500, NASDAQ, Dow Jones) and the CBOE Volatility Index (VIX).
* **Stock Financial Metrics:** Quickly retrieve and display essential financial ratios and metrics for any valid stock ticker (P/E, ROE, FCF, etc.).
* **Portfolio Optimization (Min Volatility/Target Return):**
    * Uses historical stock returns to calculate the **efficient frontier**.
    * Finds the **optimal portfolio weights** to minimize volatility for a given target return constraint (or minimum volatility overall).
    * Displays the **covariance matrix** of asset returns, a key component of Markowitz Portfolio Theory.
* **Stock PCA Analysis (Principal Component Analysis):**
    * Decomposes a stock's historical returns, along with major indices, to identify the main factors (**Principal Components**) driving its movement.
    * Loadings on the first component often represent overall market sensitivity.
* **Linear Regression & Least Squares:**
    * Applies the **Least Squares method** to fit a linear model, predicting one stock's returns based on others.
    * The solution is calculated using the formula $\beta = (X^T X)^{-1} X^T y$, showcasing a core application of linear algebra in finance.
    * Visualizes actual vs. predicted returns and reports the $R^2$ value.

## Installation

To run this application locally, you will need Python and the required libraries.

### Prerequisites

* Python 3.8+

### Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd stock-analysis-suite
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate     # On Windows
    ```

3.  **Install the dependencies:**
    The application requires the following packages, which you can typically install via `pip`:
    ```bash
    pip install streamlit yfinance pandas numpy scikit-learn matplotlib scipy
    # Or install from a requirements.txt file if you create one
    ```

## Usage

Once the dependencies are installed, you can run the Streamlit application from your terminal:

```bash
streamlit run app.py
```

---

## License

This project is open-source and available under the MIT License.

### The MIT License (MIT)

Copyright (c) [2025] [Hannah Friedman]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
