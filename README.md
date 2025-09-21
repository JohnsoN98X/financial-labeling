# TripleBarrierLabel

A compact, production-ready implementation of **triple-barrier labeling** for financial time series.  
It follows López de Prado’s logic while enforcing **alignment safety** (volatility is shifted by one step to avoid look-ahead) and **clean separation** between the **forward look** (`horizon`) and the **backward look** (`volatility_window`) used for volatility.

> **Note**: This implementation includes a practical extension, `time_to_hit`, which is **not** part of López de Prado’s original presentation.

## 🛠️ Installation

```bash
pip install numpy pandas matplotlib
```

## 📌 Quick Start

```python
from TripleBarrierLabel import TripleBarrierLabel
import pandas as pd

prices = pd.Series(...)               # price series
tbl = TripleBarrierLabel(prices, 20)  # horizon = 20
tbl.fit_labels(up_t=2.0, low_t=2.0, volatility_func="moving_std", volatility_window=50)
```

![Triple Barrier Example](images/tbl-2.jpg)
*Figure 1: Example of triple-barrier labeling with horizon and barriers.*

## 📐 Usage Notes

- **No look-ahead bias**:  
  Barrier levels at time `t` are always set using volatility estimated from prices up to `t-1`.  
  This avoids “peeking ahead” by ensuring today’s barrier does not depend on today’s price.  

- **Index alignment**:  
  Rolling or exponential volatility requires a minimum number of observations to initialize.  
  As a result, the first few entries in the series will be `NaN`. This is expected behavior.  

- **Separation of concepts**:  
  - `horizon`: how far forward the algorithm looks to check if a barrier is hit (future).  
  - `volatility_window`: how many past observations are used to estimate volatility (history).  
  These parameters are independent — you can use a short horizon with a long volatility window, or vice versa.

## 📜 Documentation

This README provides a high-level overview.  
Full API documentation is available in the source code docstrings and can be expanded in a dedicated `docs/` folder or project wiki.  

## 🔮 Future Work

Planned extensions for this project include:

- A dedicated class for **meta-labeling** to enhance predictive power.  
- Adaptation of the code to work with **Imbalance Volume Bars (IVB)** and other non-time-based bars.  
- Integration of **time series embargo** to further reduce data leakage in model training.

## ⚠️ Disclaimer

This project is for research and educational purposes only.  
It is **not** financial advice or a recommendation to trade or invest. Use at your own risk.  

## ⚖️ License

This project is licensed under the MIT License.