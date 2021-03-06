
# Bitcoin prediction <img src="https://upload.wikimedia.org/wikipedia/commons/4/46/Bitcoin.svg" alt="bitcoin" width="27"/>

## Classroom work 

---



This project is a classroom work. We choose to predict [bitcoin](https://bitcoin.org/) price with a dataset.

Language : <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="python" width="20"/>  **[python](https://www.python.org/)** 

### Analyse

**Type** : forecast, regression

**Input** : (date, price) *existing*

**Output** : (date, price) *prediction*

---

**Data format** : [csv](https://en.wikipedia.org/wiki/Comma-separated_values)

**Data size** : 3,6M *valid rows*

**Data columns** : 8

**Data useful** : numeric *(timestamp, float)*

---

### Dependencies

#### Linear regression
- pandas
- matplotlib.pyplot
- pickle
- sklearn

#### RNN - LSTM
- numpy
- pandas
- pandas_datareader
- matplotlib.pyplot
- datetime
- sklearn.preprocessing (MinMaxScaler)
- tensorflow.keras.layers (Dense, Dropout, LSTM)
- tensorflow.keras.models (Sequential)