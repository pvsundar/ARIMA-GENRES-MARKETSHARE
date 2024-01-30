# Import necessary libraries
import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import itertools
import seaborn as sns

# Streamlit app title
st.title("FBIMOVIE Genre Analysis")

# Introduction text
introduction = """
Created by: **Prof. P.V. (Sundar) Balakrishnan**

**ARIMA Code for Genres Market Share Analysis**
This analysis covers each of the following 12 genres:
`PropertyCrimes`, `Action`, `Adventure`, `BlackComedy`, `Comedy`, `Documentary`, `Drama`, `Horror`, `Musical`, `RomanticComedy`, `Thriller_Suspense`, `Western`.


*Process of Analysis**:
1. Difference the series to achieve stationarity.
2. Conduct the Augmented Dickey-Fuller (ADF) test to check for stationarity.
3. Perform ARIMA modeling and Grid Search to find the best model parameters.
4. Include a dummy variable for the COVID-19 (or other external shock) impact and year of impact as part of the ARIMA model.
"""

st.markdown(introduction)


# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
# Function to perform grid search for ARIMA parameters with a dummy variable
def arima_grid_search_with_dummy(data_series, dummy_series, p_range, d_range, q_range):
    best_aic = float("inf")
    best_order = None
    for order in itertools.product(p_range, d_range, q_range):
        try:
            model = ARIMA(data_series, exog=dummy_series, order=order)
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = order
        except:
            continue
    return best_order, best_aic

# Streamlit app title
st.title("FBIMOVIE Genre Analysis")

# Preloading data
@st.cache_data
def load_data():
    file_path = 'YEARLY-HHI-ZIPF-FBI-GENRE_MS.xlsx'
    return pd.read_excel(file_path, sheet_name='Sheet1')

# Try to load preloaded data, or allow file upload
try:
    FBIMOVIEdata = load_data()
    st.write("Using preloaded data.")
except Exception as e:
    st.error(f"Failed to load preloaded data: {e}")

uploaded_file = st.file_uploader("Or upload your file (this will override preloaded data)")
if uploaded_file is not None:
    FBIMOVIEdata = pd.read_excel(uploaded_file, sheet_name='Sheet1')

# Check if FBIMOVIEdata is loaded
if 'FBIMOVIEdata' in locals() or 'FBIMOVIEdata' in globals():
    # (Continue with your existing Streamlit code here...)
    # Genre Selection
    genre = st.selectbox("Select a Genre", [
        'Action', 'Adventure', 'BlackComedy', 'Comedy', 'Documentary',
        'Drama', 'Horror', 'Musical', 'RomanticComedy', 'Thriller_Suspense',
        'Western'
    ])

    # COVID Dummy Option
    include_covid_dummy = st.checkbox("Include COVID Dummy", value=True)
    covid_year = st.number_input("COVID Year", value=2020, step=1, format="%d")
    # Process data based on user input
    if include_covid_dummy:
        FBIMOVIEdata['COVID'] = (FBIMOVIEdata['YEAR'] == covid_year).astype(int)

    # Processing for the selected genre
    if genre in FBIMOVIEdata.columns:
        series = FBIMOVIEdata[genre].astype(float)  # Convert to float
        series_diff = series.diff().dropna()  # Differencing for stationarity

        # Adjust the 'YEAR' series to match the length of the differenced series
        year_adjusted = FBIMOVIEdata['YEAR'][1:len(series_diff) + 1]

        # Plotting Original and Differenced Series
        st.write(f"Original and Differenced Time Series for {genre}")
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        ax[0].plot(FBIMOVIEdata['YEAR'], series, marker='o')
        ax[0].set_title(f'Original {genre} Time Series')
        ax[0].set_xlabel('Year')
        ax[0].set_ylabel(genre)

        ax[1].plot(year_adjusted, series_diff, marker='o')
        ax[1].set_title(f'Differenced {genre} Time Series')
        ax[1].set_xlabel('Year')
        ax[1].set_ylabel(f'Differenced {genre}')
        st.pyplot(fig)

        # Perform ADF Test
        adf_test = adfuller(series_diff)
        st.write(f'Augmented Dickey-Fuller Test for {genre}:')
        st.write(f'ADF Statistic: {adf_test[0]}')
        st.write(f'p-value: {adf_test[1]}')
        for key, value in adf_test[4].items():
            st.write(f'Critical Value ({key}): {value}')

        # ARIMA Modeling with COVID-19 Impact
        if include_covid_dummy:
            covid_series = FBIMOVIEdata['COVID'][1:]
        else:
            covid_series = None

        best_order_covid, best_aic_covid = arima_grid_search_with_dummy(series_diff, covid_series, range(0, 3), range(0, 3), range(0, 3))

        if best_order_covid:
            covid_arima_model = ARIMA(series_diff, exog=covid_series, order=best_order_covid)
            covid_arima_result = covid_arima_model.fit()
            st.write(covid_arima_result.summary())
            st.write(f"Best ARIMA Model Order: {best_order_covid}")
            st.write(f"Best AIC: {best_aic_covid}")

            # Saving ARIMA Summaries to Files (Optional)
            summary_text = covid_arima_result.summary().as_text()
            summary_filename = f'{genre.lower()}-arima_summary.txt'
            with open(summary_filename, 'w') as file:
                file.write(f"ARIMA Model Summary for {genre} with COVID-19 Impact:\n")
                file.write(summary_text)
                file.write("\n\n")
            st.write(f"ARIMA summaries for {genre} saved to {summary_filename}")
        else:
            st.write(f"No suitable ARIMA model found for {genre} with COVID-19 impact")
import base64
# Function to create a download link
def get_download_link(filename):
    with open(filename, "rb") as file:
        # Read the file and encode it
        file_data = file.read()
        b64 = base64.b64encode(file_data).decode()
    # Create the href string for downloading
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Usage in your Streamlit app
summary_filename = f'{genre.lower()}-arima_summary.txt'
st.markdown(get_download_link(summary_filename), unsafe_allow_html=True)


# Add some space at the end of the app
st.write("")
st.write("")


# The end of the Streamlit app code
