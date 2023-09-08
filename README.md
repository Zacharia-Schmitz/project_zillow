# Project Zillow

---

Use an accurate machine learning model to forecast the property tax assessment value for Zillow's single-family residential properties with transaction dates in 2017.

### <u>Project Description</u>

As the foremost real estate platform in the United States, Zillow and its affiliated entities offer users a seamless, transparent, and highly accessed platform for selling, purchasing, renting, and securing financing. I have decided to delve into the multiple factors that impact the evaluation of property tax assessments.

### <u>Project Goal</u>

* Recognize the elements that exert influence on property values in a specific market.
* Employ these elements to construct a machine learning model capable of delivering accurate property value forecasts.
* This information will bolster our understanding of the assessment procedure for single-family property taxes and their appraisals.

### <u>Initial Thoughts</u>

I propose that elements like room count, square footage, and location are likely to emerge as significant factors affecting tax-assessed property values.

## <u>The Process</u>

1. **Acquire data from Codeup MySQL DB**

2. **Prepare data**

  * Create developed columns/features from existing data

3. **Explore data in search of drivers of property value**

  * Answer the following initial questions
    * Is there a correlation between `area` and `prop_value`?
    * Is there a correlation between age and `prop_value`?
    * Is there a correlation between the room count and `prop_value`?
    * Is there a difference in average `prop_value` between counties?

  * Provide responses to the initial inquiries:
   * Does a relationship exist between the property's size and its value?
   * Is there a connection between the property's age and its value?
   * Does the number of rooms in a property correlate with its value?
   * Are there variations in average property values among different counties?

4. **Develop a Model to predict property value**

  * Utilize the identified drivers from the exploration phase to construct various types of predictive models.
  * Assess the models using training and validation data.
  * Choose the most suitable model by considering its Root Mean Square Error (RMSE) and R-squared ($R^2$) performance metrics.
  * Evaluate the selected model's performance on the test dataset.

5. **Draw conclusions**

## <u>Data Dictionary</u>

| DF Name    |          DB Name              | dtype | Definition                  |
| ---------- | ----------------------------- | ----- | --------------------------- | 
| `age`      | NA *(2017 - `yearbuilt`)*     | int   | Age of the property         |
| `beds`     | `bedroomcnt`                  | int   | Total bedrooms              |
| `baths`    | `bathroomcnt`                 | int   | Total bathrooms             |
| `area`     | `calculatedfinishedsquarefeet`| int   | Square footage of building  |
| `value`    | `taxvaluedollarcnt`           | int   | Tax value of property (USD) |
| `county`   | `fips`                        | str   | County of California        |
| `latitude` | `latitude`                    | float | Latitude Coordinate         |
| `longitude`| `longitude`                   | float | Longitude Coordinate        |

FIPS County Codes:

* 06037 = LA County, CA
* 06059 = Orange County, CA
* 06111 = Ventura County, CA

## <u>Steps to Reproduce</u>

1) Clone this repo
2) If you have access to Codeup's MySQL DB:
   - Save **env.py** in the repo w/ `user`, `password`, and `host` variables
   - Run notebook
3) If you don't have access:
   - Request access from Codeup
   - Do step 2

# <u>Conclusions</u>

#### Takeaways and Key Findings

* Property value doesn't necessarily correlate with county, but more so the locations that are in each county
* Denser population and living area typically leads to higher property values.
* The property's location plays a significant role in determining its value.
* There is room for further enhancement and feature selection in the model's performance.

### Recommendations and Next Steps

* More time is needed to work on features to better improve the model
    - latitude and longitude could hopefully give insights into cities and neighborhoods with higher or lower property values
    - garages and pools could prove to be beneficial
    - potentially better data collection for some of the features