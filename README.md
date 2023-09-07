# Project Zillow

---

Use an accurate machine learning model to forecast the property tax assessment value for Zillow's single-family residential properties with transaction dates in 2017.

### <u>Project Description</u>

Being the most frequently visited real estate platform in the United States, Zillow and its associated companies provide users with an on-demand experience for selling, buying, renting, and financing, marked by transparency and a nearly seamless end-to-end service. I have chosen to explore the various factors that influence the assessment of property tax values.

### <u>Project Goal</u>

* Identify the factors influencing property values within a particular market.
* Utilize these factors to create a machine learning model that can make precise property value predictions.
* This knowledge will enhance our comprehension of the tax assessment process for single-family properties and their valuation.

### <u>Initial Thoughts</u>

I initially hypothesize that factors such as the number of rooms, square footage, and location will serve as significant determinants of tax-assessed property values.

## <u>The Process</u>

1. **Acquire data from Codeup MySQL DB**

2. **Prepare data**

  * Create developed columns/features from existing data

3. **Explore data in search of drivers of property value**

  * Answer the following initial questions
    * Is there a correlation between area and property value?
    * Is there a correlation between age and property value?
    * Is there a correlation between the room count and property value?
    * Is there a difference in average property value between counties?

  Provide responses to the initial inquiries:
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

| Original                     | Feature    | Type    | Definition                                              |
| :--------------------------- | :--------- | :------ | :------------------------------------------------------ |
| yearbuilt                    | year       | Year    | The year the principal residence was built              |
| bedroomcnt                   | beds       | Numeric | Number of bedrooms in home                              |
| bathroomcnt                  | baths      | Numeric | Number of bathrooms in home including fractional        |
| roomcnt                      | roomcnt    | Numeric | Total number of rooms in the property                   |
| calculatedfinishedsquarefeet | area       | SqFt    | Calculated total finished living area                   |
| taxvaluedollarcnt (target)   | prop_value | USD     | The total tax assessed value of the parcel/home         |
| fips                         | county     | County  | Federal Information Processing Standard (these 3 in CA) |
| latitude                     | latitude   | Numeric | Latitude coordinates of property                        |
| longitude                    | longitude  | Numeric | Longitude coordinates of property                       |
| Additional Features          |            | Numeric | Encoded categorical variables                           |
|                              | age        | Year    | How many years from 2017 since it was built             |

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

* Property value tends to increase with a property's younger age.
* Greater living area typically leads to higher property values.
* The property's location plays a significant role in determining its value.
* There is room for further enhancement in the model's performance.

### Recommendations and Next Steps

* It would nice to have the data to check if the included appliances or the type of heating services (gas or electric) of the property would affect property value

* More time is needed to work on features to better improve the model
    - latitude and longitude could hopefully give insights into cities and neighborhoods with higher or lower property values
    - pools and garages could also be looked into

* Having access to data to investigate whether the presence of appliances or the type of heating system (gas or electric) in a property has an impact on its value would be valuable.

* Additional time is required to enhance the model by incorporating additional features:
    - Exploring latitude and longitude data may offer insights into areas with varying property values within cities and neighborhoods.
    - Analyzing the influence of pools and garages on property values could be beneficial as well.