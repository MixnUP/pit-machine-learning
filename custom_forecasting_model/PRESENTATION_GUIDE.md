# Custom Forecasting Model - Presentation Guide

## Slide 1: Title and Team Members
- **Title:** Forecasting Energy Consumption: A Custom Linear Regression Approach
- **Team Members:** [Your Names]
- **Date:** [Presentation Date]
- **Institution:** [Your Institution/Organization]
- **Project Focus:** Machine Learning for Energy Consumption Prediction

## Slide 2: Rationale
- **Problem Statement:**
  - Need for accurate energy consumption forecasting
  - Impact on policy-making and resource allocation
- **Project Objectives:**
  - Predict future electricity consumption trends
  - Develop a custom linear regression model
  - Evaluate model performance on historical data
- **Significance:**
  - Supports sustainable energy planning
  - Helps in infrastructure development
  - Aids in policy formulation

## Slide 3: Methodology
### Data Visualization
- Time series visualization of electricity consumption
- Trend analysis and pattern recognition
- Key indicators visualization (from INDICATORS.md in /output)

### Data Process
1. **Data Collection:**
   - Source: [economy-and-growth_phl.csv, energy-and-mining_phl.csv]
   - Time period covered
   - Key indicators used

2. **Data Preprocessing:**
   - Handling missing values
   - Data normalization
   - Train-test split strategy

### Model Development
- **Algorithm:** Custom Linear Regression
- **Features:**
  - Year as the independent variable
  - Electricity consumption as the dependent variable
- **Implementation:**
  - Custom implementation details
  - Training process
  - Model parameters

## Slide 4: Results and Discussion
### Model Evaluation
- **Performance Metrics:**
  - Root Mean Square Error (RMSE): 138.77 kWh per capita
  - Model Parameters:
    - Slope (m): 9.44 (kWh/year)
    - Intercept (c): -18,353.97
- **Visual Analysis:**
  - Linear regression fit on training data (red line)
  - Predictions on test data (purple dashed line)
  - Future forecast (green dashed line)

### Key Findings
- **Trend Analysis:**
  - Strong upward trend in electricity consumption
  - Yearly increase: 9.44 kWh per capita per year
- **Model Performance:**
  - RMSE of 138.77 indicates good prediction accuracy relative to the scale of values
  - Model effectively captures the linear trend in the data
- **Forecast for Next 10 Years:**
  - Predicted increase from 754.75 kWh (2025) to 839.68 kWh (2034)
  - Total increase of 84.93 kWh (11.3%) by 2034
  - Forecasted value for 2025: 754.75 kWh per capita

## Slide 5: Recommendation
- **For Policy Makers:**
  - Infrastructure planning based on forecasted demand
  - Energy policy considerations
- **Future Work:**
  - Incorporate additional features (GDP, population growth, etc.)
  - Try more complex models
  - Expand to regional-level forecasting
- **Final Thoughts:**
  - Summary of key takeaways
  - Potential impact of the model
  - Call to action

## Presentation Tips
1. **Visuals:**
   - Use the generated `custom_forecast_plot.png` in /output folder
   - Include relevant charts and graphs
   - Use consistent color scheme

2. **Talking Points:**
   - Emphasize the custom implementation aspect
   - Highlight the practical applications
   - Be prepared to discuss model limitations

3. **Timing:**
   - Allocate ~2 minutes per slide
   - Leave time for Q&A
   - Practice transitions between slides

## Resources
*   **URL:** [data.humdata.org](https://data.humdata.org/)
*   **Philippine Origin:** The datasets (`economy-and-growth_phl.csv`, `energy-and-mining_phl.csv`) are specifically filtered or provided for the Philippines (indicated by `_phl` in filenames), ensuring relevance to the local context.


---
*Note: This is a template. Please customize the content in [brackets] with your specific information.*
