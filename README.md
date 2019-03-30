# TwitterTextMining
Attaining Tweets of any hashtag and grouping them together using Clustering.

# Overview
This is the full code for 'CPG Sales Forecasting'. This code helps to identify current sales and predict the forecast for the rest of the year with the current sales numbers. It also identifies the GAP between Actual Sales+forecast versus the initial yearly plan. It identifies the market share in the retailers, products and regions perspective.
This code performs Text Mining on any trending hashtag. The data is then pre-processed by removing stop words, links and numbers. Then, the frequently used words are found. The code finally visualizes the frequent words in total and in groups, and presents it in a front-end HTML view.

# Dependencies
* GloVe (https://nlp.stanford.edu/projects/glove/)
* pandas
* matplotlib
* wordcloud
* tweepy
* sklearn
* flask
  
# Results
* <h4>Retailer Forecast</h4>
![retailer_total](https://user-images.githubusercontent.com/20180559/48884401-67b97c80-ee4a-11e8-93b0-ce6ddb0cdd51.png)

* <h4>PlanVsActual</h4>
![retailerplanvsactual](https://user-images.githubusercontent.com/20180559/48884424-81f35a80-ee4a-11e8-9056-b1632c414c7c.png)

* <h4>Product Category</h4>
![category_bar](https://user-images.githubusercontent.com/20180559/48884458-a2231980-ee4a-11e8-9642-0ff7275cad0d.png)

* <h4>Retailer Region</h4>
![region_bar](https://user-images.githubusercontent.com/20180559/48884492-bd8e2480-ee4a-11e8-8451-4d65fd079ffb.png)

* <h4>Retailers</h4>
![retailer_bar](https://user-images.githubusercontent.com/20180559/48884518-d4cd1200-ee4a-11e8-9dcd-54bbf9fef26d.png)

* <h4>Products</h4>
![upc_bar](https://user-images.githubusercontent.com/20180559/48884532-eca49600-ee4a-11e8-8658-78e093b0fbe3.png)
