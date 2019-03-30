# TwitterTextMining
Attaining Tweets of any hashtag and grouping them together using Clustering.

# Overview
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
* <h4>HTML View</h4>
<img width="371" alt="front_end" src="https://user-images.githubusercontent.com/20180559/55277197-67168280-5323-11e9-83e3-d4c7c43d1e92.png">

* <h4>WordCloud of all words</h4>
![twitter_word_cloud](https://user-images.githubusercontent.com/20180559/55277132-e061a580-5322-11e9-9990-a64bc80278d4.png)

* <h4>Group 1</h4>
![Group 1](https://user-images.githubusercontent.com/20180559/55277133-e6f01d00-5322-11e9-9165-a87bd9f7c4df.png)

* <h4>Group 2</h4>
![Group 2](https://user-images.githubusercontent.com/20180559/55277135-e8214a00-5322-11e9-90e3-61d35c4d273c.png)

* <h4>Group 3</h4>
![Group 3](https://user-images.githubusercontent.com/20180559/55277136-e9527700-5322-11e9-8023-0cf43249120b.png)

* <h4>TSNE Graph</h4>
![TSNE](https://user-images.githubusercontent.com/20180559/55277137-ebb4d100-5322-11e9-86bc-d7f10e9e75a3.png)

* <h4>PCA Graph</h4>
![PCA](https://user-images.githubusercontent.com/20180559/55277139-ed7e9480-5322-11e9-9a7f-b5de4822b88f.png)
