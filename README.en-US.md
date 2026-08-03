

## 2020 First Digital Sichuan Innovation Competition - Algorithm Track - Fraud Call Recognition
### Competition URL: [Fraud Call Recognition](http://www.scdata.net.cn/common/cmpt/%E8%AF%88%E9%AA%97%E7%94%B5%E8%AF%9D%E8%AF%86%E5%88%AB_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)
### Competition Summary

This was a competition I participated in during my spare time. During the semi-finals, it coincided with a busy project deadline for acceptance, so regrettably, I only performed feature engineering on one table and submitted results a few times. I didn't have time to clean and submit results for the other tables, which wasted the opportunity to achieve a better ranking. Nevertheless, participating in the competition was very rewarding, and I'm documenting it here for future reference.

##### 1. Data Description
Preliminary Stage: Predict data for April 2020;

Semi-final Stage: Predict data for May 2020;

Final Stage: Predict data for June 2020
The data fields are described as follows:

Note 1: In the preliminary test set, fields such as arpu_201908, arpu_201909……arpu_202003 do not exist. Instead, there is only one field: arpu_202004.     
   
   ![Image](https://github.com/AiIsBetter/sichuan_voice_phishing2020/blob/main/IMG/train_user.png)
    ![Image](https://github.com/AiIsBetter/sichuan_voice_phishing2020/blob/main/IMG/train_voc.png)
    ![Image](https://github.com/AiIsBetter/sichuan_voice_phishing2020/blob/main/IMG/train_sms.png)
    ![Image](https://github.com/AiIsBetter/sichuan_voice_phishing2020/blob/main/IMG/train_app.png)
    
##### 2. Data Analysis

The preliminary training data goes up to March, and the test data is for April. In the semi-finals, the preliminary test data was used as new training data to predict the May test set. Specific data statistics are available in html_初赛 and html_复赛. Conventionally, statistics are performed on each data table, calculating relevant statistical features for each column, such as missing values, numerical types, numerical distributions, variance, etc., to get an overall impression of the data. At the same time, we check for any anomalies. These anomalies may need to be removed or could be strong features to be leveraged. For example, in this competition, for the 'arpu_xx' field, extracting and checking all positive samples reveals that most are missing. By adding a post-processing step after model prediction to force the prediction results of missing samples to 1, we can significantly improve the online score. This was effective in both the preliminary and semi-final rounds. The main purpose of preliminary data analysis is twofold: to understand the overall data profile and to identify minor anomalies, ensuring we are on the same starting line as other participants. Regarding training sample processing in the preliminary round, there are basically two approaches: 1. Use only March ARPU data as features to train the model, ensuring the model's timeliness and keeping the training data distribution consistent with the test data. 2. Use a sliding window approach, treating each month's data as a new sample, thereby expanding the training set by 8 times. I chose the first approach. After the online results were announced, my offline and online scores were basically the same with no fluctuation, so my ranking improved. I also tried the second method. The offline score skyrocketed, but the online score dropped. I believe this is because the samples obtained via the sliding window share most identical features in the train_user feature table. Even with sliding windows, only features in the other tables change. Since the most important features for this competition are in the user basic feature table, the sliding window did not provide richer feature information, leading only to offline overfitting with poor performance. In the semi-finals, the newly added April data had an even different distribution from the previous months. Therefore, I only used April data for training in the semi-finals. Moreover, since the April data distribution differed from May's, directly using the preliminary feature engineering caused overfitting. However, lacking time for extensive processing, I only made some feature improvements using the voc table, as its data distribution was largely consistent with May's. Later, the semi-final B-list results showed a ranking fluctuation of about 100, which was smaller than the fluctuations of some top-ranking participants, further proving this point.

##### 3. Feature Engineering

The most crucial aspect of this competition was likely feature engineering. For competitions with small datasets, models are primarily based on traditional machine learning models like tree-based models. Deep learning models such as DNN and FFM are actually unsuitable due to the small data volume. Below is a summary of the feature engineering performed on the four tables:

train_user:
'city_name' is one-hot encoded for some high-risk areas separately, e.g., whether it equals 'Guangan', 1 for yes, 0 for no; categorical features are label encoded. Other features use their raw values.

train_voc:
Statistical features for call volume by month, day, and hour; call type statistics by month and day; segmented call duration statistics by month; encrypted number call statistics by month and day; peer-encrypted call count statistics by month and day; geographical and district/county encoded statistical features; monthly geographical change statistics; active days per month (currently defined as active if daily call volume > 20); time interval trends for the last 10, 20, 50, ... outgoing and incoming calls. The final active features and trend features significantly boosted the score. My single voc table score was very high in both the preliminary and semi-final rounds, indicating that the feature information in this table was sufficiently mined and well-aligned with the model's target business requirements.

train_sms:
A blast send is defined as sending to more than 5 people at once. Time is defined using periods with set thresholds, e.g., 0 seconds or within 5 seconds are considered the same time, with multiple statistical passes; SMS sending volume statistics; SMS count trends for intervals of the last 10, 20, 50, ...; time interval statistics and trend changes for sent and received SMS; uplink/downlink SMS ratios and separate statistics; SMS statistics by month, day, and hour.

train_app:
Categorical features are label encoded; busi_name length statistics; flow is discretized into segments.

Some features were not extensively tried due to time constraints, but are listed below for record-keeping:
The app table feels very important to me, but I never managed to mine particularly useful features from it. tf-idf and count vector were tried without obvious results. For the busi_name field of apps, treating it as a sequence feature (understood as a sentence describing customers' daily habits) and applying w2v after sorting by traffic also didn't yield good results, possibly due to improper parameter tuning or incorrect processing methods. Based on test data, I looked up each customer's top N apps by traffic usage, extracted traffic separately, calculated the proportion of total traffic, created a 0-N sorted column, selected the top 0-N app names for each customer, and then encoded them. Traffic rank features. Not sure if they are useful; they easily cause offline overfitting and couldn't be verified via submission.

##### 4. Model

The model chosen was lightgbm. Training used 5-fold cross-validation. For each fold, by iterating through predicted values at different percentiles of the validation set predictions, a set of thresholds was obtained to calculate the f1 score, selecting the threshold that yielded the best f1 score. The average f1 score across the five folds was taken as the local score, and the average of the thresholds was used as the threshold for the test set predictions before submitting the results.

##### 5. Reflections

The focus of this competition was indeed on feature engineering. Effective feature engineering is what brings substantial improvements, whereas model tuning and some clever tricks cannot deliver real results. Additionally, I need to study how to effectively combine w2v with tree-based models. There should be ways to properly mine the sequence features in the app table; I will come back and update this if I figure out a better approach. Through this competition, I gained a better understanding of the telephone fraud business scenario. Hopefully, I won't be scammed in the future, as I've been receiving more of these calls lately.
Attached below is the image of feature importances from the 5-fold cross-validation after model training:

   ![Image](https://github.com/AiIsBetter/sichuan_voice_phishing2020/blob/main/IMG/importances.png)
