# **COMPUTER SCIENCE FINAL YEAR PROJECT**

# **BY IBITOKUN MOROUNFOLUWA DAVID**

# **2021/10521**

# **COLLEGE OF NATURAL AND APPLIED SCIENCES,**

# **BELLS UNIVERSITY OF TECHNOLOGY,**

# **IN PARTIAL FULFILLMENT OF THE REQUIREMENT FOR THE AWARD OF BACHELOR'S DEGREE IN TECHNOLOGY COMPUTER SCIENCE (B.TECH)** 

# **PROJECT TOPIC: ASSESSING THE IMPACT OF CLIMATE CHANGE ON FOOD SECURITY IN NIGERIA USING DEEP LEARNING TECHNIQUES**

# **CHAPTER 1**

## **1.1 Introduction**

The climate of the earth before the climate change began had a natural balance, which ensured predictable weather patterns and stable agricultural cycles. Temperatures, rainfall and humidity were subject to regular seasonal patterns, which helped sustain ecosystems and traditional farming systems that depended on these patterns to be sustainable. In Nigeria, the farmers were relying on the predictability of the rains, fertile soils, and predictable temperature levels to produce the major crops including maize, yam, and cassava without the uncertainty that exists today due to extreme climatic changes (Calvin et al., 2023; Effiong, 2024; *The State of Food Security and Nutrition in the World 2023*, 2023; Udeh et al., 2024).

Climate change is among the largest environmental and developmental issues of the 21st century. It is simply a lasting change in weather patterns, that is, temperature, rain, humidity, and wind patterns which is to a large extent caused by us humans. The combustion of fossil fuels, forest clearing and all the industrial emissions are pouring the greenhouse gases into the air in large quantities, particularly CO₂. The gases cause solar heat to be trapped, and they heat the planet, thus causing global warming and climate instability (Ezekwe et al., 2024). At this point, with the planet continuing to get warmer, weather patterns are becoming more extreme, and extreme temperatures, unpredictable rainfall, floods, and droughts are all over.

In Nigeria, the effects of climate change are becoming more obvious and threatening. Rural and peri-urban communities are experiencing unpredictable rainfall Lots of areas are experiencing massive floods which destroy crops and homes, whilst others are enduring extended droughts which dry rivers and farmland. Increasing temperatures, particularly in the north, are reducing soil moisture and reducing the ability to cultivate crops; the south, in turn, experiences flooding and erosion, which means that fertile land is destroyed (Effiong, 2024; Udeh et al., 2024). These changes have a great danger on agriculture since agriculture is the engine of the Nigerian economy.

This implies the industry will be highly exposed to weather fluctuations. Any slight shift in temperature or rainfall has enormous impacts on the crop output, livestock production and the livelihoods of millions of smallholders. There is already the loss of crops such as maize, rice, cassava due to floods, erosion, and heat stress (Akinkuolie et al., 2025).

The Food and Agriculture Organisation (FAO) characterises food security as the state wherein "all people, at all times, have physical, social, and economic access to sufficient, safe, and nutritious food" (*The State of Food Security and Nutrition in the World 2023*, 2023). Unpredictable weather patterns disrupt the cultivation and harvesting periods of crops; extreme temperatures result in dehydration of soils; and flooding destroys the yield of crops, and leading to a decrease in the nutrient content of soils. When agricultural productivity decreases, food costs increase, intensifying issues of hunger, undernourishment, and economic hardship, while compromising both national economic resilience and public health outcomes (Akinkuolie et al., 2025; Effiong, 2024).

The largest factors among the key climate variables that strike agriculture are the CO₂ emissions, temperature, and rainfall. Although plants require CO₂ to photosynthesise, excess of it (perhaps when mixed with either heat or moisture stresses) can suffocate crop growth and trigger pest epidemics. The temperature alters the growth rate and evapotranspiration rates of plants, whereas rainfall determines the availability of water to crops and animals (Udeh et al., 2024). Whenever any of these factors is off-balance, the productivity of food and its sustainability is severely affected.

According to the recent studies, more sophisticated methods should be implemented to measure and model the interaction of the climatic variables to influence the agricultural productivity and food security. Machine-learning and deep-learning models, e.g. feed-forward neural networks (FNN), recurrent neural networks (RNN), and long short-term memory (LSTM) models are seen as potential promising tools to model the complex, nonlinear relationship between climate indicators and yield in crops (Lionel et al., 2025; Yakubu et al., 2024). Nonetheless, only a few studies have been done in Nigeria, which have combined CO₂, temperature and rainfall on the same basis using such integrative techniques. This research will address this gap by applying deep-learning models to understand how the three key indicators of climates can influence food security to enable farmers, policymakers, and stakeholders to develop food systems that are stable to climatic changes.

## **1.2 Problem Statement**

Nigerian rural life and GDP rely on the agricultural sector, which is receiving an impact of the changing weather patterns, particularly rain, heat, and CO₂ (Akinkuolie et al., 2025; Ezekwe et al., 2024). Such alterations reduce harvests, threaten food security and put pressure on farmers who depend on rain (Emegha et al., 2025).

Unpredictable drought, unpredictable rain, and floods destroy farm lands and hotter temperatures cause heat stress, increase water evaporation, and lead to pest problems (Gebeyehu et al., 2025). Although, increased CO₂ will increase photosynthesis, water and nutrient constraints most of the time overcome the increase. These climate changes cause food prices to go up, malnutrition and an unstable economy.

The relationship between CO₂, temperature and rain is complex and cannot be described using simple statistical analyses. Further deep learning models, such as Feedforward Neural Networks (FNN) or Long Short-Term Memory (LSTM) networks can give valuable insights that farmers and policymakers can use to create resilience and adaptability (Lionel et al., 2025).

## **1.3 Motivation**

The study is informed by the shocking and growing problem of food insecurity in Nigeria that requires evidence-based and scalable interventions. The current methodological paradigm aims to use advanced deep learning models to interpret complex data and come up with predictive models. This can be applied directly to the fact that there is a need to actionable insights in order to reduce supply chain disaffiliation, manage resources more efficiently, and increase agricultural resilience.

## **1.4 Aim and Objectives**

### **Aim**

The aim of this study is to assess the impact of climate change on food security using deep learning techniques.

### **Objectives**

* To design a deep learning model for predicting food security based on ainfall, temperature and CO₂ levels.

* ### To implement the deep learning model (FNN and LSTM) using python libraries like TensorFlow/Keras.

* ### To evaluate the performance of the model using accuracy, precision, recall and F1-score metrics.

## **1.5 Research Methodology**

The methodology of this research is organised into a step-by-step approach to assess the impact of changes in rain, temperature and CO₂ on food security in Nigeria with the help of deep learning:

1. ### **Data Collection**

   1. ### Find data about rain, temperature, CO₂ level and crop production from credible sources (e.g. agricultural databases, climate data).

2. ### **Data Preprocessing:**

   1. ### Clean and prepare the datasets, eliminate inconsistencies and gaps. 

   2. ### Separate the data into training and testing divisions to develop and evaluate the model.

3. ### **Model Development** 

   1. ### Develop deep-learning algorithms, e.g., FNN and LSTM, to forecast the level of agricultural production and food security.

   2. ### Evaluate the performance based on accuracy, precision, recall and F1-score.

4. ### **System Integration** 

   1. ### Deploy the trained model with a backend. 

   2. ### Tools: Python, TensorFlow/Keras, Pandas, NumPy and backend systems such as Django/Flask.

## **1.6 Definition of Terms**

1. ## **Climate Change:** The term denotes any changes in weather patterns in the long run, especially in terms of temperature, rainfall, and the composition of the atmosphere, mostly resulting due to human impact (greenhouse emissions and deforestation) (Ezekwe et al., 2024).

2. ## **Food Security:** This term is used to describe the availability of all individuals at all times with adequate, healthy, safe, and affordable food to live healthily (Emegha et al., 2025).

3. ## **Rainfall Variability:** The term signifies the changes in the quantity, intensity, and timing of precipitation and it also significantly influences the growth and yield of crops (Udeh et al., 2024).

4. ## **Temperature Change:** The ambient temperature fluctuates with time and may affect crops, evapotranspiration and pest/disease proliferation (Gebeyehu et al., 2025).

5. ## **CO₂ Concentration:** This is the amount of carbon dioxide in the air and this may have an impact on photosynthesis and crop yield and climatic conditions (Gebeyehu et al., 2025).

6. ## **Deep Learning:** This is a branch of artificial intelligence that entails the use of neural networks with layers, which can be trained to understand complex, non-linear correlations between large volumes of data (Lionel et al., 2025).

# **CHAPTER 2: LITERATURE REVIEW**

## **2.1 Introduction**

One of the most significant fields of 21st century research is the interplay of climate change, the security of food, and artificial intelligence. Increasing global temperatures and changing weather patterns require the agriculture sector to have new challenges on how to maintain the food systems strong enough to support a rising population [(Gebeyehu et al., 2025\)](https://www.zotero.org/google-docs/?broken=x4Fvm7). Nigeria is also particularly prone to climate variability that endangers food availability, access, and stability due to the prevalence of rain-fed farms, as well as, smallholder farmers [(Akinkuolie et al., 2025; Effiong, 2024\)](https://www.zotero.org/google-docs/?broken=7ku7RI).

## **2.2 Climate Variables and Crop Yield Relationships** 

The relationship between climatic variables and crop yield is investigated and graphically represented on graphs that provide quantitative information concerning the relationship between crop yield and climatic variables. The dynamics of climate aspects such as the rain and heat on crop yields are complex. The impacts are not necessarily linear and minute additions can result into huge disparities abruptly. It is the interaction among these factors that allows us to create more useful forecasting instruments and adaptations.

### **2.2.1 Rainfall** 

In farms that depend on rainfall, it is the amount and pattern of rainfall that determines the extent of crop growth in the farm. That, as well, is very important when it rains and how hard it rains.  
The seeds require sufficient water to begin to grow. When it rains late or during dry season in the initial stages the seeds are not going to sprout or die hence this prevents the crop to produce to its full potential despite the quality of the rest of the harvest [(Udeh et al., 2024\)](https://www.zotero.org/google-docs/?broken=yZWYDF). In the case of crops that require water at a particular time in their lifestyles e.g. maize, the trend of rainfall is important than the amount of rainfall.  
Various crops are not sensitive to the season they face drought. Maize itself is prone to this weakness during the flowering phase, but at the beginning of the grainfilling period. The drought at this period may lead to the death of the pollen and even aborted grain [(Ezekwe et al., 2024\)](https://www.zotero.org/google-docs/?broken=W7oTzo). 

**Consequences of variability: Flooding vs. Drought**  
The variability of rainfall may be in the form of shocks or drought, each of which is fraught with dissimilar issues:  
**Flooding:**  
Floods may lead to soil erosion, such as waterlogging as well as destruction of crops [(Udeh et al., 2024\)](https://www.zotero.org/google-docs/?broken=m2ykHV). Rainfall can heavily erode the topsoil, leaving the soils without nutrients and organic materials in case of heavy rains on hillsides. Withstanding constant saturation of soil roots cannot find oxygen and may die, and bad soil organisms may be cultivated.  
Rainfall may conceal the nutrients like nitrogen excessively further into the soils, leaving out the plant roots. This negatively affects the performance of the plants used in the present season and damages the soil of the upcoming seasons.  
Sloppy soils promote fungus and bacteria. Such diseases like late blight in cassava, downy mildew in maize and root rots are aggravated by keeping the soil moist.The flooding may prevent farmers to plant in time, shorten a crop growing period and lead to the premature or poor apple maturity. The heavy rain in the harvest period slows down the efforts leading to losses and poor quality of the grains.  
**Drought**  
In the absence of sufficient water, plants grow at a reduced rate or cease to grow thus causing smaller leaves, weaker roots and general retarded growth. Prolonged drought causes a decrease in size of plants and an ultimate decrease in final yield. Drought that occurs when crops are making flowers and seeds may disrupt the process of reproduction [(Udeh et al., 2024\)](https://www.zotero.org/google-docs/?broken=6Hl4mI). In the case of maize, dry spells at flowering may dry the pollen and silk halting the process of fertilization and resulting in the loss of the number of grains. Even the dry season can reduce the yield significantly.  
Drought causes the leaves to age prematurely and cease to work thus the time available to fill the grain reduces and decreases the weight of the grain. This is more so where drought occurs during the stage of filling the grains.  
When water becomes unavailable, the plants become more vulnerable to the pests. The pests also do not like wet weather as some of them like hot weather hence the combined stress exposes the crop to further agony.  
**Patterns across Time and Cumulative Effects**  
The timing of rain in regards to the growth process of the crop is a big issue. Planting late may result due to drought in the early of the growing season, mid-season drought damages growth and flowering of the plant and drought in the late season truncates grain filling. The cumulative effects of rainfall also exist: in case, the soil remains dry over a couple of weeks, the plant is not likely to be completely repaired and is exposed to more damage further [(Udeh et al., 2024\)](https://www.zotero.org/google-docs/?broken=t4MXW9).  
Climate change does not only alter the amount of rainfall but it also alters the time of rain. There are numerous locations where there are additional swings throughout the season, dry seasons and wet ones. This is difficult to farms since it combines the effects of drought with those of soil erosion and loss of nutrients and the downpour is too sporadic to assist.

### **2.2.2 Temperature**

Temperature practically directs what the plants do; the activity of the enzymes, how they produce food, how it breathes, and when it grows. The average temperature and highs or lows of the temperatures influence the production output of crops.  
**Effect on physiologic Processes**

1. Photosynthesis: this is the process through which plants convert light to food. It also does well under specific temperature which varies depending on the crop. It is known that maize (a C4 crop) does better at higher temperature of 30-35°C whereas cassava (a C3 crop) does better at 25-30o C. In the majority of the plants, the rate of photosynthesis increases rapidly with temperatures up to a point of optimum, and then decreases when the enzyme is destroyed at extremely high temperatures [(Bernacchi et al., 2023\)](https://www.zotero.org/google-docs/?broken=6xjXwb).  
2. Respiration: Plants also breathe as well. During hot nights, breathing rates go up significantly but photosynthesis does not and hence the plant becomes deficient in carbon. It will reduce the amount of sugar in the filling of the grain and look at us with fewer yields [(Effiong, 2024\)](https://www.zotero.org/google-docs/?broken=JxsRE0).  
3. Evapotranspiration: Hot weather causes plants to consume a lot of water. Although rain supply is sufficient, further loss of water may lead to the drying of the soil and drought state, in particular, in the conditions of dry air and powerful winds (Effiong, 2024).  
4. Developmental Timing: Temperature accumulates in units of heat that causes growth of the plants. It is developed by warm weather. It can prevent late-season issues by the crops but also it may be used to cut down on important phases such as grain filling, which damages harvests. There are also chances of mismatches when fast growth results in flowers which are not good to be pollinated.

**Heat Stress: Sub-Saharan Presentation in Northern Nigeria**  
Northern Nigeria is also experiencing a lot of hot weather and this fact is important to the farms where there is cool climate. Heat lowers the moisture content of soil and makes the crops well beyond the usual temperature ranges (Effiong, 2024). Heat stress manifests itself in a number of ways:

1. **Heat Damages Enzymes and membranes:** Extreme temperatures that exceed the critical temperature range (typically 35-40°C) can destroy both proteins and cell membrane destroying cells. Even with latent drop in temperature, a brief exposure of high heat is still likely to cause enduring issues.  
2. **Pollen Unproductiveness:** Since pollen is drying up during flowering by the hot weather the plant is unable to grain a grain. It is a great danger to such cereals as maize and sorghum when an adjacency of several hot hours may significantly decrease the amount of grains (Ezekwe et al., 2024).  
3. **Increased Water Stress:** Heat increases the loss and demand of water. This added stress is particularly deleterious in the Sahel and Sudan savanna of Nigeria in the north where water is in short supply.  
4. **Soil Degradation:** Surface heat is also able to kill beneficial soil life, reduce the speed of decomposition of organic matter, and disorient the mobility of nutrients, damaging the long-term health of the soil.

**Rainfalls and Temperature**  
There are complicated associations between temperature and water. Hot climate also intensifies the demand of water hence drought is also more threatening. Evaporative cooling is able to reduce heat stress if there is a sufficient amount of water. Therefore, the impact of high temperatures has a great deal to do with the precipitation pattern, which makes the results of the process rather hard to predict. The improved models should demonstrate the interaction between temperature and water (Effiong, 2024).  
Climate changes in Nigeria are projected to experience further warming and more so in the northern regions. These temperature increases will reduce the productivity of the farms without good adaptation, e.g. heat-tolerant varieties, altered planting dates and extra irrigation.

### **2.2.3 Carbon Dioxide (CO₂ )**

The levels of atmospheric CO₂ have been continuously increasing over the decades and nearly the same amount can be found anywhere on the planet. This is a long-term tendency both that opens opportunities (CO₂ fertilization) and causes problems (negative trade-offs) in farming.  
**The Fertilization Effect: Potential C3 Crops Biomass Growth**  
Photosynthesis uses CO₂ as its fuel. CO₂ increased could help to enhance photosynthesis particularly in C3 plants that are cassava, yam, wheat and rice. C3 plants typically become CO₂-limited, and additional CO₂ accelerates the rate of carbon uptake and eliminates wastful photorespiration, which may enhance growth (Wang et al., 2024; Dakora et al., 2025). Recent thus indicate that increased CO₂ enhances net photosynthesis, water-use efficiency of wheat, yielding increasing returns of 8 \-33% based on the wheat varieties and the conditions (Liu et al., 2025). Researchers discover that C3 crops respond better to additional CO₂, whereas C4 crops already have high capabilities in capturing CO₂ and receive less direct benefits on their fertility (Moore et al., 2021).  
Where the CO₂ effective response is the maximum potential CO₂ fertilization effect, and the coefficient of means equals 1.026, known as the regression coefficient:

1. **Cassava and Yam (C3 Crops):** These are able to gain out of increased CO₂. A high quantity of CO₂ may augment the tuber size and root, which will counter the effects of climate. This assistance is only effective when there is sufficient amount of water and nutrients. Under non-drought situations, increased CO₂ intensity enhances leaf photosynthesis, carbon accumulation, as well as an increase in water use efficiency (Zhang, 2024).  
2. **Maize (C4 Crop):** Maize has also another type of photosynthesis, which is already very efficient with the use of CO₂. It does not directly benefit much because of an increased CO₂ but may indirectly benefit by an improved use of water since the stomata slightly close during increased CO₂ concentrations (Moore et al., 2021).

Such non-linear relationships are too complicated to be effectively estimated using classical methods in which the underlying biology is poorly comprehended, and deep learning is convenient in this case to predict how crops will respond to CO₂ (Lionel et al., 2025).

## **2.3 Food Security and Climate Change.**  

### **2.3.1 Climate Change**

Higher and more frequent heat waves, droughts, floods, and storms are being caused as a result of climate change [(Calvin et al., 2023\)](https://www.zotero.org/google-docs/?broken=NOdMJc). These extremities damage farms and the structures that they operate in, and they also disrupted food systems by damaging the supply chains and prices.    
The increase in atmospheric levels of carbon dioxide has increased approximately twice before industry to more than 420 parts per million (ppm) due primarily to the production of fossil fuels and land use alteration [(Calvin et al., 2023\)](https://www.zotero.org/google-docs/?broken=1zc57N). Although increased CO₂ does have the benefit of enhancing the growth of certain crops due to increased photosynthesis, increased temperature and reduction of nutrients usually negate this benefit.    
The Nigerian agriculture is experiencing certain climatic challenges:  

* Rainfall Unpredictability: The variations involve the increased number of droughts and increased rains which disorganize the planting plans and potent crops [(Udeh et al., 2024\)](https://www.zotero.org/google-docs/?broken=yUk1ar). Knowledge that was used traditionally in planting and harvesting based on weather in the past is becoming unreliable.    
* Climate Shocks: When the temperatures are extreme either high or low they cause damage to the crop production and quality [(Effiong, 2024\)](https://www.zotero.org/google-docs/?broken=A1Ggqr). Northern states are experiencing heat waves which go beyond the limit of most staple crops and are reducing the harvest, putting water usage into the spotlight.    
* Flooding and Erosion Southern Nigeria has had disastrous floods in the past years that have swept down the fields, blowing the top soil and contaminating the water sources [(Akinkuolie et al., 2025\)](https://www.zotero.org/google-docs/?broken=5gFtjo). All these events reduce the agricultural outputs and impair the soil productivity in the long run.  

Social and economic factors add to the susceptibility of the Nigerian agriculture to climate change. Four small farmers out of five are financially vulnerable to climate risks, 80% of them are susceptible to climate change, and 37% are severely constrained with regard to finances [(Akinkuolie et al., 2025\)](https://www.zotero.org/google-docs/?broken=bva7ks).  

### **2.3.2 Food Security in Nigeria**  

According to the definition of Food and Agriculture Organization (FAO), food security refers to the fact that all individuals have physical, social and economic access to adequate, safe and nutritious food to satisfy their dietary needs and preferences in an active and healthy generation [(The State of Food Security and Nutrition in the World 2023, 2023\)](https://www.zotero.org/google-docs/?broken=zvVeph).  

* **Availability**: Availability refers to the availability or the presence of the physical food available, locally produced, imported, is in food stocks and the availability of food aid [(Gebeyehu et al., 2025\)](https://www.zotero.org/google-docs/?broken=znDk0s). Climate change in Nigeria reduces production, supply chains, and increases the disparity between the poor and the rich [(Ezekwe et al., 2024\)](https://www.zotero.org/google-docs/?broken=CUSHcv). The fact that the country grows mostly on the domestic staple crops means that any loss of the yields will pose a direct threat to the availability of food.    
* **Accessibility**: Accessibility refers to the ability of people to access available food by purchasing, trading, or getting assistance. Low income is associated with food insecurity, whereby individuals lack access to adequate and sufficient food and instead experience diets that are low in nutrients, which in turn causes such diseases like obesity, heart diseases, high blood pressure, diabetes and other chronic diseases [(Akinkuolie et al., 2025\)](https://www.zotero.org/google-docs/?broken=r1wo2F). When climatic occurrences lower the agricultural output, the food prices will be high and the people living below the poverty mark will struggle to access food.    
* **Stability**: To have food stability, availability, accessibility and use should remain adequate throughout the seasons and years. Climate instability and extreme weather systems bring about instability because they lead to sudden upsets to the production system and food system [(Gebeyehu et al., 2025\)](https://www.zotero.org/google-docs/?broken=T2Zg3q).  

**Nutritional Effect**  
Climate change is causing food security as well as reducing the nutritional value of staple foods besides reducing their quantity. To a large extent, Nigeria relies on yam, cassava and maize that have been widely used to give the populace many calories [(Ezekwe et al., 2024\)](https://www.zotero.org/google-docs/?broken=W7CSYR).  

* **Yam:** Yams are very susceptible to climatic extremists and drought is the primary stressor. The middle belt and the southeastern states are the primary sources of production but the disruption of the traditional farming has been caused by altered rain patterns. High temperatures used in the growth of tubers reduce yield and quality of tubers [(Ezekwe et al., 2024\)](https://www.zotero.org/google-docs/?broken=L0MmnJ).    
* **Cassava:** Cassava is the most fragile crop to drought, but it is referred to as being drought-tolerant. The long spells of dry conditions lower mass of roots and enhance cyanogenic compounds and this affects the yield and safety. Cassava is one such fallback crop since it can be planted at any time and can be subjected to drought and therefore when the climate stresses occur, cassava plays a vital role [(Ezekwe et al., 2024\)](https://www.zotero.org/google-docs/?broken=wPdGB5).    
* **Maize:** Maize is the worst hit food crop particularly to drought and heat. It is more sensitive to water stress during flowering and filling of grains as well as to minimal dry spells which lead to massive loss of yield. There are also increased temperatures that promote the accumulation of mycotoxins in maize due to high temperatures, which is a risk to food safety [(Ezekwe et al., 2024\)](https://www.zotero.org/google-docs/?broken=bAst7F).

Climate change does not only reduce the crop yields it also alters the food we eat in terms of its nutritional value. An increased concentration of CO₂ in the air can cause plants to get larger but normally decreases the quantity of protein and vitamins present in the plants.  
**Social aspects of Food Insecurity.**  
Not everybody is affected by climate change. The poorer or more susceptible groups are faced with more people experiencing floods and droughts. They are particularly impacted on women, children and small farmers.

* **Women:** Women are most of the time expected to tend to and prepare food. In cases where there are limited resources, they have additional work to attend to [(Akinkuolie et al., 2025\)](https://www.zotero.org/google-docs/?broken=NuOtym). The women farmers also receive inferior land, loans, seeds, and even the government assistance than the men and due to this, they find it difficult to adapt.  
* **Children:** In the event of lack of food, children may fall ill and fail to grow well. Poverty keeps the families poor because malnutrition in early years may damage brain and body development. Every time the food price increases due to the climatic issues, the poor families reduce their food consumption, and children are the most affected by the blow [(Akinkuolie et al., 2025\)](https://www.zotero.org/google-docs/?broken=dlU6o8).  
* **Elder:** The older people are even more susceptible since they are incapable of working in the fields, they might need to eat more, and they struggle to adapt to the new farming practices [(Gebeyehu et al., 2025\)](https://www.zotero.org/google-docs/?broken=FfOm6U).

The correlation of climate variables and crop yield is examined, presenting graphs that supply quantitative data about the relationship between crop yield and climatic conditions.

## **2.4 Artificial Intelligence**

### **2.4.1 Artificial Intelligence (AI)**

Artificial Intelligence refers to computer systems, which are able to perform tasks traditionally performed by the human mind such as the recognition of patterns, making predictions, decision making and increased efficiency [(Yakubu et al., 2024\)](https://www.zotero.org/google-docs/?broken=1BiM90). In agriculture, AI is applicable to numerous approaches that assist farmers in enhancing the operations of the farms, predicting crops outcomes, monitoring the environment and managing resources more intelligently.    
The AI in the field of agriculture has no longer stayed at the level of a rule-driven system; nowadays, it is a smart algorithm capable of learning vast quantities of data. Today, AI can: make assumptions on the volume of crop to produce in relation to previous data and prevailing circumstances, picture recognise diseases of the spot plants, send out word to farmers on when and how much to water or to fertilize, warn about pest outbreaks, evaluate the climate change risks [(Lionel et al., 2025; Yakubu et al., 2024\)](https://www.zotero.org/google-docs/?broken=bHaFJJ).    
The tools are particularly helpful where such services as extensions are not widely available e.g. in Nigeria where farmers require convenient decision support.  
 **The Application of AI in Agriculture**

* Predictive Analytics: Artificial intelligence can predict the performance of crops across the various climate conditions, and therefore farmers can take action prior to these issues occurring (Lionel et al., 2025).    
* Early Warning Systems: AI can notify farmers about a threat, e.g. drought, pests, or disease by examining the data collected during weather, soil health, and crop health (Yakubu et al., 2024).    
*  Precision Agriculture: AI allows farmers to apply water, fertilizer, pesticide at the point of need and enhance reduction of waste and pollution, as well as maintain high yields [(Hoque & Padhiary, 2024\)](https://www.zotero.org/google-docs/?broken=Fg9ZCC). Precision irrigation and fertilization increased yields by 5-15-percent and reduced water, and fertilizer 25-40-percent, and 30-40-percent respectively [(Hoque & Padhiary, 2024\)](https://www.zotero.org/google-docs/?broken=00f8zs).    
* Climate-Smart Decision Support: AI has the potential to propose the most appropriate dates of planting, change of crops and activities that correspond with local climatic changes, transforming technical information into understandable instructions to be taken by farmers [(Ikendi et al., 2024\)](https://www.zotero.org/google-docs/?broken=RXMUMv).

### **2.4.2 Machine Learning (ML)**  

Machine Learning refers to the subset of AI that analyzes programs that are learned with data rather than written with set rules (Lionel et al., 2025). It is through practice that machines can identify a pattern in data and make predictions of an unseen event.   
ML normally uses three parts:    
1\. A representation of the relationship between inputs and outputs,    
2\. The modified data by the training data model.    
3\. A performance metric that determines the quality of the predictions (error) (Lionel et al., 2025). The routine matures with a lot of input-output pairs (supervised learning) that it may identify the best patterns that describe the outputs.  
ML is trained in agriculture to recognize crop output basing understanding of weather, soil, and agricultural inputs on outputs or quality. After the training the model is able to provide a prediction in a new place or the new time.  
**Widespread Machine learning algorithms in Agriculture**  

* **Decision Trees:** Separate data by simple yes/no questions and in the form of a tree, one can see the way every step contributes to a prediction. They are simple to comprehend and have the capacity to utilize numerical or nominal data (Lionel et al., 2025).    
* **Random Forests:** Construct numerous decision trees and combine their outputs to provide superior and more to the point forecasts. They are effective in prediction of the yield and can be useful in ranking the climate or management factor that is the most important (Lionel et al., 2025).    
* **Support Vector Machines (SVM):** Optimal boundary line dividing various classes in data. They are useful in stress/ land use classification of plants on images and non-linear relationships can be addressed with special methods (Lionel et al., 2025).    
* **Gradient Boosting:** Build a sequence of models, with each of them addressing the errors of the prior ones. Even more powerful such models as XGBoost and lightGBM are capable of learning non-linear models and winning numerous farming prediction challenges (Lionel et al., 2025).

### **2.4.3 Deep Learning (DL)**

Deep learning is a more modern type of machine learning that addresses most of the problems encountered in the older ones. DL employs artificial neural networks with multiple layers (hence "deep") to learn hierarchical data representations automatically, without requiring manual feature engineering (Lionel et al., 2025; Yakubu et al., 2024).  
**Climate-Relevance to Agriculture Applications.**  
Deep learning applies particularly well to climate and food security research due to a number of reasons:

* **Satellite Image Analysis:** The analysis of satellite images is required to monitor the state of crops, estimate the yields of crops, and identify environmental stresses in large regions. Convolutional neural networks (CNNs) have transformed remote sensing in that multispectral imagery and time-lapse phenomena of satellites are automatically extracted, classified, and yielded by a convolutional neural network to extract vegetation indices, crop, and yield (Yakubu et al., 2024).  
* **Complex Non-linear Climate Interactions:** Climate variables contribute to crop productivity are full of non-linearities, thresholds beyond which stress increases rapidly, multiple stress interactions where a combination of stresses combining in an interaction or time-lag effects in which current outcomes depend upon past weather. This complexity can be modeled with the use of deep learning, which makes it the best product in such applications (Lionel et al., 2025).  
* **Multi-type Data Combination:** Climate-agriculture studies use a lot of data: quantitative weather measurements, nominal soil classes, images, and texts. Deep learning models are capable of integrating all these sources of data in one prediction system (Yakubu et al., 2024).  
* **Temporal Prediction:** Agriculture is sensitive to climatic conditions on long spatial scales weeks to months to grow crops, year to year to a perennial crop, decades to climate change. Such time-series data are learned by long-short-term memory (LSTM) networks and other recurrent networks, which are sensitive to the ways of how previous events can cause future and present states (Lionel et al., 2025).

## **2.5 Deep Learning Architectures for Climate-Agriculture Models** 

### **2.5.1 Feedforward Neural Networks (FNN)**

The simplest type of deep-learning architecture is the feedforward neural network which is used as foundation in more advanced neural networks. In an FNN, the information moves in a single direction, it goes through the input layer, through the hidden layers, and to the output one with no loops and feedback [(Lamba et al., 2021\)](https://www.zotero.org/google-docs/?broken=bfaSZw).  
**Structure of FNN**  
An FNN is made up of three kinds of layers:

1. **Input Layer:** It is the layer of raw features of the problem. In farming it could involve elements of climate (temperature, rainfall, humidity), soil characteristics (pH, organic matter, texture), and agricultural practices (planting date, fertilizer applied, irrigation) (Sivanandhini and Prakash, 2020).  
2. **Hidden Layers:** This is one or more layer in between which the actual learning occurs. Every hidden layer consists of numerous neurons, which do both (a) weighted sums of the previous layer and (b) provide a non-linear function to add complexity to the model (Lamba et al., 2021).  
3. **Output Layer:** The final prediction provided is by this layer. On tasks of type regression such as the prediction of yield, typically only one neuron is required which will provide a continuous number as its output. In the case of classification (e.g. labelling the levels of stress in crops) it contains several neurons, each corresponding to one class (Ankitha et al., 2024).

![][image2]

#### **FIG 2.1 FNN STRUCTURE**

The name feedforward implies that data flows in one direction in the network. This occurs during training the data forwards to take a prediction, error signals reverse to refine the weights, yet when the model is utilized to make a forecast it never reverses (Lamba et al., 2021).  
**Agricultural Prediction Applications**

1. **Wheat Yield Prediction:** FNNs yield have been used to predict the yields of wheat. The models predict the yields accurately by relying on the historical data of the weather, soil, and farming practices, which can be used by farmers and policymakers (Lamba et al., 2021).  
2. **Multi-Crop:** According to the research on comparing the use of FNNs and RNNs in predicting crop yields, FNNs are most effective when using static features or aggregated features. They can use a combination of climate and soil in order to forecast the yields of multiple crops (Sivanandhini & Prakash, 2020).  
3. **Crop Selection Support:** This type of FNN based recommendation systems takes numerous soil and climate parameters into consideration and proposes an appropriate crop. They consider nutrient content, pH, temperature, humidity, and rainfalls to provide farmers with evidence-based crop decisions that will help to increase production and optimize the utilization of resources (Ankitha et al., 2024).

**2.5.2 Long Short Term Memory (LSTM)**   
Long Short Term Memory (LSTM) networks are a special case of recurrent neural network (RNN) that was developed to address the timing issue of FNNs through the incorporation of memory cells that are trained on sequences of data (Lionel et al., 2025).  
**Structure of LSTM**  
The general concept of LSTMs is the so-called memory cells, which are special units capable of lasting a long time, along with gates that regulate the inputs and outputs [(Lan et al., 2021\)](https://www.zotero.org/google-docs/?broken=GRnUJO). This allows LSTMs to retain critical historical data and discard the irrelevant one, which is a major issue with simple RNNs because it can happen that gradients disappear or explode with many runs of their execution.  
The LSTM unit consists of four major components:

* Cell State: A sequence of memory which traverses the entire sequence enabling information to remain more or less comparable over a considerable duration. Imagine that it is a conveyor belt which transfers any appropriate data into the future [(Lan et al., 2021\)](https://www.zotero.org/google-docs/?broken=tb1kAQ).   
* Forget Gate: Determines what to drop in the information of the previous state in the cell. It takes the present input and the past hidden state and returns a value between 0 (forget all) and 1(keep all) of each component of the cell state [(Lan et al., 2021\)](https://www.zotero.org/google-docs/?broken=eXrQSK).  
* Input Gate: Determines what new information to have in the cell state. It consists of two components, a sigmoid that selects values to revise, and a tanh one that produces candidature values to insert [(Lan et al., 2021\)](https://www.zotero.org/google-docs/?broken=VRulXb).  
* Output Gate: Determines the hidden output value in the current step. It uses the input and past hidden state to filter the cell state then gives a result of the LSTM at that time point [(Lan et al., 2021\)](https://www.zotero.org/google-docs/?broken=goy2xk).  
  ![][image3]

#### **FIG 2.2 LSTM STRUCTURE**

This control system allows LSTMs to store information of many time steps before and is therefore applicable to the temporal tasks in which long time dependencies are of interest (Lionel et al., 2025).  
**Applicability to Climate Agriculture**  
LSTMs can be more useful in fields of farming involving time-related data especially so:

* **Rainfall Pattern Forecasting:** Rain patterns consist of intricate time structures- they occur every single season, vary over the years and the wet or dry seasons appear to cluster together. LSTMs are able to generalize these patterns on previous data and project the future rain which will provide useful information to the farmer such as planting and watering schedules (Lionel et al., 2025).  
* **Phenological Stage Modeling:** The growth of crops follows definite stages of growth-germination, the growth of vegetable, the flowering, grain filling, maturation. The situation at each stage has a different influence on the end yield. The LSTMs will be able to monitor the level of stress the crop was subjected to in the entire growing season, e.g., the effect of early drought on late stages (Lionel et al., 2025).  
* **Detection of Climate Anomaly:** With the knowledge of normal climatic patterns, LSTMs are able to identify anomaly conditions such as off-season temperatures, the unexpected raindrop, or a strange weather pattern that may cause harm to crops (Lionel et al., 2025).

### **2.5.3 Hybrid Deep Learning Models**

Hybrid deep learning models are the recent form of combining multiple types of architectures, typically the FNNs and LSTMs, in order to leverage the advantages of each alongside minimize the disadvantages.  
**Strengths of FNN and LSTM**  
The notion of hybrids is easy to comprehend: the agriculture forecasting usually requires time-specific information (such as climate series or farm activities across time) and non-temporal/spatial information (soil, elevation, features of the farm) (Lionel et al., 2025). Individual versions of FNNs and LSTMs are not the best for both:

* FNNs: Good in the non-linear association of features but is incapable of handling sequences across time.  
* LSTMs: Can naturally learn sequences, however, adding context that is static and time-invariant cannot be learned easily.

All this can be dealt with using hybrid models in which special parts are used to process each piece of data, which are then combined to produce the final prediction. An example hybrid FNN-LSTM of crop yield may have the following manner of operation:

* Branch of LSTM: The daily data of climate factors (temperature, rain, and humidity) during the growing season are inserted and identified as time patterns and links (Lionel et al., 2025).  
* Branch of FNN: Renowned use of fixed data (soil classification, altitude, long-term climatic regime, type of crop) to obtain background data which determine the potential yield [(Lamba et al., 2021\)](https://www.zotero.org/google-docs/?broken=OE9ZiQ).  
* Combined Layer: This is an extension of the LSTM, but it takes an additional step of combining both the LSTM final hidden state with the FNN output, typically via fusing the two, and then feeds the combined data through further layers in order to generate the final yield prediction (Lionel et al., 2025).

With this design, the model can examine what has happened with           time (in the LSTM) and what the underlying conditions are (in the FNN), which make the model give more accurate and context uptonic forecasts.  
**Reason for Hybrid**

* No Information Loss in LSTM Models: A pure LSTM would require that we feed it with any static information and this is both wasteful and disorganizing to the time sequences, which the time clustering should acquire (Lionel et al., 2025).  
* Temporal Blindness in FNN Models: A pure FNN requires engineers to design time features (means, extremes, trends) explicitly such as extremes, the task of creating this kind of time features is difficult, and one may fail to capture complex time structure (lamba et al., 2021).  
* Inefficient Resource Utilization: There is a type of data that a given architecture is constructed. It causes the model to struggle with the task it is not designed to and is an example of making an LSTM process static data or an FNN process long series.

These issues are addressed by the hybrid models which ensure each aspect performs what they are most efficient in and then pool their knowledge together. This is comparable to the process applied by a farmer in solving a problem: in thinking about the past weather pattern (time reasoning) and the inherent nature of the field by way of soil and other predetermined influences (context knowledge). The same is done with the hybrid models which integrate temporal and contextual processing.  
**2.6 Review of Related Works**  
More and more agricultural problems, in particular, crop yield prediction, are being addressed using artificial intelligence, machine learning, and deep learning. This part will examine prominent research, the comparison of work and research across the world and Nigeria.

| Author (Year) | Title | Methodology/Methods | Findings/Results | Limitations/Gaps |
| :---- | :---- | :---- | :---- | :---- |
| Lionel et al. (2025) | A comparative study of machine learning models in predicting crop yield | They performed the comparison on Decision Trees, Random Forest, SVM, Gradient boosting, and Neural Networks using the weather data of 8 crops. | Random Forest provided the highest R² (0.875 when using potatoes, 0.817 when using maize); XGBoost provided minimum error (0.07) when using cotton | No deep learning, time-series data, remote sensing, or hybrid models; only used classic ML.  |
| Gebeyehu et al. (2025) | Effects of climate change on food security in Africa: Meta-analysis | They carried out a meta-analysis of the African research on the impacts of climate change on crop yields    | There is a reduction in yield in Africa due to climate change, depending on the type of stress, region and crop | They have not proposed new ways or data to be used in future; they have simply reviewed earlier studies that employed other methods |
| Akinkuolie et al. (2025) | Resilience to climate-induced food insecurity in Nigeria: A systematic review of adaptation strategies | They analyzed how the Nigerian farmers cope with floods and drought | Conservation Agriculture, management of Soil fertility 80% of smallholders are financially threatened; knowledge is the greatest barrier | No new data, predictive models; no AI tools; there is a gap between the identification of solutions and their practical implementation |
| Yakubu et al. (2024) | Artificial Intelligence Applications in Sustainable Agriculture in Nigeria: A Comprehensive Review | They have examined AI usage within the agricultural sector of Nigeria, such as the yield control, disease detection and precision farming | AI has the potential to change the state of agriculture in Nigeria but encounters various obstacles: inadequate digital infrastructure, training, and high associated costs | No models were created; no description of algorithms or benchmarks; no discussion of cassava and yam; no performance comparison |
| Wang et al. (2024) | A Temporal-Geospatial Deep Learning Framework for Crop Yield Prediction | They constructed CNN-GAT-LSTM model, using spatial ( CNN ), geographic ( GAT ) and temporal ( LSTM ) features on 1,115 counties of the US between 1980 and 2018 | The model outperforms the previous ones by RMSE, R² and correlation; it captures spatial and temporal change in yields | Only soybean in the USA; no climate \-change predictions; untested in Africa or Nigeria; GAT requires geographic data which is not always available |
| Lu et al. (2024) | Deep Learning for Multi-Source Data-Driven Crop Yield Prediction in Northeast China | They employed CNN-LSTM-Attention model based on vegetation indices, environmental data and photosynthesis data within the years 2014-2020 | Precision greater than 90%. maize, rice, soybean; optimum in July-August; the hybrid is better than a bare CNN or LSTM | Northeast of china only; No CO₂ effects; intensive computing requirements can be a bottleneck on low resource device |
| Joshi et al. (2024) | An explainable Bi-LSTM model for winter wheat yield prediction | They trained LSTM, 1D-CNN, and Bi-LSTM models based on EVI, Integrated Gradients and SHAP explanations based on temperature and precipitation | Bi-LSTM displayed the highest R² (0.88); SHAP indicated that EVI, temperature, and precipitation were the most significant; explanation aided interpretation | Winter wheat only; small areas; no CO₂ or socioeconomic; not done on actual crops or Africa |
| Jabed et al. (2024) | Crop yield prediction in agriculture: A comprehensive review of machine learning and deep learning approaches | They overviewed AI methods of crop yield forecasting, including CNN, LSTM, DNN, and hybrid | CNN, LSTM and DNN have the best potentials; vegetation indices and environmental data are vital; increasing the usage of multi-data sources is arising | No actual models, only a review; little to be said about explainability analysis; African application gaps |
| Ezekwe et al. (2024) | Climate Change and Food Security in Nigeria: Implications for Staple Crop Production | They evaluated the impact of climate change on maize, cassava and yam in Nigeria | 70% of food production is crops, maize is the most susceptible to drought and heat with cassava and yam coming in second and third | Descriptive only; no AI/ML or predictive modelling;  |
| Effiong (2024) | Variability of climate parameters and food crop yields in Nigeria: A statistical analysis (2010-2023) | He statistically analyzed (correlation, regression) the impact of temperature and precipitation on the yields of crops between 2010 and 2023 | Good correlations between changes in temperature/rainfall and declining staple yields | No more than basic statistics, no ML or DL models, no CO₂, no forecasting and causal analysis |
| Kalmani et al. (2023) | Crop yield prediction using deep learning algorithm based on CNN-LSTM with Attention Layer and Skip Connection | They experimented in a multi-head attention hybrid 1D CNN-LM skip connections tested on wheat and rice in India | 98% accuracy, lowest RMSE (0.13) and MAE (0.14); it worked better than SVM, Decision Tree, and Random Forest | Only wheat, rice; requires huge data; can be modeled very complexly; there are no explanation and climate change experiments; it is not applied to African staples |
| Lamba et al. (2021) | Wheat yield prediction using feedforward neural networks | They applied FNN models that employed temperature, rainfall, and soil moisture among other climate parameters on multi-year data | The models reflected the climate-yield correlations and generated precise predictions that can facilitate the decision of farmers | Only wheat; does not do temporal modeling (manual feature engineering); does not compare LSTMs; only a small area; does not have interpretability |
| Sivanandhini and Prakash (2020) | Crop yield prediction analysis using feed forward and recurrent neural network | They compared both FNN and RNN with the data of soil, weather and crop management | RNN and FNN performed well with fixed data and sequence data respectively; both provided good predictions | No hybrid FNN-RNN models; poor spatial-temporal integration; no attention; no CO₂; inadequate regional context |
| Khaki and Wang (2020) | A CNN-RNN Framework for Crop Yield Prediction | They used Weather- CNN/Soil-CNN together with RNN-LSTM, using 2,247 US Midwest corn and soybean sites 2008-2016. | The hybrid model was superior to Lasso, plain neural nets and regression trees; it detected the annual yield variation caused by genetics and the environment | There are no CO₂ or genotype data; temperate crops are considered only; no explanation of the model; not experimented on data-deficient areas such as Africa. |

## **2.7 Conclusion**  

The chapter examined former researches on a cooperative relationship between climate change, food security, and deep learning (DL) in Nigerian agriculture. It demonstrated that climate variables, particularly rainfall, temperature, and CO₂ are highly significant to crops in farms particularly in Nigeria area where crops depend on rain. Another thing that was discussed during the review is that food security is weak and small farmers are highly susceptible to climatic changes. These are complex evolving interactions that are not well treated by traditional methods of making predictions.  
The following chapter was a discussion of how machine learning, and particularly the Deep Learning model, including Feedforward Neural Networks (FNN) and Long Short-Term Memory (LSTM), are being applied to achieve agricultural outcomes. It has indicated that LSTMs are more suitable in this regard since they can trace the occurrence of events with time, and also enable memory of previous impacts, which cannot otherwise be done with FNNs. It was also proposed in the review that mixed deep learning models that constitute the strengths of FNN and LSTM should be used. Such mixed models would provide more precise and flexible forecasts through the utilization of both constant and dynamic patterns of data.  
These new ideas notwithstanding, the review observed that a number of significant gaps exist. These are that they do not utilize a large number of other climate factors other than the major three, that mixed models are not utilized extensively in Nigeria, and that there is overall lack of the tools that could be easily comprehended and utilized by small farmers. Such lapses will assist in defining methods of the study. The research aims to contribute to the existing evidence on climate-smart agriculture and provide practical and real answers to enhance food security in Nigeria by exploring the interaction between rainfall, CO₂, and temperature.

## **References**

* Akinkuolie, T. A., Ogunbode, T. O., & Adekiya, A. O. (2025). Resilience to climate-induced food insecurity in Nigeria: A systematic review of the role of adaptation strategies in flood and drought mitigation. *Frontiers in Sustainable Food Systems*, *8*, 1490133\. https://doi.org/10.3389/fsufs.2024.1490133  
* Bernacchi, C. J., Ruiz-Vera, U. M., Siebers, M. H., DeLucia, N. J., & Ort, D. R. (2023). Short- and long-term warming events on photosynthetic physiology, growth, and yields of field grown crops. *Biochemical Journal*, *480*(13), 999–1014. https://doi.org/10.1042/BCJ20220433  
* Calvin, K., Dasgupta, D., Krinner, G., Mukherji, A., Thorne, P. W., Trisos, C., Romero, J., Aldunce, P., Barrett, K., Blanco, G., Cheung, W. W. L., Connors, S., Denton, F., Diongue-Niang, A., Dodman, D., Garschagen, M., Geden, O., Hayward, B., Jones, C., … Péan, C. (Eds.). (2023). *IPCC, 2023: Climate Change 2023: Synthesis Report. Contribution of Working Groups I, II and III to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change* \[Core Writing Team, H. Lee and J. Romero (eds.)\]. Intergovernmental Panel on Climate Change (IPCC). https://doi.org/10.59327/IPCC/AR6-9789291691647  
* Effiong, M. O. (2024). Variability of climate parameters and food crop yields in Nigeria: A statistical analysis (2010–2023). *Journal of Infrastructure, Policy and Development*, *8*(16), 9321\. https://doi.org/10.24294/jipd9321  
* Emegha, K. N., Bosah, P. C., Chugo Idigo, B., & Ofobuike, C. L. (2025). The effects of climate change on food security in Nigeria: A review. *International Journal of Research and Scientific Innovation*, *XII*(IV), 904–914. https://doi.org/10.51244/IJRSI.2025.12040076  
* Ezekwe, C. I., Humphrey, J. I. N., & Esther, A. (2024). Climate change and food security in Nigeria: Implications for staple crop production. *International Journal of Environment and Climate Change*, *14*(12), 486–495. https://doi.org/10.9734/ijecc/2024/v14i124639  
* FAO; IFAD; UNICEF; WFP; WHO. (2023). *The state of food security and nutrition in the world 2023*. https://doi.org/10.4060/cc3017en  
* Gebeyehu, L., Gemechu, A., & Bedemo, A. (2025). Effects of climate change on food security in Africa: Meta-analysis. *Journal of Agriculture and Food Research*, *23*, 102214\. https://doi.org/10.1016/j.jafr.2025.102214  
* Hoque, A., & Padhiary, M. (2024). Automation and AI in precision agriculture: Innovations for enhanced crop management and sustainability. *Asian Journal of Research in Computer Science*, *17*(10), 95–109. https://doi.org/10.9734/ajrcos/2024/v17i10512  
* Ikendi, S., Pinzón, N., Koundinya, V., Taku-Forchu, N., Roche, L. M., Ostoja, S. M., Parker, L. E., Zaccaria, D., Cooper, M. H., Diaz-Ramirez, J. N., Brodt, S., Battany, M., Rijal, J. P., & Pathak, T. B. (2024). Climate smart agriculture: Assessing needs and perceptions of California’s farmers. *Frontiers in Sustainable Food Systems*, *8*, 1395547\. https://doi.org/10.3389/fsufs.2024.1395547  
* Lamba, V., Hooda, S., Ahuja, R., & Kaur, A. (2021). Wheat yield prediction using feedforward neural networks. *2021 9th International Conference on Reliability, Infocom Technologies and Optimization (Trends and Future Directions) (ICRITO)*, 1–6. https://doi.org/10.1109/ICRITO51393.2021.9596464  
* Lan, P., Xia, K., Pan, Y., & Fan, S. (2021). An improved equilibrium optimizer algorithm and its application in LSTM neural network. *Symmetry*, *13*(9), 1706\. https://doi.org/10.3390/sym13091706  
* Li, Y., Wang, T., Zu, H. *et al.* Explainable AI-driven interpretation of environmental drivers of tomato fruit expansion in smart greenhouses using IoT sensing. *Sci Rep* **15**, 40870 (2025). https://doi.org/10.1038/s41598-025-24800-3  
* Lionel, B. M., Musabe, R., Gatera, O., & Twizere, C. (2025). A comparative study of machine learning models in predicting crop yield. *Discover Agriculture*, *3*(1), 151\. https://doi.org/10.1007/s44279-025-00335-z  
* Lu, J.; Li, J.; Fu, H.; Tang, X.;  Liu, Z.; Chen, H.; Sun, Y.; Ning, X.  Deep Learning for Multi-Source  Data-Driven Crop Yield Prediction in  Northeast China. Agriculture 2024, 14,  794\. https://doi.org/10.3390/  agriculture14060794  
* Udeh, E. L., Abdullahi, T. Y., & Bulama, L. (2024). Analysis of rainfall and temperature variability on crop yield in Lere Local Government Area of Kaduna State, Nigeria. *British Journal of Earth Sciences Research*, *12*(4), 44–54. https://doi.org/10.37745/bjesr.2013/vol12n44454  
* Wang, L., Chen, Z., Liu, W., & Huang, H. (2024). A Temporal–Geospatial Deep Learning Framework for Crop Yield Prediction. *Electronics*, *13*(21), 4273\. https://doi.org/10.3390/electronics13214273  
* Yakubu, M. A., Yakubu, U., Yakubu, H., & Mayun, F. A. (2024). Artificial intelligence applications in sustainable Agriculture in Nigeria: A comprehensive review. *Journal of Basics and Applied Sciences Research*, *2*(4), 84-94. https://doi.org/10.33003/jobasr-2024-v2i4-70