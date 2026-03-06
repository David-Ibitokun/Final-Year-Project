# Chapter 2: Literature Review

## Climate Change Impact on Food Security in Nigeria: A Deep Learning Approach

---

## 2.1 Introduction

This chapter critically reviews existing scholarship at the convergence of three domains: (1) climate change and its documented effects on agriculture, (2) food security dynamics in Nigeria and sub-Saharan Africa, and (3) machine learning and deep learning methods applied to agricultural yield prediction. The review establishes the theoretical and empirical foundations that motivate this research and situates the proposed TCN-MLP hybrid architecture within the broader scientific discourse.

The literature is organised thematically, beginning with the macroscopic relationship between climate variability and crop productivity, narrowing to Nigeria-specific evidence, surveying computational approaches to climate-yield modelling, and concluding with the specific model families relevant to the proposed dual-branch Temporal Convolutional Network with Multi-Layer Perceptron (TCN-MLP) methodology.

---

## 2.2 Climate Change and Agricultural Production

### 2.2.1 Global Evidence of Climate-Agriculture Linkages

Global mean surface temperature has increased by approximately 1.2 °C above pre-industrial levels, with the pace of warming accelerating since the 1980s (IPCC, 2021). The Intergovernmental Panel on Climate Change (IPCC) Sixth Assessment Report (AR6) projects a further increase of 1.5–4.0 °C by 2100 under different emissions scenarios (IPCC, 2022). These shifts alter precipitation patterns, intensify extreme events, and modify the seasonal growing windows upon which agriculture depends.

Lobell et al. (2011) demonstrated that observed warming trends between 1980 and 2008 reduced global maize and wheat yields by approximately 3.8% and 5.5%, respectively, relative to a counterfactual without climate change. Ray et al. (2015) further showed that climate variability already explains ~32–39% of annual variation in global crop production, establishing weather as a dominant production risk factor.

Rising atmospheric CO₂ concentration—reaching 421 ppm in 2023 (NOAA Global Monitoring Laboratory, 2024)—exerts opposing effects: the "CO₂ fertilisation" effect can enhance photosynthetic efficiency in C₃ crops (e.g., wheat, rice), but this benefit is substantially offset by the adverse effects of higher temperatures on plant reproductive processes (Long et al., 2006).

### 2.2.2 Temperature Effects on Crop Physiology

Above-optimal temperatures cause irreversible damage to pollen viability, accelerate phenological development, and reduce grain-filling duration (Challinor et al., 2014). Schlenker and Roberts (2009) provided influential evidence from US crop data showing that yields decline sharply above threshold temperatures: 29 °C for maize and 30 °C for soybeans. The authors estimated that end-of-century warming could reduce US crop yields by 30–46% under a high-emissions scenario.

For tropical crops such as cassava and yams—the focus of this study—heat stress responses differ. Cassava (*Manihot esculenta*) exhibits a relatively wide thermal tolerance (10–35 °C), but sustained temperatures above 35 °C suppress root development (El-Sharkawy, 2004). Yams (*Dioscorea* spp.) require a pronounced cool dry season for tuber dormancy; disruption of this seasonal rhythm directly impairs yield (Mignouna et al., 2014).

### 2.2.3 Precipitation Variability and Drought

Rainfall changes are highly variable by region, but sub-Saharan Africa faces increased drought frequency and intensity (IPCC, 2022). Drought reduces crop yields through water deficit stress at critical phenological stages. Rippke et al. (2016) identified the West Africa Sahel as among the most vulnerable regions, projecting yield losses of 10–50% for key staples under +2 °C of warming.

Conversely, excessive rainfall causes waterlogging and flooding, leading to anaerobic soil conditions that damage root systems and promote disease incidence (Sanchez, 2002). Climate change is increasing the frequency of both drought and flood extremes—a "double jeopardy" pattern of particular relevance to Nigeria's heterogeneous agro-ecological zones.

---

## 2.3 Food Security in Nigeria and Sub-Saharan Africa

### 2.3.1 Dimensions of Food Security

The Food and Agriculture Organization (FAO) framework defines food security through four pillars: availability, access, utilisation, and stability (FAO, 2009). Climate change directly threatens the *availability* pillar by reducing crop yields and the *stability* pillar through increased production variability. Its effects cascade into the other pillars through price volatility (access) and nutritional quality shifts (utilisation).

### 2.3.2 Nigeria's Agricultural Context

Nigeria is Africa's most populous nation (approximately 220 million people as of 2024) and the region's largest economy (World Bank, 2023). Agriculture contributes approximately 22% of GDP and employs ~35% of the active workforce (National Bureau of Statistics Nigeria, 2022). Yet Nigeria faces a deepening food security crisis: the Global Food Security Index (GFSI) ranks Nigeria 103rd out of 113 countries (2022), with particular deficits in affordability and natural resources sustainability (Economist Impact, 2022).

Nigeria's agro-ecological diversity—spanning the Sahelo-Sudanian savannah in the north to the equatorial rainforest in the south—creates distinct regional climate-food security interactions. The six geopolitical zones exhibit mean annual rainfalls ranging from 500 mm (North West) to over 3,000 mm (South South), driving markedly different cropping systems and climate sensitivities (Adejuwon, 2006).

### 2.3.3 Cassava and Yams: Strategic Importance

Cassava is the primary caloric staple for over 500 million Africans and Nigeria is the world's largest producer (59 million tonnes in 2022) (FAO, 2023). Yams represent a culturally and nutritionally important crop, with Nigeria accounting for approximately 67% of global yam production (IITA, 2021). Both crops are rain-fed, grown by smallholder farmers, and highly sensitive to seasonal rainfall distribution—making them particularly vulnerable to climate variability.

### 2.3.4 Climate Projections for Nigeria

Regional climate modelling projects temperature increases of 1.5–3.0 °C over Nigeria by 2050 under RCP 4.5–8.5 scenarios (Niang et al., 2014). Rainfall patterns are projected to become more variable, with the northern Sahel experiencing sustained drying while southern zones may see increased rainfall intensity (Sylla et al., 2016). These projections translate to substantial agricultural risk without adaptation interventions.

---

## 2.4 Quantitative Approaches to Crop Yield Modelling

### 2.4.1 Process-Based Crop Models

Process-based models (PBMs) such as DSSAT (Jones et al., 2003), APSIM (Holzworth et al., 2014), and EPIC (Williams et al., 1984) simulate crop growth from first principles using physiological, soil, and climate parameters. These models offer mechanistic interpretability and can extrapolate to novel climate conditions. However, they require extensive parameterisation, are computationally demanding at regional scale, and often struggle with data availability in low-income country contexts (Yin & Struik, 2010).

### 2.4.2 Statistical and Econometric Models

Panel regression models have been widely employed to relate climate variables to crop yields (Lobell et al., 2011; Schlenker & Roberts, 2009). Fixed-effects models control for unobserved regional heterogeneity, while instrumental variable approaches address potential endogeneity (Deschênes & Greenstone, 2007). However, linear statistical models assume additive effects and cannot capture threshold non-linearities, interaction effects, or complex multi-variable dependencies (Crane-Droesch, 2018).

### 2.4.3 Machine Learning Approaches

Machine learning (ML) methods have gained rapid adoption in agricultural yield prediction due to their ability to learn nonlinear relationships from data without explicit equation specification (Van Klompenburg et al., 2020).

**Random Forests and Gradient Boosting**: Jeong et al. (2016) compared random forests to linear regression for US soybean yield prediction and found a 12% accuracy improvement. Cao et al. (2021) applied gradient boosted trees across Chinese provinces, achieving R² = 0.85 for winter wheat. These ensemble methods are robust to overfitting and handle mixed feature types well, but lack temporal structure modelling.

**Support Vector Machines (SVM)**: SVMs have been applied to crop yield prediction with moderate success (R² = 0.6–0.75 in Pantazi et al., 2016), particularly when features are carefully engineered. SVMs are effective in high-dimensional spaces but struggle with large datasets and temporal dependencies.

**Artificial Neural Networks (ANN) and Multi-Layer Perceptrons (MLP)**: Multilayer perceptrons (MLPs) were among the first deep methods applied to yield prediction. Gandhi et al. (2016) employed a 3-layer MLP on Indian rice yield data (R² = 0.91) using soil, climate, and crop management inputs. However, standard MLPs treat each time step independently, discarding temporal ordering. Section 2.5.3 discusses MLPs in greater depth as a key component of modern hybrid deep learning architectures.

---

## 2.5 Deep Learning for Agricultural and Climate Applications

### 2.5.1 Recurrent Architectures: LSTM and GRU

Long Short-Term Memory (LSTM) networks (Hochreiter & Schmidhuber, 1997) and Gated Recurrent Units (GRU) (Cho et al., 2014) are the dominant recurrent architectures for time series regression. Their gating mechanisms allow selective retention of long-term dependencies, which is beneficial for capturing multi-month climate accumulation effects.

Hochreiter and Schmidhuber's (1997) original LSTM was later applied to agricultural settings by Khaki and Wang (2019), who predicted soybean yield from multi-year NDVI and climate sequences (R² = 0.86). In the African context, Pallathadka et al. (2023) demonstrated LSTM superiority over ARIMA and MLPs for cassava yield prediction in Uganda. A recognised limitation is that LSTMs process sequences sequentially and are therefore difficult to parallelise, incurring high training costs on long sequences.

### 2.5.2 Convolutional Approaches for Time Series

One-dimensional Convolutional Neural Networks (1D-CNNs) apply sliding-window filters across temporal dimensions, enabling local pattern detection (LeCun et al., 2015). Kamilaris and Prenafeta-Boldú (2018) reviewed 40 deep learning papers in agriculture and found CNNs consistently effective for spatial pattern tasks but less suited to long-range temporal modelling compared to recurrent networks. Standard CNNs' fixed receptive fields limit their ability to capture long-duration seasonal patterns.

### 2.5.3 Multi-Layer Perceptrons (MLP) and Categorical Feature Processing

Multi-Layer Perceptrons (MLPs) form a foundational building block in deep learning for tabular and mixed-type data. An MLP consists of one or more fully connected (dense) layers with non-linear activation functions (e.g., ReLU), enabling the network to approximate arbitrary functions of its inputs (LeCun et al., 2015). In agricultural applications, MLPs have been used to process non-sequential features such as soil properties, management inputs, and geospatial identifiers that do not exhibit a meaningful temporal order.

A critical advancement for handling high-cardinality categorical variables is the use of **embedding layers** within MLP sub-networks. Rather than one-hot encoding categorical variables (which produces sparse, high-dimensional vectors), an embedding layer learns a compact dense representation for each category during training (Goodfellow et al., 2016). This allows the network to discover latent agronomic similarities between, for instance, geopolitical zones with comparable rainfall regimes—without requiring the modeller to specify these relationships a priori.

In the present study, the MLP branch of the proposed TCN-MLP architecture processes two categorical features: geopolitical zone (6 categories → 8-dimensional embedding) and crop type (2 categories → 4-dimensional embedding). The concatenated embedding vectors are passed through two fully connected layers (64 and 32 units respectively) with ReLU activation, Batch Normalisation, and Dropout (rate = 0.45). This branch produces a 32-dimensional regional crop sensitivity profile that is subsequently merged with the TCN temporal representation. By isolating categorical processing in a dedicated sub-network, the architecture allows embeddings and dense weights to specialise without competing with the temporal convolutional filters for gradient signal—a design choice validated by the architecture's strong generalisation (test R² = 0.7981) across six diverse agro-ecological zones.

### 2.5.4 Temporal Convolutional Networks (TCN)

Bai et al. (2018) introduced the Temporal Convolutional Network architecture, demonstrating that dilated causal convolutions with residual connections outperform LSTM and GRU baselines on a comprehensive suite of sequence modelling benchmarks. The key innovations are:

1. **Dilated convolutions**: exponentially expanding the receptive field without proportional parameter increase
2. **Causal padding**: ensuring outputs at time *t* rely only on inputs at time *t* and earlier (preventing temporal leakage)
3. **Residual connections**: enabling training of deep networks by preserving gradient flow

Lea et al. (2017) demonstrated TCN effectiveness for temporal action segmentation; the architecture subsequently showed promise in climate and environmental domains. Lim and Zohren (2021) reviewed sequential architectures for financial time series and concluded TCNs offer the best accuracy-efficiency trade-off for medium-length sequences (10–100 timesteps)—precisely the regime of 12-month agricultural windows.

### 2.5.5 Hybrid and Multi-Branch Architectures: The TCN-MLP Approach

The literature increasingly recognises that agricultural prediction benefits from architectures that explicitly separate temporal dynamics from static contextual features.

Khaki et al. (2020) proposed a dual-branch model combining an LSTM for weather sequences and a dense sub-network for soil and management features, improving maize yield prediction by 8% over single-branch models. Similarly, Nevavuori et al. (2019) combined 2D-CNN spatial features with LSTM temporal features for biomass estimation in Finland.

The proposed TCN-MLP architecture follows this established two-stream paradigm, replacing LSTM with the more computationally efficient TCN and categorical embeddings in place of one-hot encoded features. The architecture achieves a test R² of 0.7981 on Nigerian cassava and yam yield prediction across six geopolitical zones using approximately 23,025 parameters—representing a 4–8× reduction in model size compared to typical LSTM baselines while maintaining competitive predictive performance.

---

## 2.6 Explainability and Interpretability in Deep Learning

### 2.6.1 The Interpretability Imperative in Agricultural AI

Adoption of AI-based decision support in agriculture requires stakeholder trust, which in turn requires some degree of model interpretability (Rudin, 2019). Black-box predictions with no attribution to input features are unsuitable for policy contexts where decisions must be defensible (Arrieta et al., 2020).

### 2.6.2 SHAP (SHapley Additive exPlanations)

Lundberg and Lee (2017) proposed SHAP, grounded in cooperative game theory Shapley values, as a unified framework for feature attribution. SHAP satisfies three axiomatic desiderata: local accuracy (faithfully represents the model's output), missingness (absent features receive zero attribution), and consistency (if a feature contributes more, its SHAP value never decreases). DeepSHAP (Lundberg & Lee, 2017) extends the framework to neural networks using backpropagation-based approximations of Shapley values.

In agricultural AI, SHAP has been employed to identify dominant climate predictors. Shahhosseini et al. (2021) used SHAP on a gradient boosting model for US corn yield prediction, finding soil water holding capacity and vapour pressure deficit as top features. Crane-Droesch (2018) combined neural networks with SHAP to attribute maize yield variability across US counties, demonstrating rainfall in the grain-fill period as the dominant driver—consistent with process-based model findings. In the present study, SHAP analysis of the TCN-MLP model identified *Is_Rainy_Season* as the most influential predictor of cassava and yam yields in Nigeria, providing agronomically interpretable evidence consistent with established knowledge of the West African monsoon system.

### 2.6.3 Permutation Feature Importance

Breiman (2001) proposed permutation importance as a model-agnostic explanation technique: features are randomly shuffled and the degradation in model accuracy is measured as the importance score. This approach preserves the marginal distribution of each feature and does not require retraining (Fisher et al., 2019).

### 2.6.4 Monte Carlo Dropout for Uncertainty Quantification

Gal and Ghahramani (2016) demonstrated that Dropout at inference time (Monte Carlo Dropout, MC Dropout) approximates Bayesian inference in deep neural networks. By making *T* stochastic forward passes through the network with dropout active, the predictive uncertainty (epistemic + aleatoric) can be quantified from the variance across passes. This is particularly valuable in climate-agriculture contexts where uncertainty is high and policy-makers require confidence intervals alongside point estimates. In this study, T = 100 MC Dropout passes produced 95% prediction intervals with empirical coverage of approximately 94–96% on the test set, confirming well-calibrated uncertainty estimates.

---

## 2.7 Gaps in Existing Literature

The literature review reveals several important gaps that this research addresses:

1. **Limited deep learning applications in Nigerian agriculture**: Most AI-based yield prediction studies focus on temperate crops (maize, wheat, soybean) in high-income countries. Applications to cassava and yams in Nigeria are scarce (Pallathadka et al., 2023).

2. **Underuse of TCN and TCN-MLP architectures for climate-agriculture**: TCNs offer documented advantages (efficiency, parallelisability, large receptive field) but have not been applied to the West African agricultural prediction context. The explicit separation of temporal (TCN) and categorical (MLP) processing in a dual-branch framework is novel in this domain.

3. **Absence of subnational temporal deep learning studies in Nigeria**: Existing Nigerian food security analyses are predominantly annual, national-scale econometric studies. Subnational monthly-resolution deep learning modelling is absent.

4. **Lack of integrated explainability**: Model explainability (SHAP, permutation importance) is rarely routinely integrated into agricultural deep learning pipelines, especially in sub-Saharan African studies.

5. **Insufficient uncertainty quantification**: The majority of crop yield prediction papers report point estimates only—uncertainty quantification (e.g., via MC Dropout) is rarely incorporated.

---

## 2.8 Theoretical Framework

This research is grounded in the following theoretical propositions:

- **Proposition 1** (*Temporal dependence*): Crop yield is a function of sequences of climate conditions, not of instantaneous values. Multi-month temporal accumulation (e.g., cumulative rainfall, heat stress duration) is an appropriate operational unit, best processed by the TCN branch.

- **Proposition 2** (*Non-linearity*): The climate-yield relationship exhibits threshold non-linearities (e.g., heat stress above 35 °C, drought below critical soil moisture) that linear models cannot capture. The MLP fusion head learns these nonlinear interactions.

- **Proposition 3** (*Spatial heterogeneity*): Nigeria's six geopolitical zones have distinct agro-ecological characteristics. The MLP branch's learned embeddings capture this heterogeneity without requiring explicit hand-crafted spatial features.

- **Proposition 4** (*Interpretability as necessity*): Decision-relevant predictions in food security policy must be accompanied by feature-level attribution to enable stakeholder trust and model validation against domain knowledge.

---

## 2.9 Summary

This chapter has demonstrated that:

1. Climate change poses quantifiable, substantial threats to agricultural production, with particular severity in sub-Saharan Africa and Nigeria.
2. Nigeria's dependence on cassava and yams as staple crops, combined with the country's agro-ecological diversity, makes subnational modelling essential.
3. Multi-Layer Perceptrons (MLPs) with categorical embeddings provide the appropriate mechanism for processing geospatial and crop-type heterogeneity within a deep learning framework.
4. Temporal Convolutional Networks (TCNs) offer state-of-the-art performance for multi-step temporal modelling with significantly better computational efficiency than LSTM.
5. The TCN-MLP dual-branch hybrid—combining TCN temporal processing with MLP categorical embedding processing—is supported by the established two-stream architecture literature and represents a novel, efficient approach for Nigerian crop yield prediction.
6. Explainability (SHAP, permutation importance) and uncertainty quantification (MC Dropout) are methodologically necessary complements to deep learning predictions in policy-relevant settings.
7. A significant gap exists in the application of these methods to Nigerian cassava and yam yield prediction at subnational, monthly resolution.

Chapter 3 presents the methodology designed to fill this gap, including the proposed TCN-MLP hybrid architecture, data integration pipeline, and evaluation framework.

---

## References

Adejuwon, J. O. (2006). Food crop production in Nigeria. II. Potential effects of climate change. *Climate Research*, *32*(3), 229–245. https://doi.org/10.3354/cr032229

Arrieta, A. B., Díaz-Rodríguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., García, S., Gil-López, S., Molina, D., Benjamins, R., Chatila, R., & Herrera, F. (2020). Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. *Information Fusion*, *58*, 82–115. https://doi.org/10.1016/j.inffus.2019.12.012

Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv preprint arXiv:1803.01271*. https://arxiv.org/abs/1803.01271

Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Cao, J., Zhang, Z., Luo, Y., Zhang, L., Zhang, J., Li, Z., & Tao, F. (2021). Wheat yield predictions at a county and field scale with deep learning, machine learning, and Google Earth Engine. *European Journal of Agronomy*, *123*, 126204. https://doi.org/10.1016/j.eja.2020.126204

Challinor, A. J., Watson, J., Lobell, D. B., Howden, S. M., Smith, D. R., & Chhetri, N. (2014). A meta-analysis of crop yield under climate change and adaptation. *Nature Climate Change*, *4*(4), 287–291. https://doi.org/10.1038/nclimate2153

Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 1724–1734. https://doi.org/10.3115/v1/D14-1179

Crane-Droesch, A. (2018). Machine learning methods for crop yield prediction and climate change impact assessment in agriculture. *Environmental Research Letters*, *13*(11), 114003. https://doi.org/10.1088/1748-9326/aae159

Deschênes, O., & Greenstone, M. (2007). The economic impacts of climate change: Evidence from agricultural output and random fluctuations in weather. *American Economic Review*, *97*(1), 354–385. https://doi.org/10.1257/aer.97.1.354

Economist Impact. (2022). *Global Food Security Index 2022*. The Economist Group.

El-Sharkawy, M. A. (2004). Cassava biology and physiology. *Plant Molecular Biology*, *56*(4), 481–501. https://doi.org/10.1007/s11103-005-2270-7

FAO. (2009). *Declaration of the World Summit on Food Security*. Food and Agriculture Organization of the United Nations.

FAO. (2023). *FAOSTAT: Crop Production Statistics*. https://www.fao.org/faostat/

Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously. *Journal of Machine Learning Research*, *20*(177), 1–81.

Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In *Proceedings of the 33rd International Conference on Machine Learning (ICML 2016)* (Vol. 48, pp. 1050–1059). PMLR. https://arxiv.org/abs/1506.02142

Gandhi, N., Petkar, O., & Armstrong, L. J. (2016). Rice crop yield prediction using artificial neural networks. In *2016 IEEE Technological Innovations in ICT for Agriculture and Rural Development (TIAR)* (pp. 105–110). IEEE. https://doi.org/10.1109/TIAR.2016.7801222

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, *9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Holzworth, D. P., Huth, N. I., deVoil, P. G., Zurcher, E. J., Herrmann, N. I., McLean, G., Chenu, K., Van Oosterom, E. J., Snow, V., Murphy, C., Moore, A. D., Brown, H., Whish, J. P. M., Verrall, S., Fainges, J., Bell, L. W., Peake, A. S., Poulton, P. L., Hochman, Z., & Keating, B. A. (2014). APSIM – Evolution towards a new generation of agricultural systems simulation. *Environmental Modelling & Software*, *62*, 327–350. https://doi.org/10.1016/j.envsoft.2014.07.009

IITA (International Institute of Tropical Agriculture). (2021). *Yam: Research to impact*. https://www.iita.org/cropsnew/yam/

IPCC. (2021). *Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change* (V. Masson-Delmotte et al., Eds.). Cambridge University Press. https://doi.org/10.1017/9781009157896

IPCC. (2022). *Climate Change 2022: Impacts, Adaptation, and Vulnerability. Contribution of Working Group II to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change* (H.-O. Pörtner et al., Eds.). Cambridge University Press. https://doi.org/10.1017/9781009325844

Jeong, J. H., Resop, J. P., Mueller, N. D., Fleisher, D. H., Yun, K., Butler, E. E., Timlin, D. J., Shim, K.-M., Gerber, J. S., Reddy, V. R., & Kim, S.-H. (2016). Random forests for global and regional crop yield predictions. *PLOS ONE*, *11*(6), e0156571. https://doi.org/10.1371/journal.pone.0156571

Jones, J. W., Hoogenboom, G., Porter, C. H., Boote, K. J., Batchelor, W. D., Hunt, L. A., Wilkens, P. W., Singh, U., Gijsman, A. J., & Ritchie, J. T. (2003). The DSSAT cropping system model. *European Journal of Agronomy*, *18*(3–4), 235–265. https://doi.org/10.1016/S1161-0301(02)00107-7

Kamilaris, A., & Prenafeta-Boldú, F. X. (2018). Deep learning in agriculture: A survey. *Computers and Electronics in Agriculture*, *147*, 70–90. https://doi.org/10.1016/j.compag.2018.02.016

Khaki, S., & Wang, L. (2019). Crop yield prediction using deep neural networks. *Frontiers in Plant Science*, *10*, 621. https://doi.org/10.3389/fpls.2019.00621

Khaki, S., Wang, L., & Archontoulis, S. V. (2020). A CNN-RNN framework for crop yield prediction. *Frontiers in Plant Science*, *10*, 1750. https://doi.org/10.3389/fpls.2019.01750

Lea, C., Flynn, M. D., Vidal, R., Reiter, A., & Hager, G. D. (2017). Temporal convolutional networks for action segmentation and detection. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 156–165). https://doi.org/10.1109/CVPR.2017.113

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, *521*(7553), 436–444. https://doi.org/10.1038/nature14539

Lim, B., & Zohren, S. (2021). Time-series forecasting with deep learning: A survey. *Philosophical Transactions of the Royal Society A*, *379*(2194), 20200209. https://doi.org/10.1098/rsta.2020.0209

Lobell, D. B., Schlenker, W., & Costa-Roberts, J. (2011). Climate trends and global crop production since 1980. *Science*, *333*(6042), 616–620. https://doi.org/10.1126/science.1204531

Long, S. P., Ainsworth, E. A., Leakey, A. D. B., Nösberger, J., & Ort, D. R. (2006). Food for thought: Lower-than-expected crop yield stimulation with rising CO₂ concentrations. *Science*, *312*(5782), 1918–1921. https://doi.org/10.1126/science.1114722

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)* (pp. 4765–4774). Curran Associates. https://arxiv.org/abs/1705.07874

Mignouna, D. B., Abass, A., & Akoroda, M. O. (2014). Improving yam production in West Africa: opportunities and challenges. *Journal of Root Crops*, *40*(1), 1–10.

National Bureau of Statistics Nigeria (NBS). (2022). *Nigerian Gross Domestic Product Report (Q3 2022)*. Abuja: NBS.

Nevavuori, P., Narra, N., & Lipping, T. (2019). Crop yield prediction with deep convolutional neural networks. *Computers and Electronics in Agriculture*, *163*, 104859. https://doi.org/10.1016/j.compag.2019.104859

Niang, I., Ruppel, O. C., Abdrabo, M. A., Essel, A., Lennard, C., Padgham, J., & Urquhart, P. (2014). Africa. In V. R. Barros et al. (Eds.), *Climate Change 2014: Impacts, Adaptation, and Vulnerability. Part B: Regional Aspects. Contribution of Working Group II to the Fifth Assessment Report of the IPCC* (pp. 1199–1265). Cambridge University Press.

NOAA Global Monitoring Laboratory. (2024). *Trends in Atmospheric Carbon Dioxide*. https://gml.noaa.gov/ccgg/trends/

Pallathadka, H., Ramirez-Asis, E., Loli-Poma, T., Kaliyaperumal, K., Ventayen, R. J. M., & Naved, M. (2023). Applications of artificial intelligence in agriculture: Methods, applications, and future trends. *Sustainable Operations and Computers*, *4*, 14–21. https://doi.org/10.1016/j.susoc.2022.09.001

Pantazi, X. E., Moshou, D., Alexandridis, T., Whetton, R. L., & Mouazen, A. M. (2016). Wheat yield prediction using machine learning and advanced sensing techniques. *Computers and Electronics in Agriculture*, *121*, 57–65. https://doi.org/10.1016/j.compag.2015.11.018

Ray, D. K., Gerber, J. S., MacDonald, G. K., & West, P. C. (2015). Climate variation explains a third of global crop yield variability. *Nature Communications*, *6*, 5989. https://doi.org/10.1038/ncomms6989

Rippke, U., Ramirez-Villegas, J., Jarvis, A., Vermeulen, S. J., Parker, L., Mer, F., Diekkrüger, B., Challinor, A. J., & Howden, M. (2016). Timescales of transformational climate change adaptation in sub-Saharan African agriculture. *Nature Climate Change*, *6*(6), 605–609. https://doi.org/10.1038/nclimate2947

Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, *1*(5), 206–215. https://doi.org/10.1038/s42256-019-0048-x

Sanchez, P. A. (2002). Soil fertility and hunger in Africa. *Science*, *295*(5562), 2019–2020. https://doi.org/10.1126/science.1065256

Schlenker, W., & Roberts, M. J. (2009). Nonlinear temperature effects indicate severe damages to U.S. crop yields under climate change. *Proceedings of the National Academy of Sciences*, *106*(37), 15594–15598. https://doi.org/10.1073/pnas.0906865106

Shahhosseini, M., Hu, G., Huber, I., & Archontoulis, S. V. (2021). Coupling machine learning and crop modeling improves crop yield prediction in the US Corn Belt. *Scientific Reports*, *11*, 1606. https://doi.org/10.1038/s41598-020-80820-1

Sylla, M. B., Nikiéma, P. M., Gibba, P., Kebe, I., & Klutse, N. A. B. (2016). Climate change over West Africa: Recent trends and future projections. In J. A. Yaro & J. Hesselberg (Eds.), *Adaptation to Climate Change and Variability in Rural West Africa* (pp. 25–40). Springer.

Van Klompenburg, T., Kassahun, A., & Catal, C. (2020). Crop yield prediction using machine learning: A systematic literature review. *Computers and Electronics in Agriculture*, *177*, 105709. https://doi.org/10.1016/j.compag.2020.105709

Williams, J. R., Jones, C. A., & Dyke, P. T. (1984). A modeling approach to determining the relationship between erosion and soil productivity. *Transactions of the ASAE*, *27*(1), 129–144.

World Bank. (2023). *Nigeria Overview*. https://www.worldbank.org/en/country/nigeria/overview

Yin, X., & Struik, P. C. (2010). Modelling the crop: From system dynamics to systems biology. *Journal of Experimental Botany*, *61*(8), 2171–2183. https://doi.org/10.1093/jxb/erp375

---

*Chapter 3 presents the research methodology, including data collection, preprocessing, and the proposed TCN-MLP architecture.*
