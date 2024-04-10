<a href="https://www.linkedin.com/in/thitirat-meennuch" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>
</a>

<a href="mailto:thitiratmnc@gmail.com">
    <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"/>
</a>

# Portfolio 

## Table of Contents
- [Data Science & Machine Learning](#heading2)
  - [Credit Card Market Segmentation and Cluster Prediction](#heading3)
  - [Restuarants Review Rating Classification](#heading3)
- [Artificial Intelligence (AI)](#heading2)
  - [Facial Expression Recognition System](#heading3)
- [Machine Learning for NLP](#heading2)
  - [Aspect CateMachine Learning for NLP Projectsgory and Polarity Classification](#heading3)
  - [Next Word Prediction](#heading3)
  - [Name Entity Recognition (NER) for Thai Language](#heading3)
- [Project Repositories](#heading2)
- [Data Engineer Workshop](#heading2)

## Data Science & Machine Learning Projects <a name="heading2"></a>

- ### Credit Card Market Segmentation and Cluster Prediction - [*Explore Project*](https://github.com/thitirat-mnc/credit-card-customer-segmentation/tree/main) <a name="heading3"></a>
    - [x] *Python, Exploratory Data Analysis (EDA), Pandas, Clustering, PCA, Logistic Regression* <br>
        [![CreditCard](https://github-readme-stats.vercel.app/api/pin/?username=thitirat-mnc&repo=credit-card-customer-segmentation&show_icons=true&theme=highcontrast)](https://github.com/thitirat-mnc/credit-card-customer-segmentation/) <img width="200" src="https://github.com/thitirat-mnc/DataSci-ML-Portfolio/assets/134206687/23fad05a-4afe-49ee-98b9-16ab577d40b5"> <img width="330" src="https://github.com/thitirat-mnc/DataSci-ML-Portfolio/assets/134206687/29c8be88-4f7d-4560-86d2-686d2fcb637e">
---


- ### Restuarants Rating Classification - [*Explore Project*](https://github.com/thitirat-mnc/wongnai-restuarants-rating-classification/) <a name="heading3"></a>
  - [x] *Python, BERT, Logistic Regression, Pandas* <br>
  
    [![Restaurant Review](https://github-readme-stats.vercel.app/api/pin/?username=thitirat-mnc&repo=wongnai-restuarants-rating-classification&show_icons=true&theme=algolia)](https://github.com/thitirat-mnc/wongnai-restuarants-rating-classification/)<img width="200" src="https://github.com/thitirat-mnc/DataSci-ML-Portfolio/assets/134206687/b64e5c9a-adec-4c82-af9b-d714e7ab524e"> <img width="200" src="https://github.com/thitirat-mnc/DataSci-ML-Portfolio/assets/134206687/e0419034-d316-45e0-960e-0f13059f0d5b">
---

## Artificial Intelligence (AI) Projects <a name="heading2"></a>
- ### Facial Expression Recognition System - [*Explore Project*](https://github.com/thitirat-mnc/DataSci-ML-Portfolio/tree/main/Facial%20Recognition) <a name="heading3"></a>
    - [x] *Python, Exploratory Data Analysis (EDA), Pandas, Image augmentation, Data normalization, CNNs, RESNET* <br>
    
        <img width="230" src="https://github.com/thitirat-mnc/DataSci-ML-Portfolio/assets/134206687/41985af9-023c-4e80-84a3-563fdf03cfe6"> <img width="300" src="https://github.com/thitirat-mnc/DataSci-ML-Portfolio/assets/134206687/f47f7375-db7c-4856-ae30-7870c0b91b8e"> <img width="300" src="https://github.com/thitirat-mnc/DataSci-ML-Portfolio/assets/134206687/1f5d6889-bb9e-40c0-ab63-c12974c243fd">
        
         <details><summary>Overview</summary>
          <ul>
            <li><strong>A system that automatically monitors people's emotions and expressions based on facial images</strong>
              <ul>
                <li>The dataset comprises 2000 images with facial key-point annotations and 20,000 facial images, each labeled with facial expression categories.</li>
                <li>The tasks include detecting <strong>facial key points</strong> and categorizing each face into one of five <strong>emotion</strong> categories.</li>
              </ul>
            </li>
            <li><strong>Tasks:</strong>
              <ul>
                <li>Perform image visualizations to understand the dataset.</li>
                <li>Perform <strong>image augmentation</strong> to increase dataset diversity.</li>
                <li>Conduct data <strong>normalization</strong> and prepare training data for model training.</li>
                <li>Build deep Convolutional Neural Networks <strong>(CNNs)</strong> and residual neural network <strong>(RESNET)</strong> models for facial key points detection.</li>
                <li>Save the trained model for <strong>deployment</strong>.</li>
              </ul>
            </li>
          </ul>
        </details>
---

## Machine Learning for NLP Projects  <a name="heading2"></a>
- ### Aspect Category and Polarity Classification - [*Explore Project*](https://github.com/thitirat-mnc/DataSci-ML-Portfolio/tree/main/Restuarant-Review%20Sentiment-Aspect%20Classification)  <a name="heading3"></a>
    - [x] *Python, Pandas, nltk toolkit, Spacy, Logistic Regression, DAN, CNNs*<br>
    
        <img width="300" src="https://github.com/thitirat-mnc/DataSci-ML-Portfolio/assets/134206687/a1af57b5-06ba-4554-b087-166623f64921"> <img width="300" src="https://github.com/thitirat-mnc/DataSci-ML-Portfolio/assets/134206687/e9aa286b-3d45-428f-9f9b-93a139458296"> <img width="120" src="https://github.com/thitirat-mnc/DataSci-ML-Portfolio/assets/134206687/d0aa8e5e-28f1-4097-8945-a90409a0c121">

        <details><summary>Overview</summary>
            <ul>
              <ul>
                <li>The dataset contains 3156 rows. The text is drawn from restaurant reviews, tokenized using nltk.word_tokenize and non-English alphabet symbols were cleaned out using regular expression.</li>
                <li>The tasks include categorizing each text review into one of five <strong>aspect categories</strong> and into one of four <strong>sentiments</strong>.</li>
              </ul>
               <li><strong>Tasks :</strong>
                  <ul>
                    <li>Bag-of-word <strong>logistic regression</strong> model as a baseline for both sentiment and aspect classification. The features are created from the cleaned text.</li>
                    <li>Perform <strong>oversampling</strong> by multiplying the number of conflict label data in the training set to increase dataset diversity.</li>
                    <li>Trained both <strong>multi-class</strong> and multi-label logistic regression models for aspect classification.</li>
                    <li>For <strong>multi-label</strong>, used a binary logistic regression model to train each aspect model separately, and combine the end result prediction.</li>
                    <li>For Deep Learning Models, tried both pre-trained <strong>GloVe</strong> 300-dimensional word embeddings from stanford.edu and <strong>Word2Vec</strong>.</li>
                    <li>Build <strong>Deep Averaging Network (DAN)</strong> and <strong>Convolutional Neural Network (CNN)</strong>.</li>
                    <li>Tuned Hyperparameters using <strong>grid search</strong>.</li>
                  </ul>
              </li>
           <ul>
        </details>
---


- ### Next word Prediction - [*Explore Project*](https://github.com/thitirat-mnc/DataSci-ML-Portfolio/tree/main/Predicting%20Next%20Word)  <a name="heading3"></a>
      - [x] *Python, Tensorflow, Pandas* <br>
  
    <img width="350" alt="Screenshot 2567-04-11 at 00 25 23" src="https://github.com/thitirat-mnc/DataSci-ML-Portfolio/assets/134206687/b8a9bfc1-420b-48e6-9f8d-b09a3a9dc625">
    <details><summary>Overview</summary>
      <ul>
         <li><strong>Predicting next word based on the first letter</strong>
            <ul>
                <li>The training set is drawn from <a href="https://huggingface.co/datasets/gigaword">https://huggingface.co/datasets/gigaword</a></li>
                <li>The development set provided for evaluation contains 94,825 rows with 3 columns:</li>
                <li>'context' column, 'first letter' column, and 'prediction' column.</li>
                <li>The 'first letter' column provides the initial letter of the word to be predicted for each context, while the 'prediction' column contains the actual word that is to be generated.</li>
            </ul>
        </li>
        <li><strong>Tasks :</strong>
          <ul>
                <li>For <strong>trigram</strong> model, a counter dictionary is used to count the number of occurrences of each trigram in the training data.</li>
                <li>The model is trained in small batches, with a batch size of 2048, to accommodate the large size of the training set.</li>
                <li>Once the model has been trained, the probability of each word is computed based on its frequency in the training data.</li>
                <li>The trigram model is then used to predict the next word in a given context by selecting the word with the highest probability.</li>
                <li>For kenlm (pre-trained <strong>5-gram model</strong>), the next word in a given context is generated by looping over each word in the model's vocabulary and selecting the word with the highest probability.</li>
          </ul>
        </li>
        </ul>
    </details>
---        


- ### Name Entitiy Recognition (NER) for Thai Language - [*Explore Project*](https://github.com/thitirat-mnc/DataSci-ML-Portfolio/tree/main/Thai%20Name%20Entity%20Recognition)  <a name="heading3"></a>
    - [x] *Python, Scikit-learn, Pandas, pythainlp* <br>
    
    <img width="300" alt="Screenshot 2567-04-11 at 00 21 21" src="https://github.com/thitirat-mnc/DataSci-ML-Portfolio/assets/134206687/9b980dd2-8f16-4e2b-8a53-03448b72c970">
    <details><summary>Overview</summary>
      <ul>
        <li><strong>Name Entity Recognition (NER) for Thai Language</strong>
            <ul>
                <li>The training and development data for this project were in Thai, and were first tokenized and separated by '|' using the pythainlp library (newmm dictionary).</li>
                <li>The resulting text was then tagged with entity types, including 'ORG', 'PER', 'MEA', 'LOC', 'TTL', 'DTM', 'NUM', 'DES', 'MISC', 'TRM', and 'BRN', using 'B_' before each tag.</li>
                <li>Each word and tag were separated by '\t', while sentences were separated by '\n'.</li>
                <li>To preprocess the training data for the models, each word and tag in the dataset were split and stored in two separate lists: one for token sequences and one for label sequences.</li>
            </ul>
        </li>
        <li><strong>Tasks :</strong>
            <ul>
                <li>Implemented Conditional Random Fields <strong>(CRF)</strong> with only the word, the previous word, and the next word as features as baseline.</li>
                <li>Added <strong>conjunctive features</strong> to the model, which took the form of {word i-1 – word i – word i+1}, resembling bigram and trigram features to capture more contextual information about the words.</li>
                <li>Explored the use of conjunctive part- of-speech <strong>(POS) tags</strong> as a feature to recognize named entities based on grammatical context, using the pythainlp pos_tag (orchid_ud).</li>
              </ul>
        </li>
        </ul>
    </details>

---

## Projects Repositories  <a name="heading2"></a>
[![CreditCard](https://github-readme-stats.vercel.app/api/pin/?username=thitirat-mnc&repo=credit-card-customer-segmentation&show_icons=true&theme=highcontrast)](https://github.com/thitirat-mnc/credit-card-customer-segmentation/)
[![Restaurant Review](https://github-readme-stats.vercel.app/api/pin/?username=thitirat-mnc&repo=wongnai-restuarants-rating-classification&show_icons=true&theme=algolia)](https://github.com/thitirat-mnc/wongnai-restuarants-rating-classification/)
[![Netflix-top10](https://github-readme-stats.vercel.app/api/pin/?username=thitirat-mnc&repo=Netflix-top10-dataset&show_icons=true&theme=algolia)](https://github.com/thitirat-mnc/Netflix-top10-dataset/)
[![BizGen](https://github-readme-stats.vercel.app/api/pin/?username=thitirat-mnc&repo=BizGen&show_icons=true&theme=highcontrast)](https://github.com/thitirat-mnc/BizGen/)


## Data Engineer Workshop  <a name="heading2"></a>
[![Data-EngineerR2DE](https://github-readme-stats.vercel.app/api/pin/?username=thitirat-mnc&repo=Data-Engineer-Workshop-R2DE&show_icons=true&theme=catppuccin_latte)](https://github.com/thitirat-mnc/Data-Engineer-Workshop-R2DE/)
