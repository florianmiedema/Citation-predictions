#### --- CITATIONS PREDICTION --- ####

## Florian Miedema
## florianmiedema@hotmail.com

#############################################
## libraries
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from numpy import inf
## imputation method from: https://scikit-learn.org/stable/modules/impute.html
from sklearn.impute import IterativeImputer
import re
#############################################

## Seed for reproducibility
np.random.seed(42)

## Train
train = pd.read_json('~/trainingdata.json')
## Test
test = pd.read_json('~/testingdata.json')

## Set citations apart
ytrain = train['citations']
ytrain = np.log1p(np.maximum(0, ytrain))

## Take away from featurespace
train = train.loc[:, train.columns != 'citations']

## Stack train and test set
df = pd.concat([train, test])

## Reset index
df = df.reset_index()
df = df.drop(['index'], axis = 1)

## Put citations back
train['citations'] = ytrain


### I. FEATURE ENGINEERING. ###


## Convert identifier to string
df['doi'] = df['doi'].astype(str)

## Before splitting into x and ytrain, inspect, clean and preprocess data
print(df.dtypes)

## Check missing values per column
df.isna().sum()

## Fill missing with [] for fields of study
df['fields_of_study'] = df['fields_of_study'].replace(np.nan, '[]')

## Create counting variables 
no_authors = []
no_topics = []
no_fields = []

for index, row in df.iterrows():
    size1 = len(row[3])
    size2 = len(row[7])
    size3 = len(row[9])
    ## Store
    no_authors.append(size1)
    no_topics.append(size2)
    no_fields.append(size3)
    
# 1. Authors
df['no_authors'] = no_authors

# 2. Topics
df['no_topics'] = no_topics

## -  3. Venue rank variable creation - ##

## Count amount of papers in venue
venue_appearance = df['venue'].value_counts()
venue_appearance = venue_appearance.reset_index()
venue_appearance.columns = ['venue', 'appearance']

## We correct for appearance in the ranking. Otherwise, a venue with only 1
## publication having a lot of citations could end up taking the hightest rank.
## merge
df["x"] = np.arange(len(df))
df = df.merge(venue_appearance, on = 'venue', how = 'left').set_index('x')
df = df.reset_index()
df = df.drop(['x'], axis = 1)

# Group mean citations by venue to make the ranking in the training set
tmp = train.groupby(['venue']).median()
## Save only citations
tmp = tmp[['citations']]
## Merge with appearance
tmp = tmp.merge(venue_appearance, on = 'venue')
tmp['appearance'] = np.log1p(np.maximum(0, tmp['appearance']))
## Multiply scaled appearance with number of citations in the venue.
tmp['rank'] = tmp['citations']*np.sqrt(tmp['appearance'])

## a. Create ranking for venue based on citations
tmp = tmp.sort_values(by = 'rank', ascending = False)
## Give rank to the venues
tmp['venue_rank'] = range(1,len(tmp)+1)
## Ranking done, now merge to the training set + preserve order.
## The trick for keeping the axis appropriate comes from:
## https://stackoverflow.com/questions/20206615/how-can-a-pandas-merge-preserve-order/28334396
df["x"] = np.arange(len(df))
df = df.merge(tmp, how='left', on='venue').set_index("x")
## Drop extra variable and rename old one
df = df.drop(['citations'], axis = 1)
df['venue_rank'] = df['venue_rank'].fillna(df['venue_rank'].median())
## Drop appearance_x
df = df.drop(['appearance_x'], axis = 1)

## - 4. Construct more features - ##

# 1 & 2. Topics (first two) and fields of study (first two) -> categorical
topics = []
fos = []
is_comp_science = []

for i in range(0, len(df)):
    topic_i = []
    topic_i2 = []
    topic_i3 = []
    fos_i = []
    fos_i2 = []
    fos_i3 = []
    
    ## Take out string
    if len(df['topics'][i]) >= 0:
        for a in range(len(df['topics'][i])):
            topic_i.append(df['topics'][i][a].lower())
        topic_i2 = ' '.join([str(s) for s in topic_i])
        topic_i3 = topic_i2.split()[:1]
        topics.append(' '.join([str(s) for s in topic_i3]))
        
    ## Take out string
    if len(df['fields_of_study'][i]) >= 0:
        for a in range(len(df['fields_of_study'][i])):
            fos_i.append(df['fields_of_study'][i][a].lower())
        fos_i2 = ' '.join([str(s) for s in fos_i])
        fos_i3 = fos_i2.split()[:4]
        fos.append(' '.join([str(s) for s in fos_i3]))
        
    if 'computer science' in fos[i]:
        is_comp_science.append(1)
    else:
        is_comp_science.append(0)
    
## Add both to dataframe after conversion to categorical
topics = pd.Categorical(topics)
fos = pd.Categorical(fos)
## Add
df['topics'], df['fields_of_study'], df['fos_is_cs'] = topics, fos, is_comp_science
## Encode categoricals
df['topics'] = df['topics'].cat.codes
df['fields_of_study'] = df['fields_of_study'].cat.codes

## b. Create ranking for fields of study, as for venue
## Merge citations to df
df2 = df
df2['citations'] = ytrain
## Fill missing values with median
tmp = df2.groupby(['fields_of_study']).median()
## Save only citations
tmp = tmp[['citations']]

## Create ranking for fos based on citations
tmp = tmp.sort_values(by = 'citations', ascending = False)
tmp.reset_index(inplace=True)
## Give rank to the venues
tmp['fos_rank'] = range(1,len(tmp)+1)
## Ranking done, now merge to the training set + preserve order
df["x"] = np.arange(len(df))
df = df.merge(tmp, how='left', on='fields_of_study').set_index("x")
## Drop extra variable and rename old one
df['fos_rank'] = df['fos_rank'].fillna(df['fos_rank'].median())
## Drop citations
df = df.drop(['citations_x', 'citations_y'], axis =1)

# 3. venue -> categorical
df['venue'] = pd.Categorical(df.venue)
df['venue_cat'] = df['venue'].cat.codes

# 4. is_open_acces, boolean -> binary
df['is_open_access'] = df['is_open_access'].astype(int)

# 5. year -> integer (same as in testing data)
df['year'] = df['year'].fillna(df['year'].mean())
df['year'] = df['year'].astype(int)

# 6. title length
df['title_length'] = df.title.str.len()

# 7. abstract length
df['abstract_length'] = df.abstract.str.len()
df['abstract_length'] = df['abstract_length'].fillna(0)

## reset df index
df = df.reset_index()
# df = df.drop(['index'], axis = 1)
df = df.drop(['x'], axis = 1)


### II. FIRST SPLIT IN TO XTEST AND XTRAIN ###


## Drop irrelevant (character) features
df = df.drop(['title', 'abstract', 'venue', 'venue_cat'], axis = 1)

## Split in sets
xtest = df[len(df)-1000:len(df)]
xtrain = df[0:len(df)-1000]


### III. TEXT FEATURE ENGINEERING. ###


## Libraries
from collections import Counter
import string

### Open training set and drop NA
df = pd.concat([train, test])
abstract = df['abstract']
abstract = abstract.dropna()
abstract = abstract.astype('str') 

### Loop for turning complete abstracts into single words
### Source: https://stackoverflow.com/questions/70160871/counting-occurrences-in-string-column 
abstractwords = []
for sentence in abstract:
    sentence = sentence.replace(".", " ")
    abstractwords.extend(sentence.split())
    
### Loop for lowercase all the abstractwords
### Source: https://stackoverflow.com/questions/17329113/convert-list-to-lower-case
def lowercase(words):
    for i in range(len(words)):
        words[i] = words[i].lower()
    return words
abstractwords = lowercase(abstractwords)

### Loop for delete punctuation in abstractwords
### Source: https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
abstractwords = [''.join(c for c in s if c not in string.punctuation) for s in abstractwords]

### Creating a list with stopwords
### Source: https://programminghistorian.org/en/lessons/counting-frequencies 
stopwords = ['', 'a', 'about', 'above', 'across', 'after', 'afterwards']
stopwords += ['again', 'against', 'all', 'almost', 'alone', 'along']
stopwords += ['already', 'also', 'although', 'always', 'am', 'among']
stopwords += ['amongst', 'amoungst', 'amount', 'an', 'and', 'another']
stopwords += ['any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere']
stopwords += ['are', 'around', 'as', 'at', 'back', 'be', 'became']
stopwords += ['because', 'become', 'becomes', 'becoming', 'been']
stopwords += ['before', 'beforehand', 'behind', 'being', 'below']
stopwords += ['beside', 'besides', 'between', 'beyond', 'bill', 'both']
stopwords += ['bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant']
stopwords += ['co', 'computer', 'con', 'could', 'couldnt', 'cry', 'de']
stopwords += ['describe', 'detail', 'did', 'do', 'done', 'down', 'due']
stopwords += ['during', 'each', 'eg', 'eight', 'either', 'eleven', 'else']
stopwords += ['elsewhere', 'empty', 'enough', 'etc', 'even', 'ever']
stopwords += ['every', 'everyone', 'everything', 'everywhere', 'except']
stopwords += ['few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first']
stopwords += ['five', 'for', 'former', 'formerly', 'forty', 'found']
stopwords += ['four', 'from', 'front', 'full', 'further', 'get', 'give']
stopwords += ['go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her']
stopwords += ['here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers']
stopwords += ['herself', 'him', 'himself', 'his', 'how', 'however']
stopwords += ['hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed']
stopwords += ['interest', 'into', 'is', 'it', 'its', 'itself', 'keep']
stopwords += ['last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made']
stopwords += ['many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine']
stopwords += ['more', 'moreover', 'most', 'mostly', 'move', 'much']
stopwords += ['must', 'my', 'myself', 'name', 'namely', 'neither', 'never']
stopwords += ['nevertheless', 'next', 'nine', 'no', 'nobody', 'none']
stopwords += ['noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of']
stopwords += ['off', 'often', 'on','once', 'one', 'only', 'onto', 'or']
stopwords += ['other', 'others', 'otherwise', 'our', 'ours', 'ourselves']
stopwords += ['out', 'over', 'own', 'part', 'per', 'perhaps', 'please']
stopwords += ['put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed']
stopwords += ['seeming', 'seems', 'serious', 'several', 'she', 'should']
stopwords += ['show', 'side', 'since', 'sincere', 'six', 'sixty', 'so']
stopwords += ['some', 'somehow', 'someone', 'something', 'sometime']
stopwords += ['sometimes', 'somewhere', 'still', 'such', 'system', 'take']
stopwords += ['ten', 'than', 'that', 'the', 'their', 'them', 'themselves']
stopwords += ['then', 'thence', 'there', 'thereafter', 'thereby']
stopwords += ['therefore', 'therein', 'thereupon', 'these', 'they']
stopwords += ['thick', 'thin', 'third', 'this', 'those', 'though', 'three']
stopwords += ['three', 'through', 'throughout', 'thru', 'thus', 'to']
stopwords += ['together', 'too', 'top', 'toward', 'towards', 'twelve']
stopwords += ['twenty', 'two', 'un', 'under', 'until', 'up', 'upon']
stopwords += ['us', 'very', 'via', 'was', 'we', 'well', 'were', 'what']
stopwords += ['whatever', 'when', 'whence', 'whenever', 'where']
stopwords += ['whereafter', 'whereas', 'whereby', 'wherein', 'whereupon']
stopwords += ['wherever', 'whether', 'which', 'while', 'whither', 'who']
stopwords += ['whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with']
stopwords += ['within', 'without', 'would', 'word', 'yet', 'you', 'your']
stopwords += ['yours', 'yourself', 'yourselves']
stopwords += ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
stopwords += ['30', '40', '50', '60', '70', '80','90','100','1000','10000']
stopwords += ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stopwords += ['using', 'approach', 'new', 'use', 'used', 'et', 'al', '2014', 'e']

### Define a fuction to remove stopwords from abstractwords
### Source: https://programminghistorian.org/en/lessons/counting-frequencies
def removeStopwords(wordlist, stopwords):
    return [w for w in wordlist if w not in stopwords]
abstractwords = removeStopwords(abstractwords, stopwords)


############# COUNTING THE MOST OCCURING WORDS IN THE ABSTRACTS ###############


### Create a counter for abstractwords
counts = Counter(abstractwords)
### Count the 10 most common words of abstractwords

top10 = counts.most_common(10)
top10 = ('model','language','models','task', 'data', 'paper', 'results', 'information', 'text','performance')
### Count the 25 most common words of abstractwords

top25 = counts.most_common(25)
top25 = ('model','language','models','task', 'data', 'paper', 'results', 'information', 'text','performance', 'learning',
         'translation','propose','based','neural','method','tasks', 'different', 'training','semantic','methods',
         'systems','features', 'present','work')
### Count the 100 most common words of abstractwords

top100 = counts.most_common(100)
top100 = ('model','language','models','task', 'data', 'paper', 'results', 'information', 'text','performance', 'learning',
         'translation','propose','based','neural','method','tasks', 'different', 'training','semantic','methods',
         'systems','features', 'present','work', 'languages','knowledge','dataset','words','corpus','experiments', 'machine',
          'stateoftheart','natural','analysis','evaluation','proposed','datasets','sentence','classification','representations',
          'set','problem','sentences','embeddings','approaches','novel','english','existing','generation','context','human',
          'network','research','large','representation','accuracy','parsing','domain','framework','linguistic','nlp','demonstrate',
          'sentiment','question','better','previous', 'study','syntactic','trained','entity','improve','relations','dialogue',
          'given','attention','target','input','available','automatic','detection','multiple', 'structure', 'quality','lexical',
          'shared','test','processing','extraction', 'best','introduce', 'outperforms','time','source','corpora', 'annotation',
          'provide','speech','baseline','algorithm')
 
###################### COUNTING ON TRAINING SET ###############################

### Reopen abstract and write as string
abstract = train['abstract']
abstract = abstract.astype('str') 

### Count the occurances of the top 10 list in all the abstracts of the training set
### Source: https://stackoverflow.com/questions/70243562/count-the-occurrences-of-a-wordlist-within-a-string-observation
counttop10 =[]
for i,sentence in enumerate(abstract):
    c = 0
    for word in re.findall('\w+',sentence):
        c += int(word.lower() in top10)
    counttop10 += [c]
    
### Count the occurances of the top 25 list in all the abstracts of the training set
counttop25 =[]
for i,sentence in enumerate(abstract):
    c = 0
    for word in re.findall('\w+',sentence):
        c += int(word.lower() in top25)
    counttop25 += [c]
    
### Count the occurances of the top 100 list in all the abstracts of the training set
counttop100 =[]
for i,sentence in enumerate(abstract):
    c = 0
    for word in re.findall('\w+',sentence):
        c += int(word.lower() in top100)
    counttop100 += [c]
    
### Logtransform the variables
top10log = np.log(counttop10)
top25log = np.log(counttop25)
top100log = np.log(counttop100)

### Create df for all the variables above
counts = pd.DataFrame(
    {'top10': counttop10,
     'top25': counttop25,
     'top100': counttop100,
     'logtop10': top10log,
     'logtop25': top25log,
     'logtop100': top100log
    })
### Store the variables for training set:
    
txt_train = counts

########################### COUNTING ON TEST SET ##############################

### Reopen abstracts testset and write as string

abstract = test['abstract']
abstract = abstract.astype('str') 

### Count the occurances of the top 10 list in all the abstracts of the training set
### Source: https://stackoverflow.com/questions/70243562/count-the-occurrences-of-a-wordlist-within-a-string-observation
counttop10 =[]
for i,sentence in enumerate(abstract):
    c = 0
    for word in re.findall('\w+',sentence):
        c += int(word.lower() in top10)
    counttop10 += [c]
    
### Count the occurances of the top 25 list in all the abstracts of the training set
counttop25 =[]
for i,sentence in enumerate(abstract):
    c = 0
    for word in re.findall('\w+',sentence):
        c += int(word.lower() in top25)
    counttop25 += [c]
    
### Count the occurances of the top 100 list in all the abstracts of the training set
counttop100 =[]
for i,sentence in enumerate(abstract):
    c = 0
    for word in re.findall('\w+',sentence):
        c += int(word.lower() in top100)
    counttop100 += [c]
    
### Logtransform the variables
top10log = np.log(counttop10)
top25log = np.log(counttop25)
top100log = np.log(counttop100)

### Create df for all the variables above
counts = pd.DataFrame(
    {'top10': counttop10,
     'top25': counttop25,
     'top100': counttop100,
     'logtop10': top10log,
     'logtop25': top25log,
     'logtop100': top100log
    })

### Store the variables to use in remainder of script
txt_test = counts


### IV. EXTRA FEATURES, TRAIN/TEST SPLIT. ###


## Open file with extra scholarly features. See README file for script.
extra = pd.read_csv('~/scholarly_data.csv')
extra = extra.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis = 1)


## -- XTRAIN -- ##


## Merge back the citations
xtrain['citations'] = ytrain

## Merge text feature to xtrain
xtrain['abstract_top10'] = txt_train['top10']

## Make ready to merge with scholarly features
xtrain["x"] = np.arange(len(xtrain))
xtrain = xtrain.explode('authors')
xtrain = xtrain.drop_duplicates(subset=['doi'])
## Merge
xtrain = xtrain.merge(extra, how='left', on='authors').set_index("x")

## Drop text features and encode affiliation
xtrain = xtrain.drop(['authors', 'doi'], axis = 1)

## Multiple imputation object ##
imp = IterativeImputer(max_iter=5, random_state=42)
imp.fit(xtrain[['hindex', 'publications']])
xtrain[['hindex', 'publications']] = imp.transform(xtrain[['hindex', 'publications']])

## Drop text features and encode affiliation + prestige variable
xtrain['prestige'] = xtrain['publications']/xtrain['hindex']
xtrain[xtrain['prestige'] == inf] = 0

## Fill nas for appearance and rank
imp.fit(xtrain[['appearance_y', 'rank']])
xtrain[['appearance_y', 'rank']] = imp.transform(xtrain[['appearance_y', 'rank']])

## Replace empties with zeroes in xtrain
xtrain['affiliation'] = xtrain['affiliation'].replace('', 0)      

## Encode affiliation
xtrain['affiliation'] = pd.Categorical(xtrain.affiliation)
xtrain['affiliation'] = xtrain['affiliation'].cat.codes

## Extract citations again
ytrain = xtrain['citations']
xtrain = xtrain.drop(['citations'], axis = 1)


## -- XTEST -- ##

## Reset axis
xtest = xtest.reset_index()
xtest = xtest.drop(['index'], axis = 1) 

## Merge the text feature (top 10)
xtest['abstract_top10'] = txt_test['top10']

## Make ready to merge scholarly features
xtest["x"] = np.arange(len(xtest))
xtest = xtest.explode('authors')
xtest = xtest.drop_duplicates(subset=['doi'])
## Merge
xtest = xtest.merge(extra, how='left', on='authors').set_index("x")

## Drop text features and encode affiliation
xtest = xtest.drop(['authors', 'doi'], axis = 1)

## Replace empties by zeroes
xtest['affiliation'] = xtest['affiliation'].replace('', 0)  

## Encode affiliation
xtest['affiliation'] = pd.Categorical(xtest.affiliation)
xtest['affiliation'] = xtest['affiliation'].cat.codes

## Multiple imputation for hindex and publications values ##
imp.fit(xtest[['hindex', 'publications']])
xtest[['hindex', 'publications']] = imp.transform(xtest[['hindex', 'publications']])

## Compute prestige variable
xtest['prestige'] = xtest['publications']/xtest['hindex']
xtest[xtest['prestige'] == inf] = 0

## Fill nas for appearance and rank
imp.fit(xtest[['appearance_y', 'rank']])
xtest[['appearance_y', 'rank']] = imp.transform(xtest[['appearance_y', 'rank']])

## Final step: topic rank based on hindex per topic ##
train_topic = xtrain[['topics', 'hindex']]
test_topic = xtest[['topics', 'hindex']]
## Vstack frames
tt_topics = train_topic.append(test_topic)
## Median hindex per topic
tmp =  tt_topics.groupby(['topics']).median()
## Create ranking for fos based on citations
tmp = tmp.sort_values(by = 'hindex', ascending = False)
tmp.reset_index(inplace=True)
## Give rank to the venues
tmp['topic_rank'] = range(1,len(tmp)+1)
## Ranking done, merge back to frames
df3 = xtrain.append(xtest)
## Ranking done, now merge to the training set + preserve order
df3["x"] = np.arange(len(df3))
df3 = df3.merge(tmp, how='left', on='topics').set_index("x")
## Drop extra variable and rename old one
df3 = df3.drop(['hindex_y'], axis =1)

## Split back off to xtrain and xtest
xtest = df3[len(df3)-1000:len(df3)]
xtrain = df3[0:len(df3)-1000]

## Inspect frames
xtrain.isna().sum()
xtest.isna().sum()

## Imputation for extra variables with missing values (train) 
imp.fit(xtrain[['hindex5y', 'citedby', 'citedby5y', 'cited_avg', 'i10index', 'i10index5y']])
xtrain[['hindex5y', 'citedby', 'citedby5y', 'cited_avg', 'i10index', 'i10index5y']] = imp.transform(xtrain[['hindex5y', 
                                                                                                          'citedby',
                                                                                                          'citedby5y',
                                                                                                          'cited_avg',
                                                                                                          'i10index',
                                                                                                          'i10index5y']])

## Imputation for extra variables with missing values (test)
imp.fit(xtest[['hindex5y', 'citedby', 'citedby5y', 'cited_avg', 'i10index', 'i10index5y']])
xtest[['hindex5y', 'citedby', 'citedby5y', 'cited_avg', 'i10index', 'i10index5y']] = imp.transform(xtest[['hindex5y', 
                                                                                                          'citedby',
                                                                                                          'citedby5y',
                                                                                                          'cited_avg',
                                                                                                          'i10index',
                                                                                                          'i10index5y']])

## Validate if imputation worked
xtrain.isna().sum()
xtest.isna().sum()


## Drop variables based on validation R2:
xtrain = xtrain.drop(['fos_is_cs', 'fos_rank', 'rank', 'publications'], axis = 1)
xtest = xtest.drop(['fos_is_cs', 'fos_rank', 'rank', 'publications'], axis = 1)

## Correlation plot for training features
plt.matshow(xtrain.corr())
cb = plt.colorbar()
plt.show()


### V. VALIDATION AND MODEL FITTING


## Make sure that validation/train has same proportion as test/train ratio
ttr = len(xtrain)/1000
splitpoint = ttr+1 # len(xtrain) = x + ttr*x -> len(xtrain) = 1 + ttr*x. "train should be n times validation size"
valsize = (len(xtrain)/splitpoint)/len(xtrain)

## Partial train & validation. 
partial_xtrain, xval, partial_ytrain, yval = train_test_split(
    xtrain,ytrain, test_size = valsize, random_state = 42)


############################# GRADIENT BOOSTING ###############################

## Inspired by:
## https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
## However they tune a random forest instead of a GBR.

## Create parameter grid for randomized search
learning_rate = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.3, 0.5]
subsample = [0.5, 0.6, 0.7, 0.8]
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 800, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(2, 10, num = 1)]
max_depth.append(None)
min_samples_split = [2, 4, 6, 8, 10]
min_samples_leaf = [1, 2, 4, 6]

## Create the random parameter grid for the Gradient Boosting Regressor
random_grid = {'learning_rate': learning_rate,
               'subsample': subsample,
               'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

## Instantiate GBR
gbm = GradientBoostingRegressor()

## Random search in grid
gbm_random = RandomizedSearchCV(estimator = gbm, param_distributions = random_grid, 
                               n_iter = 5, cv = 5, verbose=2, 
                               random_state=42, n_jobs = -1)

## Fit the model to partial training data
gbm_random.fit(partial_xtrain, partial_ytrain)

## Store best parameters
best_params = list(gbm_random.best_params_.values())

## Create best model with these paramaters
gbmbest = GradientBoostingRegressor(learning_rate = best_params[6],
    subsample = best_params[0], n_estimators= best_params[1],
    min_samples_split = best_params[2],
    min_samples_leaf = best_params[3],
    max_features = best_params[4],
    max_depth = best_params[5])

## Fit to partial training set
gbmbest.fit(partial_xtrain, partial_ytrain)

## Predict on unseen validation set
Y_pred_is = gbmbest.predict(xval)
Y_true = yval

## Unlog the scale for both predicted and observed citations
Y_pred_is = np.expm1(Y_pred_is)
Y_pred_is = Y_pred_is.round()
Y_true = np.expm1(Y_true)

## OOS R2 estimation
def score(Y_true, Y_pred_is):
    y_true = np.log1p(np.maximum(0, Y_true))
    y_pred = np.log1p(np.maximum(0, Y_pred_is))
    return 1 - np.mean((y_true-y_pred)**2) / np.mean((y_true-np.mean(y_true))**2)
    
score(Y_true, Y_pred_is)

## Checking relative variable importance, method and plot from:
## https://stackoverflow.com/questions/44101458/random-forest-feature-importance-chart-using-python

## Feature importance
features = list(xtrain.columns)
importances = gbmbest.feature_importances_
indices = np.argsort(importances)

## Plot
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='lightseagreen', 
         align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


######################### PREDICTIONS ON TEST SET #############################


## Model fitting
finalfit = gbmbest.fit(xtrain, ytrain)

## Predict on test set
Y_pred = finalfit.predict(xtest)
## Remove log scale from predictions
Y_pred = np.expm1(Y_pred)
Y_pred = Y_pred.astype(int)

## Bind predictions to doi's
Y_pred = pd.Series(Y_pred)
predictionfile = pd.concat([test,Y_pred], axis = 1)

## Drop unwanted columns
predictionfile = predictionfile[['doi', 0]]

## Rename
predictionfile.columns = ['doi', 'citations']

## FINAL STEP: FORMAT TO JSON ##

## Convert
result = predictionfile.to_json(orient="records")
parsed = json.loads(result)
finalpredictions = json.dumps(parsed, indent=1)  

## Export file
text_file = open("~/predicted.json", "w")
# Write string to file
text_file.write(finalpredictions)
# Close file
text_file.close()


#### --- END OF THIS SCRIPT --- ####
