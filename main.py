import csv
from py2neo import Graph, Node, Relationship
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import MWETokenizer
from nltk.corpus import PlaintextCorpusReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

WordDict={}
tokenizer=0

def LoadWordData(file="questions/myWord.txt"):
    linecount=0
    with open(file, mode='r',encoding='utf-8') as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            line_list = line.strip('\n').split('\t')
            WordDict[line_list[0]]=line_list[1]
            linecount+=1
            if linecount>=100:
                break
    print("load word data complete!")

def LoadTokenizer():
    global tokenizer
    tokenizer = MWETokenizer(separator=' ')
    for spword in WordDict:
        if ' ' in spword:
            tupleword=tuple(spword.split(' '))
            tokenizer.add_mwe(tupleword)

def GetKeyInfo(que):
    tokens = tokenizer.tokenize(word_tokenize(que))
    pos_tags = nltk.pos_tag(tokens)
    print(pos_tags)
    listKey=[]
    #replace own tag
    for SigWord in pos_tags:
        if SigWord[1].startswith('NN') or SigWord[1]=='FW' or SigWord[1].startswith('JJ'):
            if SigWord[0] in WordDict:
                Windex=pos_tags.index(SigWord)
                tokens[Windex]=WordDict[SigWord[0]]
                listKey.append(SigWord[0])
    return tokens,listKey


def LoadQuestionData(file="questions/questiondata.csv",columData=2,labelData=0):
    QueDataList=[]
    QueClassList=[]
    with open(file, mode='r',encoding='utf-8') as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            if not line.startswith('#'):
                line_list = line.strip('\n').split(',')
                QueDataList.append(line_list[columData])    #use original word
                QueClassList.append(line_list[labelData])
    return QueDataList,QueClassList


def PredictQuestion(tokens):
    x_ques,y_label=LoadQuestionData()
    vec = CountVectorizer()
    X_train = vec.fit_transform(x_ques)
    oneSeq = ' '.join(tokens).replace(" 's","'s")
    X_test = vec.transform([oneSeq])
    mnb = MultinomialNB(fit_prior=False)  # 初始化朴素贝叶斯
    mnb.fit(X_train, y_label)  # 利用训练数据对模型参数进行估计
    y_predict = mnb.predict(X_test)  # 对参数进行预测
    return y_predict

def LemmatizerDataset():    #数据集 词性还原
    LoadTokenizer()
    querylist,label=LoadQuestionData("questions/questiondata_copy.csv",columData=1)
    with open("questions/questiondata.csv",'w') as fw:
        lmtzr = WordNetLemmatizer()
        for queryEle, labelEle in zip(querylist,label):
            tokens = tokenizer.tokenize(word_tokenize(queryEle))
            pos_tags = nltk.pos_tag(tokens)
            #post_tokens = [lmtzr.lemmatize(token[0],token[1][0].lower()) for token in pos_tags]
            post_tokens = [lmtzr.lemmatize(token[0]) for token in pos_tags]
            str=labelEle+','+queryEle+','+' '.join(post_tokens).replace(" 's","'s")+'\n'
            fw.write(str)


def LemmatizerQuestion(question):    #问题集 词性还原
    lmtzr = WordNetLemmatizer()
    tokens = tokenizer.tokenize(word_tokenize(question))
    post_tokens = [lmtzr.lemmatize(token) for token in tokens]
    return ' '.join(post_tokens).replace(" 's", "'s")

def GetCQL(QuestionIndex, KeywordList):
    if QuestionIndex == 0:
        CQL="MATCH (n:Chemical) WHERE n.id='{0}' or n.name='{0}'  RETURN n".format(KeywordList[0])
    else:
        CQL="error in question index"
        print("error in question index")
    return CQL

if __name__ == '__main__':
    que = "what's SNAP 5540"
    que2='hazard to SNAP 5540'
    que3="SNAP 5540's hazard"
    LoadWordData()
    LoadTokenizer()
    question=LemmatizerQuestion(que)
    tokens, keyList = GetKeyInfo(question)
    QuestionIndex=PredictQuestion(tokens)
    CQL = GetCQL(int(QuestionIndex[0]),keyList)
    print(CQL)
    print("All done!")

