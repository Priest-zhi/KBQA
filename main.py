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
import re

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
            # linecount+=1
            # if linecount>=100:
            #     break
    print("load word data complete!")

def LoadTokenizer():
    global tokenizer
    tokenizer = MWETokenizer(separator=' ')
    for spword in WordDict:
        if ' ' in spword:
            tupleword=tuple(spword.split(' '))
            tokenizer.add_mwe(tupleword)
        if ':' in spword:
            tupleword = tuple(re.split(r"(:)",spword))
            tokenizer.add_mwe(tupleword)


def GetKeyInfo(que):
    #tokens = tokenizer.tokenize(word_tokenize(que))
    tokens = tokenizer.tokenize(que.split(' '))
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
    mnb = MultinomialNB(alpha=1.0e-10,fit_prior=False)  # 初始化朴素贝叶斯
    mnb.fit(X_train, y_label)  # 利用训练数据对模型参数进行估计
    y_predict = mnb.predict(X_test)  # 对参数进行预测
    return y_predict

def LemmatizerDataset():    #数据集 词性还原
    LoadTokenizer()
    querylist,label=LoadQuestionData("questions/questiondata_copy.csv",columData=1)
    with open("questions/questiondata.csv",'w') as fw:
        lmtzr = WordNetLemmatizer()
        for queryEle, labelEle in zip(querylist,label):
            tokens = tokenizer.tokenize(question.split(' '))
            pos_tags = nltk.pos_tag(tokens)
            #post_tokens = [lmtzr.lemmatize(token[0],token[1][0].lower()) for token in pos_tags]
            post_tokens = [lmtzr.lemmatize(token[0]) for token in pos_tags]
            str=labelEle+','+queryEle+','+' '.join(post_tokens).replace(" 's","'s")+'\n'
            fw.write(str)


def LemmatizerQuestion(question):    #问题集 词性还原
    lmtzr = WordNetLemmatizer()
    #tokens = tokenizer.tokenize(word_tokenize(question))
    tokens = tokenizer.tokenize(question.split(' '))
    post_tokens = [lmtzr.lemmatize(token) for token in tokens]
    return ' '.join(post_tokens).replace(" 's", "'s")

def GetCQL(QuestionIndex, KeywordList):
    if QuestionIndex == 0:
        CQL="MATCH (n:Chemical) WHERE n.id='{0}' or n.name='{0}'  RETURN n".format(KeywordList[0])
    elif QuestionIndex == 1:
        CQL="MATCH (n:Chemical) WHERE n.id='{0}' or n.name='{0}' RETURN n.Definition".format(KeywordList[0])
    elif QuestionIndex == 2:
        CQL="MATCH (n:Chemical) WHERE n.id='{0}' or n.name='{0}' RETURN n.Definition".format(KeywordList[0])
    elif QuestionIndex == 3:
        CQL="MATCH (n:Chemical) WHERE n.id='{0}' or n.name='{0}' RETURN n.id".format(KeywordList[0])
    elif QuestionIndex == 4:
        CQL="MATCH (n:Chemical) WHERE n.id='{0}' or n.name='{0}' RETURN n.name".format(KeywordList[0])
    elif QuestionIndex == 5:
        CQL="MATCH (n1:Chemical)-[r:father]->(n:Chemical) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 6:
        CQL="MATCH (n1:Chemical)-[r]-(n:GO) WHERE n.id='{0}' or n.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 7:
        CQL="MATCH (n1:Chemical)-[r]-(n:GENE) WHERE n.id='{0}' or n.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 8:
        CQL = "MATCH (n1:Chemical)-[r]-(n:Pathway) WHERE n.id='{0}' or n.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 9:
        CQL="MATCH (n1:Chemical)-[r]-(n:Disease) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 10:
        CQL="MATCH (n1:Chemical)-[r]-(n:Exposure) WHERE n.id='{0}' or n.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 11:
        CQL="MATCH (n:GO) WHERE n.id='{0}' or n.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 12:
        CQL = "MATCH (n:GO) WHERE n.id='{0}' or n.name='{0}' RETURN n.Definition".format(KeywordList[0])
    elif QuestionIndex == 13:
        CQL = "MATCH (n:GO) WHERE n.id='{0}' or n.name='{0}' RETURN n.name".format(KeywordList[0])
    elif QuestionIndex == 14:
        CQL = "MATCH (n:GO) WHERE n.id='{0}' or n.name='{0}' RETURN n.id".format(KeywordList[0])
    elif QuestionIndex == 15:
        CQL="MATCH (n1:GO)-[r]-(n) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 16:
        CQL="MATCH (n1:GO)-[r:father]->(n:GO) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 17:
        CQL = "MATCH (n1:GO)-[r]-(n:GENE) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 18:
        CQL = "MATCH (n1:GO)-[r]-(n:Chemical) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 19:
        CQL="MATCH (n:GENE) WHERE n.attribute_gene_name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 20:
        CQL = "MATCH (n:GENE) WHERE n.attribute_gene_name='{0}' RETURN n.attribute_gene_id".format(KeywordList[0])
    elif QuestionIndex == 21:
        CQL = "MATCH (n:GENE) WHERE n.attribute_gene_name='{0}' RETURN n.seqname".format(KeywordList[0])
    elif QuestionIndex == 22:
        CQL = "MATCH (n:GENE) WHERE n.attribute_gene_name='{0}' RETURN n.start".format(KeywordList[0])
    elif QuestionIndex == 23:
        CQL = "MATCH (n:GENE) WHERE n.attribute_gene_name='{0}' RETURN n.end".format(KeywordList[0])
    elif QuestionIndex == 24:
        CQL = "MATCH (n:GENE) WHERE n.attribute_gene_name='{0}' RETURN n.strand".format(KeywordList[0])
    elif QuestionIndex == 25:
        CQL = "MATCH (n:GENE) WHERE n.attribute_gene_name='{0}' RETURN n.start, n.end".format(KeywordList[0])
    elif QuestionIndex == 26:
        CQL = "MATCH (n1:GENE)-[r]-(n) WHERE n1.attribute_gene_name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 27:
        CQL = "MATCH (n1:GENE)-[r]-(n:GO) WHERE n1.attribute_gene_name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 28:
        CQL = "MATCH (n1:GENE)-[r]-(n:Chemical) WHERE n1.attribute_gene_name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 29:
        CQL = "MATCH (n1:GENE)-[r]-(n:Pathway) WHERE n1.attribute_gene_name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 30:
        CQL = "MATCH (n1:GENE)-[r]-(n:HPO) WHERE n1.attribute_gene_name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 31:
        CQL = "MATCH (n1:GENE)-[r]-(n:HPO) WHERE n1.attribute_gene_name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 32:
        CQL="MATCH (n:HPO) WHERE n.id='{0}' or n.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 33:
        CQL = "MATCH (n:HPO) WHERE n.id='{0}' or n.name='{0}' RETURN n.id".format(KeywordList[0])
    elif QuestionIndex == 34:
        CQL = "MATCH (n:HPO) WHERE n.id='{0}' or n.name='{0}' RETURN n.Definition".format(KeywordList[0])
    elif QuestionIndex == 35:
        CQL = "MATCH (n:HPO) WHERE n.id='{0}' or n.name='{0}' RETURN n.name".format(KeywordList[0])
    elif QuestionIndex == 36:
        CQL = "MATCH (n1:HPO)-[r:father]->(n:HPO) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 37:
        CQL = "MATCH (n:HPO)-[r:father]->(n1:HPO) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 38:
        CQL = "MATCH (n1:HPO)-[r]-(n:GENE) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 39:
        CQL = "MATCH (n1:HPO)-[r]-(n:OMIM) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 40:
        CQL="MATCH (n:DO) WHERE n.id='{0}' or n.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 41:
        CQL = "MATCH (n:DO) WHERE n.id='{0}' or n.name='{0}' RETURN n.id".format(KeywordList[0])
    elif QuestionIndex == 42:
        CQL = "MATCH (n:DO) WHERE n.id='{0}' or n.name='{0}' RETURN n.Definition".format(KeywordList[0])
    elif QuestionIndex == 43:
        CQL = "MATCH (n:DO) WHERE n.id='{0}' or n.name='{0}' RETURN n.name".format(KeywordList[0])
    elif QuestionIndex == 44:
        CQL = "MATCH (n1:DO)-[r:father]->(n:DO) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 45:
        CQL = "MATCH (n:DO)-[r:father]->(n1:DO) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 46:
        CQL = "MATCH (n1:DO)-[r]-(n:Disease) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 47:
        CQL = "MATCH (n1:DO)-[r]-(n:OMIM) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 48:
        CQL="MATCH (n:OMIM) WHERE n.id='{0}' or n.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 49:
        CQL = "MATCH (n:OMIM) WHERE n.id='{0}' or n.name='{0}' RETURN n.id".format(KeywordList[0])
    elif QuestionIndex == 50:
        CQL = "MATCH (n:OMIM) WHERE n.id='{0}' or n.name='{0}' RETURN n.name".format(KeywordList[0])
    elif QuestionIndex == 51:
        CQL = "MATCH (n1:OMIM)-[r]-(n:Disease) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 52:
        CQL = "MATCH (n1:OMIM)-[r]-(n:HPO) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 53:
        CQL = "MATCH (n1:OMIM)-[r]-(n:DO) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 54:
        CQL = "MATCH (n1:OMIM)-[r]-(n:DO) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 55:
        CQL = "MATCH (n:Exposure) WHERE ANY (ex in n.exposurestressors where ex =~ '(?i).*{0}.*') RETURN n".format(KeywordList[0])
    elif QuestionIndex == 56:
        CQL = "MATCH (n:Exposure) WHERE ANY (ex in n.exposurestressors where ex =~ '(?i).*{0}.*') RETURN n.exposurestressors".format(KeywordList[0])
    elif QuestionIndex == 57:
        CQL = "MATCH (n:Exposure) WHERE ANY (ex in n.exposurestressors where ex =~ '(?i).*{0}.*') RETURN n.exposurestressors".format(KeywordList[0])
    elif QuestionIndex == 58:
        CQL = "MATCH (n1:Exposure)-[r]-(n:Disease) WHERE ANY (ex in n1.exposurestressors where ex =~ '(?i).*{0}.*') RETURN n".format(KeywordList[0])
    elif QuestionIndex == 59:
        CQL = "MATCH (n1:Exposure)-[r]-(n:Chemical) WHERE ANY (ex in n1.exposurestressors where ex =~ '(?i).*{0}.*') RETURN n".format(KeywordList[0])
    elif QuestionIndex == 60:
        CQL = "MATCH (n:Disease) WHERE n.id='{0}' or n.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 61:
        CQL = "MATCH (n:Disease) WHERE n.id='{0}' or n.name='{0}' RETURN n.id".format(KeywordList[0])
    elif QuestionIndex == 62:
        CQL = "MATCH (n:Disease) WHERE n.id='{0}' or n.name='{0}' RETURN n.name".format(KeywordList[0])
    elif QuestionIndex == 63:
        CQL = "MATCH (n1:Disease)-[r]-(n:Chemical) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 64:
        CQL = "MATCH (n1:Disease)-[r]-(n:Pathway) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 65:
        CQL = "MATCH (n1:Disease)-[r]-(n:DO) WHERE n.id='{0}' or n.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 66:
        CQL = "MATCH (n1:Disease)-[r]-(n:DO) WHERE n.id='{0}' or n.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 67:
        CQL = "MATCH (n1:Disease)-[r]-(n:OMIM) WHERE n.id='{0}' or n.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 68:
        CQL = "MATCH (n1:Disease)-[r]-(n:Exposure) WHERE n.id='{0}' or n.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 69:
        CQL = "MATCH (n1:Disease)-[r]-(n:DO) WHERE n.id='{0}' or n.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 70:
        CQL = "MATCH (n:Pathway) WHERE n.id='{0}' or n.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 71:
        CQL = "MATCH (n:Pathway) WHERE n.id='{0}' or n.name='{0}' RETURN n.id".format(KeywordList[0])
    elif QuestionIndex == 72:
        CQL = "MATCH (n:Pathway) WHERE n.id='{0}' or n.name='{0}' RETURN n.name".format(KeywordList[0])
    elif QuestionIndex == 73:
        CQL = "MATCH (n1:Pathway)-[r]-(n:Chemical) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 74:
        CQL = "MATCH (n1:Pathway)-[r]-(n:GENE) WHERE n1.id='{0}' or n1.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 75:
        CQL = "MATCH (n1:Pathway)-[r]-(n:Disease) WHERE n.id='{0}' or n.name='{0}' RETURN n".format(KeywordList[0])
    else:
        CQL="error in question index"
        print("error in question index")
    return CQL

if __name__ == '__main__':
    que = "what is REACT:R-HSA-2485179"
    que2='the definition of MESH:C481454'
    que3="SNAP 5540's hazard"
    LoadWordData()
    LoadTokenizer()
    question=LemmatizerQuestion(que)
    tokens, keyList = GetKeyInfo(question)
    QuestionIndex=PredictQuestion(tokens)
    print(QuestionIndex)
    CQL = GetCQL(int(QuestionIndex[0]),keyList)
    print(CQL)
    print("All done!")

