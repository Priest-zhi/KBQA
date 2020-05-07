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
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import Levenshtein
import gensim
import os


WordDict={}
tokenizer=0

def LoadWordData(file=os.path.dirname(os.path.realpath(__file__))+"/questions/myWord.txt"):
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
    IsFindKeyWord=False
    #replace own tag
    for SigWord in pos_tags:
        if SigWord[1].startswith('NN') or SigWord[1]=='FW' or SigWord[1].startswith('JJ'):
            if SigWord[0] in WordDict:
                IsFindKeyWord=True
                Windex=pos_tags.index(SigWord)
                tokens[Windex]=WordDict[SigWord[0]]
                listKey.append(SigWord[0])
    return tokens,listKey

def HasSimilaryChar(lstr,rstr):
    if '"' in lstr:
        return '"' in rstr
    elif '-' in lstr and ':' in lstr:
        return '-' in rstr and ':' in rstr
    elif '-' in lstr:
        return '-' in rstr
    elif ':' in lstr:
        return ':' in rstr
    return True

def FindSimilaryWord(que,topk=3):
    count=0
    MaxSimilary=0
    tokens=[]
    listKey=[]
    SimilaryWord=""
    tmpWordict=[]

    sigtopk={}
    for myWord in WordDict:
        if HasSimilaryChar(myWord, que):
            TmpSimilary=WordSimilary(que, myWord)
            if TmpSimilary>MaxSimilary:
                MaxSimilary=TmpSimilary
                SimilaryWord=myWord
        count+=1
        if count%100==0:
            if SimilaryWord not in sigtopk.keys():
                sigtopk[SimilaryWord]=MaxSimilary
                print(count, ' word: ', SimilaryWord, ' similary: ', MaxSimilary)
    listtmp = sorted(sigtopk.items(), key=lambda x: x[1], reverse=True)[:topk]
    for ele in listtmp:
        tokens.append([WordDict[ele[0]]])
        listKey.append([ele[0]])
    return tokens,listKey


def LoadQuestionData(file=os.path.dirname(os.path.realpath(__file__))+"/questions/questiondata.csv",columData=2,labelData=0):
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
    querylist,label=LoadQuestionData(os.path.dirname(os.path.realpath(__file__))+"/questions/questiondata_copy.csv",columData=1)
    with open(os.path.dirname(os.path.realpath(__file__))+"/questions/questiondata.csv",'w') as fw:
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
        CQL="MATCH (n:Chemical) WHERE n.id='{0}' or n.name='{0}' RETURN n,n.Definition".format(KeywordList[0])
    elif QuestionIndex == 2:
        CQL="MATCH (n:Chemical) WHERE n.id='{0}' or n.name='{0}' RETURN n,n.Definition".format(KeywordList[0])
    elif QuestionIndex == 3:
        CQL="MATCH (n:Chemical) WHERE n.id='{0}' or n.name='{0}' RETURN n,n.id".format(KeywordList[0])
    elif QuestionIndex == 4:
        CQL="MATCH (n:Chemical) WHERE n.id='{0}' or n.name='{0}' RETURN n,n.name".format(KeywordList[0])
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
        CQL = "MATCH (n:GO) WHERE n.id='{0}' or n.name='{0}' RETURN n,n.Definition".format(KeywordList[0])
    elif QuestionIndex == 13:
        CQL = "MATCH (n:GO) WHERE n.id='{0}' or n.name='{0}' RETURN n,n.name".format(KeywordList[0])
    elif QuestionIndex == 14:
        CQL = "MATCH (n:GO) WHERE n.id='{0}' or n.name='{0}' RETURN n,n.id".format(KeywordList[0])
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
        CQL = "MATCH (n:GENE) WHERE n.attribute_gene_name='{0}' RETURN n,n.attribute_gene_id".format(KeywordList[0])
    elif QuestionIndex == 21:
        CQL = "MATCH (n:GENE) WHERE n.attribute_gene_name='{0}' RETURN n,n.seqname".format(KeywordList[0])
    elif QuestionIndex == 22:
        CQL = "MATCH (n:GENE) WHERE n.attribute_gene_name='{0}' RETURN n,n.start".format(KeywordList[0])
    elif QuestionIndex == 23:
        CQL = "MATCH (n:GENE) WHERE n.attribute_gene_name='{0}' RETURN n,n.end".format(KeywordList[0])
    elif QuestionIndex == 24:
        CQL = "MATCH (n:GENE) WHERE n.attribute_gene_name='{0}' RETURN n,n.strand".format(KeywordList[0])
    elif QuestionIndex == 25:
        CQL = "MATCH (n:GENE) WHERE n.attribute_gene_name='{0}' RETURN n,n.start, n.end".format(KeywordList[0])
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
        CQL = "MATCH (n:HPO) WHERE n.id='{0}' or n.name='{0}' RETURN n,n.id".format(KeywordList[0])
    elif QuestionIndex == 34:
        CQL = "MATCH (n:HPO) WHERE n.id='{0}' or n.name='{0}' RETURN n,n.Definition".format(KeywordList[0])
    elif QuestionIndex == 35:
        CQL = "MATCH (n:HPO) WHERE n.id='{0}' or n.name='{0}' RETURN n,n.name".format(KeywordList[0])
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
        CQL = "MATCH (n:DO) WHERE n.id='{0}' or n.name='{0}' RETURN n,n.id".format(KeywordList[0])
    elif QuestionIndex == 42:
        CQL = "MATCH (n:DO) WHERE n.id='{0}' or n.name='{0}' RETURN n,n.Definition".format(KeywordList[0])
    elif QuestionIndex == 43:
        CQL = "MATCH (n:DO) WHERE n.id='{0}' or n.name='{0}' RETURN n,n.name".format(KeywordList[0])
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
        CQL = "MATCH (n:OMIM) WHERE n.id='{0}' or n.name='{0}' RETURN n,n.id".format(KeywordList[0])
    elif QuestionIndex == 50:
        CQL = "MATCH (n:OMIM) WHERE n.id='{0}' or n.name='{0}' RETURN n,n.name".format(KeywordList[0])
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
        CQL = "MATCH (n:Exposure) WHERE ANY (ex in n.exposurestressors where ex =~ '(?i).*{0}.*') RETURN n,n.exposurestressors".format(KeywordList[0])
    elif QuestionIndex == 57:
        CQL = "MATCH (n:Exposure) WHERE ANY (ex in n.exposurestressors where ex =~ '(?i).*{0}.*') RETURN n,n.exposurestressors".format(KeywordList[0])
    elif QuestionIndex == 58:
        CQL = "MATCH (n1:Exposure)-[r]-(n:Disease) WHERE ANY (ex in n1.exposurestressors where ex =~ '(?i).*{0}.*') RETURN n".format(KeywordList[0])
    elif QuestionIndex == 59:
        CQL = "MATCH (n1:Exposure)-[r]-(n:Chemical) WHERE ANY (ex in n1.exposurestressors where ex =~ '(?i).*{0}.*') RETURN n".format(KeywordList[0])
    elif QuestionIndex == 60:
        CQL = "MATCH (n:Disease) WHERE n.id='{0}' or n.name='{0}' RETURN n".format(KeywordList[0])
    elif QuestionIndex == 61:
        CQL = "MATCH (n:Disease) WHERE n.id='{0}' or n.name='{0}' RETURN n,n.id".format(KeywordList[0])
    elif QuestionIndex == 62:
        CQL = "MATCH (n:Disease) WHERE n.id='{0}' or n.name='{0}' RETURN n,n.name".format(KeywordList[0])
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
        CQL = "MATCH (n:Pathway) WHERE n.id='{0}' or n.name='{0}' RETURN n,n.id".format(KeywordList[0])
    elif QuestionIndex == 72:
        CQL = "MATCH (n:Pathway) WHERE n.id='{0}' or n.name='{0}' RETURN n,n.name".format(KeywordList[0])
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

semcor_ic = wordnet_ic.ic('ic-semcor.dat')
lastword=""
lastWn=""
def WordSimilary(Lstr,Rstr,alpha=0.5,deepSearch=True):
    # Lstr=Lstr.strip()
    # Rstr=Rstr.strip()
    global lastword,lastWn
    if lastword==Lstr:
        WnLstr=lastWn
    elif ' ' in Lstr:
        LstrNoSP=Lstr.replace(' ','_')
        WnLstr = wn.synsets(LstrNoSP)
        lastWn=WnLstr
        lastword=LstrNoSP
    else:
        WnLstr = wn.synsets(Lstr)
        lastWn=WnLstr
        lastword=Lstr
    if ' ' in Rstr:
        RstrNoSP=Rstr.replace(' ','_')
        WnRstr = wn.synsets(RstrNoSP)
    else:
        WnRstr = wn.synsets(Rstr)

    LinSimilary=LDSimilary=0
    if WnLstr and WnRstr and WnLstr[0].name().split('.')[1]==WnRstr[0].name().split('.')[1]:
        LinSimilary=WnLstr[0].lin_similarity(WnRstr[0], semcor_ic)
    elif (' ' in Lstr or ' ' in Rstr) and deepSearch:
        totalSimilary = 0
        for lword in Lstr.split(' '):
            lmaxSimilary=0
            for rword in Rstr.split(' '):
                wntmpl=wn.synsets(lword)
                wntmpr = wn.synsets(rword)
                if wntmpl and wntmpr and wntmpl[0].name().split('.')[1]==wntmpr[0].name().split('.')[1]:
                    lmaxSimilary=max(lmaxSimilary,wntmpl[0].lin_similarity(wntmpr[0], semcor_ic))
            totalSimilary+=lmaxSimilary
        LinSimilary = totalSimilary / max(len(Lstr.split(' ')),len(Rstr.split(' ')))
    if alpha!=1:
        LDSimilary=Levenshtein.jaro(Lstr.lower(), Rstr.lower())
    return LinSimilary*alpha + LDSimilary*(1-alpha)

#not used
def WordVecSimilary():
    corpus = nltk.corpus.brown.sents()

    fname = 'word2vec/brown/brown_skipgram.model'
    if os.path.exists(fname):
        # load the file if it has already been trained, to save repeating the slow training step below
        model = gensim.models.Word2Vec.load(fname)
    else:
        # can take a few minutes, grab a cuppa
        model = gensim.models.Word2Vec(corpus, size=100, min_count=1, workers=2, iter=50)
        model.save(fname)
    #print(model.similarity("air Pollutant", "air Pollutant pollution"))

    # fname = 'word2vec/Wikipedia/wikipedia.model1'
    # fname2='word2vec/Biomedical/wikipedia-pubmed-and-PMC-w2v.model'
    # if os.path.exists(fname):
    #     # load the file if it has already been trained, to save repeating the slow training step below
    #     model = gensim.models.Word2Vec.load(fname)
    # else:
    #     # can take a few minutes, grab a cuppa
    #     model=gensim.models.KeyedVectors.load_word2vec_format('word2vec/Wikipedia/model.bin',binary=True)
    #     model.save(fname)
    #print(model.similarity("air Pollutant", "air pollution"))
    print(model.n_similarity(["air","Pollutant"],["air","pollution"]))

def DropStopword(question):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in question.split(' ') if not w in stop_words]
    return ' '.join(filtered_sentence)

def init():
    global WordDict,tokenizer,semcor_ic,lastword,lastWn
    WordDict = {}
    tokenizer = 0
    semcor_ic = wordnet_ic.ic('ic-semcor.dat')
    lastword = ""
    lastWn = ""
    LoadWordData()
    LoadTokenizer()

def GetCQLfromQuestion(sentence):
    NSWquestion=DropStopword(sentence)
    question=LemmatizerQuestion(NSWquestion)
    tokens, keyList = GetKeyInfo(question)  #tokens=["what","is","nxx"] keylist=["yyy"]
    CQLlist=[]
    if keyList:
        QuestionIndex = PredictQuestion(tokens)
        print(QuestionIndex)
        CQL = GetCQL(int(QuestionIndex[0]), keyList)
        CQLlist.append(CQL)
        print(CQL)
    else:
        # no key word, find similary word
        tokens, listKey = FindSimilaryWord(question) #tokens=[["what","is","nxx"][...][...]] keylist=[["yyy"][...][...]]
        for token,key in zip(tokens,listKey):
            QuestionIndex=PredictQuestion(token)
            print(QuestionIndex)
            CQL = GetCQL(int(QuestionIndex[0]),key)
            CQLlist.append(CQL)
            print(CQL)
    return CQLlist
    print("All done!")

if __name__ == '__main__':
    # print(WordSimilary("Acid Rain",'air pollution'))
    # print(WordSimilary("air Pollutants", 'air pollution'))
    # exit()
    #WordSimilary("air Contamination","air pollution")
    que = "what is REACT:R-HSA-2485179"
    que2='the definition of MESH:C481454'
    que3="air pollution"
    que4='what is aluminium'
    LoadWordData()
    LoadTokenizer()
    NSWquestion=DropStopword(que2)
    question=LemmatizerQuestion(NSWquestion)
    tokens, keyList = GetKeyInfo(question)  #tokens=["what","is","nxx"] keylist=["yyy"]
    if keyList:
        QuestionIndex = PredictQuestion(tokens)
        print(QuestionIndex)
        CQL = GetCQL(int(QuestionIndex[0]), keyList)
        print(CQL)
    else:
        # no key word, find similary word
        tokens, listKey = FindSimilaryWord(question) #tokens=[["what","is","nxx"][...][...]] keylist=[["yyy"][...][...]]
        for token,key in zip(tokens,listKey):
            QuestionIndex=PredictQuestion(token)
            print(QuestionIndex)
            CQL = GetCQL(int(QuestionIndex[0]),key)
            print(CQL)
    print("All done!")

