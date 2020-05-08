import csv
from py2neo import Graph, Node, Relationship

def CreateMyword():
    Dgraph = Graph("http://localhost:7474")
    Nodes={"Chemical":"nch","DO":"ndo","Disease":"ndi","Exposure":"nex","GENE":"nge","GO":"ngo","HPO":"nhp","OMIM":"nom","Pathway":"npa"}
    nodecount=0
    with open('questions/myWord.txt','w',encoding='utf-8') as fw:
        for node in Nodes:
            resultTerms = Dgraph.run('MATCH (n:'+node+') RETURN n').data()
            for termDict in resultTerms:
                term = termDict["n"]
                if "name" in term:
                    str=term["name"]+'\t'+Nodes[node]+'\n'
                    fw.write(str)
                if "id" in term:
                    str=term["id"]+'\t'+Nodes[node]+'\n'
                    fw.write(str)
                if "attribute_gene_name" in term: #gene
                    str=term["attribute_gene_name"]+'\t'+Nodes[node]+'\n'
                    fw.write(str)
                if "exposurestressors" in term: #exposure
                    for eleStre in term["exposurestressors"]:
                        sigStre = eleStre.split('^')
                        fw.write(sigStre[0]+'\t'+Nodes[node]+'\n')
                        fw.write(sigStre[2]+":"+sigStre[1] + '\t' + Nodes[node]+'\n')    #MESH:xxx

                nodecount+=1
                if nodecount%5000==0:
                    print(nodecount)


def AddNcRNA():
    graph = Graph("http://localhost:7474")
    with open('questions/myWord.txt', 'a', encoding='utf-8') as fw:
        resultTerms = graph.run('MATCH (n:ncRNA) RETURN n').data()
        for termDict in resultTerms:
            term = termDict["n"]
            if term and "Symbol" in term:
                str = "RNA-"+term["Symbol"] + '\t' + "nnr" + '\n'
                fw.write(str)

if __name__ == '__main__':
    AddNcRNA()