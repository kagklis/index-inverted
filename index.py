import glob
import nltk
import time
import os
import re
from math import log10, sqrt
import subprocess
from bs4 import BeautifulSoup

index = {}

path = os.getcwd()
xml_path = path + "\\wikipedia\\"


try:
    os.mkdir("tokenize")
    os.mkdir("tagged")
except WindowsError:
    pass

tkn_path = path + "\\tokenize\\"
tag_path = path + "\\tagged\\"



# the list to hold the tokens
tokens_list=[]

# list of allowed word classes
open_class_cat =['JJ','JJR','JJS','NN','NNS','NP','NPS','NNP','NNPS','RB','RBR','RBS','VV','VVD','VVG','VVN','VVP','VVZ','FW']



t0 = time.time()          

i = 0
for xml in glob.glob(xml_path+"*.xml"):

    with open(xml,'r') as fh:
        s = fh.read()

    s = BeautifulSoup(s, 'html.parser')
    tokens_list = nltk.word_tokenize(s.get_text())

    tkn = "tokenized_text_%04d.txt"%i
    tag = "tagged_output_%04d.txt"%i
           
    i += 1
    with open(tkn_path+tkn,'w') as fh:
        for each_token in tokens_list:              
            if not(re.search('&',each_token)) and not(each_token.isspace()):
                fh.write(each_token.encode('utf-8'))
                fh.write("\n")                 
    
    subprocess.call('.\\TreeTagger\\bin\\tree-tagger.exe -token -lemma .\\TreeTagger\\lib\\english.par "'+tkn_path+tkn+'">"'+tag_path+tag+'"',shell=True)
    

# lemma : # texts containing the lemma
n_i = {}

N = 0
for tag_file in glob.glob(tag_path+"*.txt"):

    with open(tag_file, 'r') as fh1:
        lemmas = []
        for line in fh1.readlines():
            op = line.split()           # split line into: word POS lemma
        
            # check if POS is in the list of stop-word classes
            if ((op[1] in open_class_cat) and (op[2] != '<unknown>') and (op[2] != '@card@')and (op[2] != '@ord@')and (op[2] != '%')):
                p = re.compile('(^[\w]+[\.]$)|(^[\w]-[0-9]+)|(^[\w]-[\w])|(^[\w]-)|(-[\w]-[\w]-[\w])|([0-9]+-[0-9]+)|(^[0-9]+$)|((^[\w])([\d]$))')
                op[2] = p.sub('', op[2])
                if (op[2] != ''):
                    lemmas.append(op[2].lower())            
                     

    index[N] = {}
    for lemma in set(lemmas):
        index[N][lemma] = int(lemmas.count(lemma))
        if lemma not in n_i:
            n_i[lemma] = 0
            
        n_i[lemma] += 1
        
    N += 1

# calculate (tf_ik * idf_k)^2 for weight normalization
squares = []
for i in range(0,N):
    squares.append(0)
    for lemma,term_frequency in index[i].items():
        tf = float(term_frequency)
        if n_i[lemma] != 0:
            idf = float(log10(N/n_i[lemma]))
        else:
            idf = 0
        squares[i] += float(pow(tf*idf,2))
print(squares[0:10])

# Write index in an XML file
with open("index.xml", 'w') as fh_index:
    
    fh_index.write('<?xml version=\"1.0\" ?>\n')
    fh_index.write('<index>\n')
    
    for doc_id in index.keys():
        
        fh_index.write("\t<document id=\"%d\">\n"%doc_id)
        
        for lemma, term_frequency in index[doc_id].items():
            
            tf = float(term_frequency)
            
            #idf=log10(total documents/number of documents that contain lemma)
            if (n_i[lemma]!= 0):
                idf=float(log10(N/n_i[lemma]))
            else:
                idf = 0
            
            weight = float(float(tf*idf)/float(sqrt(squares[doc_id])))
            index[doc_id][lemma] = weight
            fh_index.write("\t\t<lemma name=\""+lemma+"\" weight=\"%f\"/>\n"%weight)
            
        fh_index.write('\t</document>\n')
        
    fh_index.write('</index>\n')

t1 = time.time()
time = (t1-t0)
minutes = time/60
sec = time%60

print ("""Index completed!!! Total time = %d min and %d sec"""%(minutes, sec))
