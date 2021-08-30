import pandas as pd
import spacy
from NegEx.Pipeline_Processing_Funcs import addDISOAndNegationRuling_
from copy import copy #for negNlp_
from NegEx.NegEx import DNegEx #for negNlp_
from pandas import concat #for humanVersusNeg

nlp = spacy.load("es_core_news_md")

#sets up NegEx, ruler, and strips all but DISOs in the passed doc. Then performs NegEx on stripped doc.    
def negNlp_(doc):
    negex = DNegEx()   #can modify exact desired negation settings at this step
    
    #set up ruler to find the PNEG, NEG, RNEG terms and DISOS
    #note: passing empty list of disos because disos will already be in the doc the way we're processing it
    ruler = addDISOAndNegationRuling_(nlp, [])
    
    #keep DISOS in doc (only DISO entities). If no DISOS, done. Reset negation status
    if 'DISO' not in [e.label_ for e in doc.ents]: return doc
    doc_copy = copy(doc)
    doc_copy.ents = [ent for ent in doc_copy.ents if ent.label_=='DISO']
    for ent in doc_copy.ents:
        ent._.negated = False
    doc_copy = negex(ruler(doc_copy))
    return doc_copy

#performs a comparison of human annotation versus NegEx for an annotated database
def humanVersusNeg(docSeries): 
    df = pd.concat([compareDocs(doc, negNlp_(doc)) for doc in docSeries])
    return df


#takes in a dataframe of human versus NegEx comparison (ie: return val from humanVersusNeg) and returns a dataframe of the agreement b/t the two (like a confusion matrix)
def negPerformance(df):
    agreementSeries = (df["D1 Negation Status"].astype(str) + " | "  + df["D2 DISO Negation Status"].astype(str)).value_counts()
    agreementSeries = agreementSeries.rename({"1.0 | 1.0": "TP", "1.0 | 0.0": "FN", "0.0 | 0.0": "TN", "0.0 | 1.0": "FP"})
    precision = agreementSeries["TP"] / agreementSeries[["TP", "FP"]].sum()
    print("Precision: " + str(precision))
    recall = agreementSeries["TP"] / agreementSeries[["TP", "FN"]].sum()
    print("Recall: " + str(recall))
    
    return pd.DataFrame(agreementSeries)


#TODO: Update for REVERSE NEG
#takes in a doc, and an ent in that doc
#finds the first negation word affecting the current DISO, if one exists. 
#If there is no negation word, this will be indicated by 'N/A'
def getNegationSwitch(doc, ent):
    if ent._.negated == False:
        return
    
    switchWord = ""
    negationWord = False
    if ent.start >=5:
        for ent in doc[ent.start - 5: ent.start].ents:
            if negationWord:   #only append one negation word per diso
                break
            if ent.label_.upper().startswith("NEG"):
                negationWord = True
                switchWord = ent.text
    elif ent.start < 5:
        for ent in doc[0: ent.start].ents:
            if negationWord:
                break
            if ent.label_.upper().startswith("NEG"):
                negationWord = True
                switchWord = ent.text
                
    if negationWord:
        return switchWord
    else:
        return "N/A"

    
#TODO: Remove illogical D1 Negation Switch (human negation switch can't be determined)
#requires pandas import to run
#must be passed spacy docs with ._.negated attributes and labeled 'NEG' and 'DISOS' ent.label_ 's
def compareDocs(doc1, doc2):
    
    docOneDisos = []
    docOneDisoPositions = []
    docOneDisoStatuses = []
    docOneSwitchWords = []
    
    docTwoDisos = []
    docTwoDisoPositions = []
    docTwoDisoStatuses = []
    docTwoSwitchWords = []
    
    for sentence in doc1.sents:
        for ent in sentence.ents:
            if ent.label_ == "DISO":
                docOneDisos.append(ent.text)
                docOneDisoPositions.append(ent.start)
                docOneDisoStatuses.append(ent._.negated)
                docOneSwitchWords.append( getNegationSwitch (doc1, ent) )
                
    for sentence in doc2.sents:
        for ent in sentence.ents:
            if ent.label_ == "DISO":
                docTwoDisos.append(ent.text)
                docTwoDisoPositions.append(ent.start)
                docTwoDisoStatuses.append(ent._.negated)
                docTwoSwitchWords.append( getNegationSwitch (doc2, ent) )            
           
    
    dat = {'Doc1 DISOS': docOneDisos, 'D1 Position in Doc': docOneDisoPositions, 'D1 Negation Status': docOneDisoStatuses,
           'D1 Negation Switch (if any)': docOneSwitchWords, 'Doc2 DISOS': docTwoDisos, 
           'D2 Position in Doc': docTwoDisoPositions, 'D2 DISO Negation Status': docTwoDisoStatuses, 
           'D2 Negation Switch (if any)': docTwoSwitchWords }
    #print(dat)
    df = pd.DataFrame(data=dat)
    #print(df)
    return df