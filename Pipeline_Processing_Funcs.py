#returns passed nlp object with negation terms added (can also choose to only add certain kinds)
import pandas as pd
from spacy.pipeline import EntityRuler

def addDISOAndNegationRuling_( nlp, disos, pneg = True, neg = True, rneg = True):
    ruler = EntityRuler(nlp, name='negex_ruler')
    
    neg_pats = pd.read_csv('/home/dfrees/projects/context/NegEx/negations.txt', sep='|')
    pneg = [{"label": row['label'], "pattern": row['pattern']} for index, row in neg_pats.iterrows() if row['label']=='PNEG']
    neg = [{"label": row['label'], "pattern": row['pattern']} for index, row in neg_pats.iterrows() if row['label']=='NEG']
    rneg = [{"label": row['label'], "pattern": row['pattern']} for index, row in neg_pats.iterrows() if row['label']=='RNEG']
    
    ruler.add_patterns(disos)
    
    if neg:
        ruler.add_patterns(neg)
    if pneg:
        ruler.add_patterns(pneg)
    if rneg:
        ruler.add_patterns(rneg)    
    
    #if 'entity_ruler' in nlp.pipe_names:
    #    nlp.remove_pipe('entity_ruler')
    #nlp.add_pipe('negex_ruler')
    
    return ruler


#------------------------------------------------------------------------------------------------------BELOW IS UNUSED RN


#reset all negation attributes in a doc

def removeNegations (doc):
    for sent in doc.sents:
        for ent in sent.ents:
            ent._.negated = False
    return doc


#reset all non-DISO entity labels

def removeNonDISOLabels( doc ):
    doc.ents = [ ent for ent in doc.ents if ent.label_ == "DISO" ]
    return doc




