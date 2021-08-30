import spacy
from spacy import Language

@Language.factory('negex')
class DNegEx:
    
    m_negation = True
    m_pseudonegation = True
    m_reversenegation = True
    m_negationRange = 5
    
    def __init__ (self, nlp=None, name=None, negation_range = 5, negations = True, pseudonegations = True, reverseNegations = True):
        m_negations = negations
        m_pseudonegations = pseudonegations
        m_reversenegations = reverseNegations
        m_negationRange = negation_range
         
    def __call__ (self, doc, negation_range = m_negationRange):
        for sentence in doc.sents:
            for switchEnt in sentence.ents:
                if self.m_negation and switchEnt.label_ == "NEG":
                    range = self.m_negationRange
                    for tok in doc[switchEnt.end:len(doc)]:
                        if range <= 0:
                            break
                            
                        if tok.is_punct:  #don't count punctuation against range
                            continue
                        
                        if tok.ent_iob_ == "" or tok.ent_iob_ == "O":
                            range -= 1
                            continue
                            
                        if tok.ent_iob_ == "B":
                            range -=1
                            for ent in doc.ents:
                                if ent.start == tok.i:
                                    ent._.negated = True
                        
                        if tok.ent_iob_ == "I":
                            continue
                            
                if self.m_negation and switchEnt.label_ == "RNEG":
                    range = self.m_negationRange
                    for tok in reversed(doc[0:switchEnt.start-1]):
                        if range <= 0:
                            break
                            
                        if tok.is_punct:  #don't count punctuation against range
                            continue
                        
                        if tok.ent_iob_ == "" or tok.ent_iob_ == "O":
                            range -=1
                            continue
                            
                        if tok.ent_iob_ == "B":
                            range -= 1
                            for ent in doc.ents:
                                if ent.start == tok.i:
                                    ent._.negated = True
                        
                        if tok.ent_iob_ == "I":
                            continue
               
        return doc
        
            
    def turnOffNegation(self):
        self.m_negation = False
    def turnOffPseudonegation(self):
        self.m_pseudonegation = False
    def turnOffReverseNegation(self):
        self.m_reversenegation = False
    def turnOnNegation(self):
        self.m_negation = True
    def turnOnPseudonegation(self):
        self.m_pseudonegation = True
    def turnOnReverseNegation(self):
        self.m_reversenegation = True
    def setNegationRange(self, negation_range):
        self.m_negationRange = negation_range