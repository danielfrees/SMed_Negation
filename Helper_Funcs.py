
import pandas as pd
pd.options.display.min_rows =60
import spacy

from context.reader import Annotator, load_NUBes
from context.visualizer import SentView

ann = Annotator(spacy.load('es_core_news_md'))

docs_sr = load_NUBes(ann, "/home/dfrees/projects/context/datasets/NUBes-negation-uncertainty-biomedical-corpus/NUBes")
docs_sr

def printDisoEntityPatterns(doc, separate = False):
    disoList = [ ent.text for ent in doc.ents if ent.label_ == "DISO" ]
    returnString = ""
    for diso in disoList:
        if separate == False:
            returnString += ( "{\"label\": \"DISO\", \"pattern\": \"" + diso + "\"},") 
        else: 
            returnString += ( "{\"label\": \"DISO\", \"pattern\": \"" + diso + "\"}," + "\n")
    print(returnString)
    return returnString
