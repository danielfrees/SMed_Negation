#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
print(sys.path)
#sys.path.append("/home/dfrees/projects/context/")


# In[6]:


import pandas as pd
pd.options.display.min_rows =60
import spacy

from context.reader import Annotator, load_NUBes
from context.visualizer import SentView


# In[11]:


ann = Annotator(spacy.load('es_core_news_md'))


# In[14]:


docs_sr = load_NUBes(ann, "/home/dfrees/projects/context/datasets/NUBes-negation-uncertainty-biomedical-corpus/NUBes")
docs_sr


# In[42]:


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


# In[43]:


printDisoEntityPatterns( docs_sr['005']['neuro']['preil'][5] )


# In[44]:


printDisoEntityPatterns( docs_sr['005']['neuro']['preil'][5], separate = True )


# In[ ]:




