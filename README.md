# SMed_Negation
Natural Language Processing of Medical Texts (Esp. in Spanish)

Goal: To identify negation status of salient entities in a sentence (in particular, disorder entities from medical texts). ie: A doctor mentions something about schizophrenia, but are they saying their patient has it? Or they don't? Or they have something else and schizophrenia is now ruled out? Knowing the negation status (negated vs. affirmed) of a disorder is very important information in parsing medical records. 

I coded these scripts as an Undergraduate Research Assistant for UCLA Paisa Project. Most scripts are intended to work with spaCy docs.

Although this is not a complete project (some files were not my intellectual property, and are on a secure server to which I no longer have access), I hope that some of my scripts might prove useful or inspirational to others working on negation identification, particularly in the context of electronic health records. My notebooks, results, etc. are not in this github repo as they rely on numerous other scripts that aren't my intellectual property.

## _Quick overview of my scripts:_

_In Helper_Funcs.py_ you will find a function which can extract disorder (DISO) patterns from medical texts that have been labeled via spaCy's entity recognition, or from annotated texts (I used [NUBes Corpus Spanish Medical Texts ](https://github.com/Vicomtech/NUBes-negation-uncertainty-biomedical-corpus/tree/master/NUBes)for all of my work and this worked for those).

_In Negex.py_ is my customizable callable which identifies negation status of disorders based on their relative location to switch terms, such as negations, pseudonegations, and reverse negations. Negation range and allowed negation types can be modified before calling a DNegEx object to achieve specific negation behavior. _Idea for future research: Systematically modifying negation ranges and allowed negation types to see which settings yield the highest specificity and sensitivity of negation identification in medical texts._

_In Pipeline_Processing_Funcs.py_ are some functions intended for dealing with entities when testing human-annotated sentences against the negation algorithm. In here are functions for cleaning out entities from spaCy docs, stripping non-DISO entities from spaCy docs, or loading the spaCy ruler with negation, pseudonegation, and reverse negation entities so that DNegEx() callables can run properly. This ruler should be used on whatever doc you want to run through the DNegEx() callable so that switch terms are identified correctly. Unfortunately, spaCy has changed how rulers work since I finished my work on this project, so the addDISOAndNegationRuling_ function may be outdated.

_In Testing_Funcs.py_ are functions designed to test DNegEx() performance versus annotation, and output statistics on how successful the algorithm was. In here is also code which can be used to produce a detailed breakdown of identified disorders, negation status, negation switch words, etc. 



