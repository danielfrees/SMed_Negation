import os
import re
import pandas as pd

import spacy
from spacy import displacy
from spacy.tokens import Span, Token, Doc

def ann_all_docs( ann, dir_loc ):
    docs_list = []
    for (dirpath, dirnames, filenames) in os.walk(dir_loc):
        for filename in filenames:
            if filename.startswith('sample') and filename.endswith('.ann'):
                print('loading: ',filename[:-4],'...')
                j=0
                for doc in ann( os.path.join(dirpath, filename), annotation="brat", return_all=False ):
                    docs_list.append([filename,j,doc])
                    j+=1
    return pd.DataFrame( docs_list, columns=['file','i','doc'])

def load_NUBes( ann, dir_loc):
    docs_df = ann_all_docs(ann, dir_loc)
    docs_df = pd.concat([pd.DataFrame(docs_df['file'].str.split('.').tolist(),
                                     columns=['sample','speciality','section','ext']),
                         docs_df[['i','doc']]],1).drop(columns="ext")
    docs_df['sample'] = docs_df['sample'].str.replace('sample-','')
    return docs_df.set_index(['sample','speciality','section','i'])['doc']

class Annotator:
    def __init__(self, nlp=None):
        self.nlp = nlp if nlp is not None else spacy.load("es_core_news_md")            
        
        Span.set_extension("ent_id"   , default=  ''   , force=True)
        # for modifiers
        Span.set_extension("scope"    , default=(-1,-1), force=True)
        # for concepts
        Span.set_extension("negated"  , default=  False, force=True)
        Span.set_extension("uncertain", default=  False, force=True)
        Span.set_extension("patient"  , default=  True, force=True)
        Span.set_extension("recent"   , default=  True, force=True)

    def __call__(self, file, annotation="brat", return_all=False):
        if annotation == "brat":
            text_df, ann_T, ann_R = self.load_brat_data( file )
            docs         = [ self.nlp(text_df.loc[i,'text'].strip('\n'), disable=["ner"]) for i in text_df.index ]
            if ann_T is None: return docs
            docs_labels, ann_T = self.label_all_docs( docs, ann_T, ann_R )
            if return_all: return docs_labels, ann_T, ann_R
            return docs_labels
        
        if annotation == "webanno":
            if self.has_neg_status(file):
                return self.get_webann_docs(file, is_simple=False)
            else:
                return self.get_webann_docs(file, is_simple=True)
    
    ##########################################################################
    #### functions for webanno format ########################################
    ##########################################################################

    def has_neg_status(self, file):
        with open( file, 'r', encoding='utf-8' ) as f:
            lines = f.readlines()
            if "#FORMAT=WebAnno" not in lines[0]:
                print("ERROR:", "'"+file+"'", "is not of WebAnno format")
                sys.exit(1)
            second_line = lines[1]
            if not second_line.startswith('#T_SP='):
                print("ERROR:", "'"+file+"'", "does not start with '#T_SP='")
                sys.exit(1)
            if "|afirmado|paciente|reciente" in second_line:
                return True
            elif len(second_line[6:].split("|")) != 2:
                print("ERROR:", "'"+file+"'", "has an unexpected number of '|' in second line")
                sys.exit(1)
            else:
                return False

    def get_categories(self, file):
        categories = []
        with open( file, 'r', encoding='utf-8' ) as f:
            for l in f.readlines():
                if l.startswith('#FORMAT='): continue
                if l.startswith('#T_SP='):
                    categories.append(l.replace(".", "|").split("|")[2].lower())
                else:
                    break
        return categories

    def get_webann_docs(self, file, is_simple):
        cats = self.get_categories(file)
        docs = []
        doc = None
        long_span = {}
        match_pat = "\*\[" if is_simple else "true\["
        fullmatch_pat = "\*" if is_simple else "true|false"
        with open( file, 'r', encoding='utf-8' ) as f:
            for line in f.readlines():
                line = re.sub(' +', ' ', line) # remove multiple spaces
                l = line.strip("\n").strip("\t").split("\t")
                if long_span:
                    # resume the last long span or start a new long span
                    if any(re.match(match_pat, status) for status in l[3:]):
                        num = self.get_long_span_num(l, is_simple=is_simple)
                        if num == long_span["num"]:
                            long_span["end"] += 1
                            continue
                        self.finish_long_span(doc, long_span, l, cats, is_simple=is_simple)
                        long_span = self.init_long_span(l, num, is_simple=is_simple)
                    # finish the current long span
                    else:
                        self.finish_long_span(doc, long_span, l, cats, is_simple=is_simple)
                        long_span = {}

                if l[0].startswith('#Text='):
                    if doc: docs.append(doc)
                    doc = self.nlp(line[6:])
                    doc.ents = []
                if len(l[0])==0 or l[0].startswith('#'): continue
                else:
                    if any(re.fullmatch(fullmatch_pat, status) for status in l[3:]):
                        doc.ents = doc.ents + (self.get_labeled_span(doc, l, cats, is_simple=is_simple),)
                    elif any(re.match(match_pat, status) for status in l[3:]):
                        long_span = self.init_long_span(l, self.get_long_span_num(l, is_simple=is_simple), is_simple=is_simple)
            if doc: docs.append(doc)
        return docs

    # return a dict containing relevent infomation for hte long span
    def init_long_span(self, l, num, is_simple):
        long_span = {}
        long_span["num"] = num
        long_span["start"] = int(l[0].split("-")[1]) - 1
        long_span["end"] = long_span["start"] + 1
        r = re.compile("\*\[.*]") if is_simple else re.compile("(true|false)\[.*]")
        for x in filter(r.match, l[3:]):
            long_span["label_start"] = l.index(x)
            break
        return long_span

    # get the id number for the long span to differentiate it from other long spans
    def get_long_span_num(self, l, is_simple):
        if is_simple: # if the doc is the simpler version
            r = re.compile("\*\[.*]")
            for x in filter(r.match, l[3:]):
                return x.strip("*[").strip("]")
        else: # if the doc is the complete version
            r = re.compile("(true|false)\[.*]")
            for x in filter(r.match, l[3:]):
                return x.strip("true[").strip("false[").strip("]")            

    # process the long span and add it to doc.ents
    def finish_long_span(self, doc, long_span, l, cats, is_simple):
        attr_i = long_span["label_start"]
        cat_i = attr_i - 3 if is_simple else int((attr_i / 3) - 1)
        span = Span(doc, long_span["start"], long_span["end"], label=cats[cat_i])
        span._.negated = l[attr_i] == "false"
        span._.patient = l[attr_i + 1] == "true"
        span._.recent = l[attr_i + 2] == "true"
        doc.ents = doc.ents + (span,)

    # assume only one category is labeled for the token
    def get_labeled_span(self, doc, line, categories, is_simple):
        token_i = int(line[0].split("-")[1]) - 1
        attr_i = -1
        if is_simple:
            attr_i = line.index("*")
        else:
            if "true" in line:
                if "false" in line:
                    attr_i = min(line.index("true"), line.index("false"))
                else:
                    attr_i = line.index("true")
            else:
                attr_i = line.index("false")
        cat_i = attr_i - 3 if is_simple else int((attr_i / 3) - 1)
        span = Span(doc, token_i, token_i+1, label=categories[cat_i])
        span._.negated = False
        span._.patient = True
        span._.recent = True
        return span

    ##########################################################################
    #### functions for brat format ###########################################
    ##########################################################################

    def load_brat_data(self, file ):
        file         = re.sub(".ann$|.txt$", "", file)
        text_df      = self.read_brat_txt( file +'.txt' )
        ann_T, ann_R = self.read_brat_ann( file +'.ann' )
        ann_T        = self.update_ann_T( ann_T, text_df )
        return text_df, ann_T, ann_R

    def read_brat_txt(self, table_dir):
        with open( table_dir, "r", encoding="utf-8") as f:
            texts = [line for line in f]
        return pd.DataFrame([(len("".join(texts[0:i])),texts[i]) for i in range(len(texts))], columns=['len','text'])

    def read_brat_ann(self, table_dir):
        with open( table_dir, "r", encoding="utf-8") as f:
            ann_l = [line.strip('\n').split('\t') for line in f]
        ann_df = pd.DataFrame(ann_l)
        if ann_df.shape == (0,0): return None, None
        ann_df = pd.concat([ann_df[0], pd.DataFrame(ann_df[1].str.split(" ").tolist()), ann_df[2]], 1)
        ann_df.columns = ["abbrev", "tag", "start", "end", "text"]
        return self.split_ann_df(ann_df)

    def split_ann_df(self, ann_df):
        ann_df_T         = ann_df.loc[ann_df['abbrev'].str.startswith('T')].copy(deep=True)
        ann_df_T         = ann_df_T.astype({'start':int,'end':int})
        ann_df_R         = ann_df.loc[ann_df['abbrev'].str.startswith('R')].copy(deep=True)
        ann_df_R['Arg1'] = ann_df_R['start'].str.lstrip('Arg1:').tolist()
        ann_df_R['Arg2'] = ann_df_R['end'  ].str.lstrip('Arg2:').tolist()
        ann_df_R         = ann_df_R.drop(columns=['text','start','end'])
        return ann_df_T, ann_df_R

    def update_ann_T(self, ann_df_T, text_df):
        if ann_df_T is None: return ann_df_T
        ann_df_T['doc_i'] = 0
        ann_df_by_doc_T = []
        for doc_i in text_df.index[1:]:
            in_doc = ann_df_T['start'] < text_df.loc[doc_i,'len']
            ann_df_by_doc_T.append( ann_df_T.loc[   in_doc].copy(deep=True).assign(doc_i=doc_i-1) )
            ann_df_T =              ann_df_T.loc[ ~ in_doc]
            ann_df_by_doc_T[-1][['start','end']] = ann_df_by_doc_T[-1][['start','end']] - text_df.loc[doc_i-1,'len']
        if not ann_df_by_doc_T: return ann_df_T  # ann_df_by_doc_T is sometimes empty
        return pd.concat(ann_df_by_doc_T)
    
    def label_all_docs(self, docs, ann_T, ann_R):
        docs_lab = [ self.label_ents_in_doc(  docs[i], ann_T.loc[ann_T['doc_i']==i]        ) for i in range(len(docs    )) ]
        docs_lab, ann_T = [x[0] for x in docs_lab], pd.concat([x[1] for x in docs_lab])
        if ann_R.shape[0]==0: return docs_lab, ann_T
        docs_lab = [ self.set_ent_scopes( docs_lab[i], ann_T.loc[ann_T['doc_i']==i], ann_R ) for i in range(len(docs_lab)) ]
        docs_lab = [ self.set_ent_status( docs_lab[i] ) for i in range(len(docs_lab)) ]
        return docs_lab, ann_T

    def label_ents_in_doc(self, doc, ents_df):
        ents_df = ents_df.assign( start_tk=-1, end_tk=-1)
        for i in ents_df.index:
            ent = self.craft_entity( doc, ents_df.loc[i] )
            if ent is None: continue
            ents_df.loc[i,'start_tk'] = ent.start
            ents_df.loc[i,  'end_tk'] = ent.end
            if len(doc.ents)>0 and self.spans_overlap( ent.start, ent.end, doc.ents[-1].start, doc.ents[-1].end ): 
                doc.ents = doc.ents[:-1] + (self.keep_one(ent, doc.ents[-1]),)
            else:
                doc.ents = doc.ents      + (ent,)
        return doc, ents_df
        
    def craft_entity(self, doc, ent_df ):
        ent = doc.char_span( ent_df['start'], ent_df['end'], ent_df['tag'])
        if ent is not None: ent._.ent_id = ent_df['abbrev']
        return ent

    def spans_overlap(self, span1_start, span1_end, span2_start, span2_end):
        for i in range(span1_start, span1_end):
            if i in range(span2_start, span2_end):
                return True
        return False

    def set_ent_scopes(self, doc, ann_T, ann_R): # for modifiers
        for ent in doc.ents:
            in_scope = ann_R.loc[ ann_R['Arg1']==ent._.ent_id, 'Arg2'].tolist()
            in_scope = ann_T.set_index('abbrev').loc[in_scope]
            ent._.scope = (max(-1,in_scope['start_tk'].min()), max(-1,in_scope['end_tk'].max()))
        return doc
    
    def set_ent_status(self, doc): # for concepts
        for ent_cpt in doc.ents:
            for ent_mod in doc.ents:
                if ent_mod == ent_cpt: continue
                if ent_mod._.scope[0] == -1: continue
                if self.spans_overlap( ent_cpt.start, ent_cpt.end, ent_mod._.scope[0], ent_mod._.scope[1]):
                    ent_cpt._.negated   = ent_mod.label_.startswith('Neg')
                    ent_cpt._.uncertain = ent_mod.label_.startswith('Uncert')
        return doc

    # Choose one span to return based on the following hierarchy
    #   .*Marker -> DISO/OTHER -> .... -> .*Item
    # Return span2 if they are of the same type; span2 occurs after
    #   span1 in the BRAT annotation document
    def keep_one(self, span1, span2):
        label1 = span1.label_
        label2 = span2.label_
        if label1 != label2 and label1 in ["OTHER", "DISO"] and label2 in ["OTHER", "DISO"]:
            return span2
        if "Item" in label1 or "Phrase" in label1: return span2
        if "Item" in label2 or "Phrase" in label2: return span1
        for label in ["Marker", "DISO", "OTHER"]:
            ret = self.find_label(label, span1, label1, span2, label2)
            if ret: return ret
        print("ERROR: unreachable region in Annotator.keep_one()")        

    def find_label(self, label, span1, label1, span2, label2):
        if label in label2:
            return span2
        if label in label1:
            return span1
        return None
