# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

import spacy

import unidecode
import pandas as pd
import re

# Initializing the spaCy model instance
spacy_md = spacy.load('en_core_web_md', disable=['ner'])
spacy_md.max_length = 2000000


PATENT_UNITS = ['k', 'h', 'v', 'wt', 'wt.', 'mhz', 'khz', 'ghz', 'hz', 'days', 'weeks', 'hours',
                    'minutes', 'seconds', 't', 'mpa', 'gpa', 'at.', 'mol.', 'at', 'm', 'n', 's-1', 'vol.',
                    'vol', 'ev', 'a', 'atm', 'bar', 'koe', 'oe', 'h.', 'mwcm−2', 'kev', 'mev', 'mev', 'day',
                    'week', 'hour', 'minute', 'month', 'months', 'year', 'cycles', 'years', 'fs', 'ns', 'ps',
                    'rpm', 'g', 'mg', 'macm−2', 'ma', 'mk', 'mt', 's-1', 'db', 'ag-1', 'mag-1', 'mag−1', 'mag',
                    'mah', 'mahg−1', 'm-2', 'mj', 'kj', 'm2g−1', 'thz', 'khz', 'kjmol−1', 'torr', 'gl-1', 'vcm−1',
                    'mvs−1', 'j', 'gj', 'mtorr', 'bar', 'cm2', 'mbar', 'kbar', 'mmol', 'mol', 'moll−1', 'mω',
                    'ω', 'kω', 'mω', 'mgl−1', 'moldm−3', 'm2', 'm3', 'cm-1', 'cm', 'scm−1', 'acm−1', 'ev-1cm-2',
                    'cm-2', 'sccm', 'cm−2ev−1', 'cm-3ev-1', 'ka', 's−1', 'emu', 'l', 'cmhz1', 'gmol−1', 'kvcm-1',
                    'mpam1', 'cm2v-1s-1', 'acm−2', 'cm−2s−1', 'mv', 'ionscm−2', 'jcm−2', 'ncm−2', 'jcm−2', 'wcm−2',
                    'gwcm−2', 'acm-2k-2', 'gcm−3', 'cm3g-1', 'mgl−1', 'mgml−1', 'mgcm−2', 'mωcm', 'cm−2s−1', 'cm−2',
                    'ions', 'moll−1', 'nmol', 'psi', 'jkg-1k-1', 'km', 'wm−2', 'mass', 'mmhg', 'mmmin−1',
                    'gev', 'm−2', 'm-2s-1', 'kmin−1', 'gl−1', 'ng', 'hr', 'w', 'mn', 'kn', 'mrad', 'rad', 'arcsec',
                    'ag−1', 'dpa', 'cdm−2', 'cd', 'mcd', 'mhz', 'm−3', 'ppm', 'phr', 'ml', 'ml', 'mlmin−1', 'mwm−2',
                    'wm-1k-1', 'kwh', 'wkg−1', 'jm−3', 'm-3', 'gl−1', 'a−1', 'ks−1', 'mgdm−3', 'mms−1',
                    'ks', 'appm', 'ºc', 'hv', 'kda', 'da', 'kg', 'kgy', 'mgy', 'gy', 'mgy', 'gbps', 'μb', 'μl',
                    'μf', 'nf', 'pf', 'mf', 'a', 'å', 'a˚', 'μgl−1', 'mgl-1']

stopwordSet =  {'ok', 'important', 'probably', 'results', 'various', 'e', 'whence', 'four', 'followed', 'its', 'whenever',
                    'thereof', 'against', 'herein', 'e.g.', 'new', 'general', 'e.x.', 'relates', 'or', 'first', 'looks', 'looking',
                    'these', 'besides', 'without', 'relatively', 'either', 'hereupon', 'need', 'thence', 'wants', 'came', 'november',
                    'entirely', 'wonder', "hasn't", 'mg', 'getting', 'those', 'at', 'become', 'my', "they'll", 'v', 'containing', 'm',
                    'however', 'others', 'description', 'december', 'doing', 'if', 'comprising', 'name', 'i', 'moreover', 'seen', 'twice',
                    'available', 'course', 'indicate', 'gone', 'me', 'as', 'everything', 'got', 'non', 'regardless', 'might', 'plus',
                    'inner', 'mainly', 'theres', 'one', 'goes', 'indicates', 'merely', 'perhaps', 'our', 'thus', 'near', 'iii', "we'll",
                    'whoever', "hadn't", 'reference', "you'll", 'so', 'always', "ain't", 'used', 'ones', 'myself', "it's", 'placed', 'be',
                    'th', 'useful', 'themselves', 'toward', 'whole', 'and', 'downwards', 'sensible', 'too', 'corresponding', 'et', 'february',
                    'could', 'off', 'october', 'for', 'together', 'needs', 'thats', 'yet', 'hence', 'below', "doesn't", "i'm", 'hi', 'know',
                    'therefor', 'theirs', 'come', 'cannot', 'from', 'prior', 'hereby', 'certainly', 'whereas', 'did', 'patent', 'appear',
                    'w', 'of', 'something', 'clearly', 'allows', 'actually', 'willing', 'known', 'on', 'than', 'variant', 'had', 'cant',
                    'ever', 'rather', 'contain', 'particular', 'respectively', 'wherein', 'therein', 'certain', "shouldn't", 'using',
                    'saw', 'contains', 'july', 'here', 'nowhere', 'regarding', 'ml', 'consequently', 'this', 'three', 'herself', 'gives',
                    'even', 'currently', 'concerning', 'sorry', "i've", 'go', "it'd", 'everyone', 'next', 'tried', 'yourselves', 'wt',
                    'y', 'yours', 'onto', 'appreciate', 'way', 'amongst', 'beside', 'having', 'ourselves', 'detail', "what's", 'through',
                    "they've", 'six', 'another', 'every', 'everywhere', 'outside', 'january', 'obviously', 'mean', 'far', 'how', "t's",
                    'but', 'little', 's', 'seeming', 'anyone', 'consider', 'still', 'neither', "they'd", 'normally', 'nm', 'whereby',
                    'being', 'different', 'references', 'whether', 'thanx', 'figs', 'april', 'examples', 'que', 'that', 'eg', 'secondly',
                    'under', 'says', 'alone', 'beyond', 'invention', 'self', 'us', 'though', 'down', 'then', 'g', 'following', 'p', 'done',
                    "couldn't", 'asking', 'let', 'fig', 'well', 'claims', 'kind', 'kinds', 'z', 'whither', "won't", 'no', 'within', 'none',
                    "wouldn't", 'few', 'j', 'according', 'anyhow', 'vs', 'awfully', 'ignored', 'help', 'you', 'believe', "don't", 'regards',
                    'inventionthe', 'applicant', 'not', "isn't", 'sent', 'around', 'sure', 'i.e', 'upab', 'present', "weren't", 'keep', 'which',
                    'along', 'quite', 'later', 'necessary', 'thereupon', 'specify', 'figure', 'furthermore', 'like', "aren't", 'disclosures',
                    'nothing', 're', 'field', 'appropriate', 'lately', 'really', 'sixth', 'namely', 'objective', 'going', 'never', 'summary',
                    'accordingly', 'can', 'despite', 'reasonably', 'welcome', "where's", 'your', 'september', 'immediate', 'changes', 'com',
                    'think', 'whereupon', 'why', 'in', 'because', 'ask', 'thorough', 'towards', 'else', 'mostly', 'described', 'right', 'unless',
                    'nobody', 'knows', "they're", 'inventions', 'seeing', 'second', 'with', 'okay', 'saying', 'after', 'afterwards', 'keeps', 'less',
                    'do', 'gets', "here's", 'last', 'we', 'former', 'during', "you'd", 'inward', 'selves', 'some', 'whose', "let's", 'they', 'uses',
                    'patents', 'claim', 'latter', 'nearly', 'enough', 'many', 'thereafter', 'hither', 'end', "haven't", 'kg', 'seems', 'the', 'example',
                    'e.g', "there's", 'already', 'once', 'disclosed', "didn't", 'serious', 'r', 'sub', 'forth', 'wish', 'otherwise', 'descriptions',
                    'until', 'a', 'take', 'is', 'must', 'look', 'unlikely', 'hardly', 'upon', 'tries', 'result', 'fourth', 'novelty', 'such', 'very',
                    'formerly', 'o', 'per', 'inc', 'only', 'via', 'background', 'ch', 'lot', 'more', 'tell', 'gotten', 'comes', 'eight', 'what',
                    'tends', 'anywhere', 'x', 'particularly', 'novel', 'qv', 'fieldthe', 'try', 'just', "wasn't", 'figures', 'both', 'anybody',
                    'taken', 'c', 'q', 'uucp', 'lest', 'to', 'causes', 'h', 'should', 'overall', 'summaries', 'itself', 'now', 'although', 'kept',
                    'detailed', 'him', 'beforehand', 'june', 'most', 'nevertheless', 'greetings', 'nine', 'shall', 'are', 'thru', 'preferably',
                    'between', 'has', 'his', 'becomes', 'whatever', 'somewhat', 'hereafter', 'went', 'trying', 'away', 'thoroughly', 'throughout',
                    "can't", 'anyway', 'again', 'anything', 'see', "you're", 'any', 'someone', 'since', 'value', 'rd', 'follows', 'said', 'comprises',
                    'aside', 'was', 'usually', 'fifth', "we'd", 'therefore', 'want', 'ex', 'soon', 'wherever', 'indicated', 'ie', 'presumably', 'where',
                    'up', 'own', 'further', 'please', 'sup', 'seemed', 'august', 'were', 'among', 'noone', 'fieldthis', 'inventionfield', 'ltd', 'un',
                    'does', 'hello', 'd', "we've", 'each', 'mm', "you've", 'say', 'thereby', 'am', 'especially', 'liked', 'use', 'whereafter', 'except',
                    'e.x', 'cm', 'seven', 'specifying', "i'll", 'meanwhile', 'happens', 'f', 'l', "c's", 'above', 'he', 'their', 'latterly', 'brief',
                    "that's", 'somebody', "we're", 'thanks', 'formula', 'cause', 'ii', 'provides', 'while', 'them', 'an', 'over', 'elsewhere', 'will',
                    'whom', 'before', 'embodiment', 'may', 'anyways', 'became', 'truly', "i'd", 'inasmuch', 'other', 'also', 'have', 'b', 'likely',
                    'about', 'same', 'al', 'five', 'least', 'sometimes', 'cross-reference', 'it', 'behind', 'definitely', 'best', 'hopefully', 'two',
                    'u', 'k', 'somehow', 'ought', 'thereforthereof', 'instead', 'insofar', 'apart', 'often', 'march', 'zero', 'edu', 'maybe', 'better',
                    'unto', 'disclosure', 'howbeit', "it'll", 'ours', "who's", 'took', 'embodiments', 'given', 'much', 'she', 'specified', 'associated',
                    'n', 'exactly', "he's", 'when', 'yes', 'hz', 'considering', 'several', 'allow', 'everybody', 'able', 'hers', 'including', 't',
                    'across', 'co', 'nor', 'into', 'possible', 'who', 'strong', 'sometime', 'third', 'ph', 'becoming', 'been', 'her', 'oh', 'out',
                    'all', 'would', 'related', 'independent', 'by', 'get', 'viz', 'there', 'unfortunately', 'preferred', 'almost', 'seriously', 'thank',
                    'inventor', 'somewhere', 'himself', 'nd', 'etc', 'indeed', 'thereto', 'old', 'yourself', 'seem'}

customizedStopwordSet = {'compound', 'threshold', 'novelty', 'structure', 'complex', 'test', 'selection', 'active', 'process', 'command', 'addition',
                             'method', 'initial', 'mixture', 'icon', 'introduction', 'program', 'basis', 'facility', 'vector', 'case', 'size', 'phase',
                             'player', 'thi', 'apparatus', 'tool', 'value', 'second', 'year', 'subset', 'write', 'solution', 'objectives', 'product',
                             'contrary', 'mini', 'advantage', 'period', 'function', 'universal', 'increasing', 'presence', 'nan', 'code', 'context',
                             'number', 'bit', 'part', 'record', 'bus', 'objectiv', 'processor', 'methods', 'procedure', 'week', 'line', 'surface',
                             'main', 'query', 'progress', 'count', 'information', 'unit', 'sample', 'copy', 'update', 'device', 'dataset', 'system',
                             'background', 'half', 'variable', 'average', 'priority', 'sub', 'content', 'project', 'multiple', 'store', 'connection',
                             'configuration', 'report', 'private', 'act', 'down', 'layer', 'public', 'concern', 'abstracts', 'approach', 'access',
                             'original', 'pair', 'execute', 'output', 'document', 'label', 'width', 'ability', 'non', 'outbreak', 'communication',
                             'integer', 'definition', 'effective', 'side', 'request', 'amount', 'analyte', 'application', 'condition', 'ticket',
                             'parent', 'rule', 'module', 'principle', 'organization', 'operation', 'titles', 'th', 'relating', 'piece', 'situation',
                             'server', 'location', 'company', 'insert', 'respect', 'compute', 'datum', 'select', 'order', 'step', 'relate', 'higher',
                             'component', 'type', 'letter', 'task', 'reviews', 'month', 'simple', 'position', 'portion', 'art', 'format', 'assignee',
                             'response', 'problem', 'null', 'link', 'member', 'paper', 'combination', 'upper', 'computer', 'increase', 'accordance',
                             'change', 'hour', 'office', 'cooperation', 'central', 'limit', 'domain', 'set', 'play', 'lower', 'array', 'easier',
                             'programmer', 'base', 'word', 'top', 'hard', 'capabilitiesthe', 'acceptable', 'unique', 'related', 'signal', 'field',
                             'box', 'steps', 'service', 'page', 'employee', 'sum', 'conduct', 'upab', 'area', 'characteristics', 'abstract', 'person',
                             'seq', 'proof', 'applicant', 'inventionthis', 'article', 'review', 'path', 'therefrom', 'virus', 'section', 'soft',
                             'property', 'data', 'opening', 'applied', 'time', 'processing', 'max', 'column', 'status', 'protocol', 'applications',
                             'region', 'form', 'list', 'manner', 'element', 'input', 'web', 'aspect', 'space', 'level', 'invention1', 'composition',
                             'view', 'fact', 'series', 'compromise', 'random', 'object', 'item', 'site', 'concept', 'result', 'stage', 'reports',
                             'friend', 'criteria', 'account', 'preceding', 'counter', 'weight', 'agent', 'aim', 'message', 'specifically', 'execution',
                             'invention', 'chance', 'segment', 'length', 'secndary', 'subject', 'group', 'job', 'open', 'internet', 'user', 'control',
                             'date', 'technology', 'study', 'drive', 'title', 'recent', 'term', 'read', 'exclusive', 'technical', 'feature', 'state',
                             'systemfield', 'means', 'minute', 'producing', 'things', 'matter', 'front', 'concepts', 'source', 'digital', 'testing',
                             'move', 'inventionfield', 'sheet', 'circuit', 'file', 'plurality', 'image', 'forward', 'table', 'center', 'technique',
                             'subgroup', 'requirement', 'wt', 'matrix', 'outbreaks'}

generalNPSet = {'japanese_application', 'desc_clm_page', 'technical_relate', 'chinese_application', 'seq_no', 'subject_matter', 'large_amount', 'technical_solution',
         'technical_field1', 'method_relate', 'input_information', 'document_japanese', 'input_data', 'technology_relate', 'application_serial',
         'application_relate', 'technical_fieldthe_relate', 'combination_compositions', 'present_invention', 'industrial_utility', 'chinese_office',
         'serial_number', 'technical_method', 'data_item', 'method_field', 'technical_application', 'meta_data', 'significant_amount', 'technical_field',
         'desc_clm_page_number', 'international_application', 'application_number', 'inventionthe_relate', 'desc_clm', 'background_art', 'page_number',
         'japanese_office', 'invention_relate', 'provisional_application', 'technical_problem', 'co-pending_application', 'technical_fieldthe'}




def clean_text(txt, spacy_nlp):

    txt = txt.lower().strip()
    txt = txt.translate({ord(c): " " for c in "!@#$%^&*ω()»¿'[]{};:,./<>?|`~°=\"+"})
    doc = spacy_nlp(txt)

    clean_text = ""
    for token in doc:

        if not token.is_stop:
            token = token.lemma_
            token = unidecode.unidecode(token)
            # token = token.translate({ord(c): " " for c in "!@#$%^&*ω()»¿'[]{};:,./<>?|`~°=\"+"})

            if not token.replace('-', '').strip().isnumeric() and not token.strip() in customizedStopwordSet and len(
                    token.strip()) >= 3 and not token.strip() in stopwordSet and not token.strip() in customizedStopwordSet:
                clean_text = clean_text + " " + token

    return clean_text



class PatentTextPreProcessor:

    def __init__(self):
        '''
        '''

    def get_terms_from_file(self, filePath):
        '''
        loading terms from a file to a set
        :param filePath:
        :return:
        '''
        terms = set(line.strip() for line in open(filePath))
        return terms


    def remove_terms(self, termSet, phrase):
        '''
        remove undesired terms such as stopwords and general terms
        :param phrase:
        :return:
        '''
        tokens = phrase.lower().split()
        tokens = [w.strip('-') for w in tokens if w not in termSet ]

        newText = ' '.join(tokens)
        return newText

    def remove_digit_unit(self, phrase):
        '''
        remove all digits and units from a text
        :param phrase:
        :return:
        '''
        tokens = phrase.lower().split()
        tokens = [w.strip('-') for w in tokens if len(w.strip()) > 2 and w.strip() not in PATENT_UNITS
                  and not w.replace('-', '').isnumeric() ]

        newText = ' '.join(tokens)
        return newText

    def remove_accent(self, txt):
        """Removes accents from a string.
        Args:
            txt: The input string.
        Returns:
            The de-accented string.
        """
        # There is a problem with angstrom sometimes, so ignoring length 1 strings.
        return unidecode.unidecode(txt) if len(txt) > 1 else txt

    def get_lemma(self, np):
        normalized_np = []
        for token in spacy_md(np):
            normalized_np.append(token.lemma_.strip())
        return ' '.join(normalized_np)

    def phrase2nGrams(self, phrase):
        '''
        spliting a phrase into nGrams
        :param phrase:
        :return:
        '''
        tokens = phrase.split()
        if len(tokens) < 3:
            return self.phrase_to_token(phrase.strip())

        newPhrases = tokens[0] + "_" + tokens[1]
        combinedWords = newPhrases
        for i in range(2, len(tokens)):
            combinedWords = combinedWords + "_" + tokens[i]
            newPhrases = newPhrases + " " + combinedWords

        return newPhrases

    def phrase_to_token(self, phrase):
        '''
        combining words of a phrase into one token
        :param phrase:
        :return:
        '''
        phraseToken = phrase.replace(" ", "_")

        return phraseToken

    def extract_nounPhrases(self, docTxt, ngram=False, normalized=False):
        '''
        extract noun phrases from each document
        :param docTxt:
        :param stopwordSet:
        :param customizedStopwordSet:
        :param generalNPSet:
        :param non_stemming_set:
        :param nlp_spacy:
        :return:
        '''
        phrases = ""
        chunkDoc = spacy_md(docTxt)
        for chunk in chunkDoc.noun_chunks:
            currentChunk = chunk.text.lower().strip().replace("\n", "")

            if currentChunk.strip() != "":
                currentChunk = currentChunk.translate({ord(c): " " for c in "!@#$%^&*ω()»¿'[]{};:,./<>?\|`~°=\"+"})

            if normalized:
                phrase = self.get_lemma(currentChunk)
            else:
                phrase = currentChunk

            phrase = self.remove_terms(stopwordSet, phrase)
            phrase = self.remove_digit_unit(phrase)
            # phrase = singularize_text(phrase, non_stemming_set)

            if len(phrase.strip().split()) == 1:
                phrase = self.remove_terms(customizedStopwordSet, phrase)
            elif len(phrase.strip().split()) > 1:
                # convert phrase to nGram
                if ngram:
                    phrase = self.phrase2nGrams(phrase)
                else:
                    phrase = self.phrase_to_token(phrase)

                # remove general phrases
                phrase = self.remove_terms(generalNPSet, phrase)

            if phrase.strip() != "":
                phrases = phrases + "\n " + phrase.strip()

        return phrases.replace("\n", " ")

    def preprocess_text(self, docText, ngram:False, normalized=False):
        '''
        preprocessing pipeline for patent texts
        :param docText:
        :param stopwordSet:
        :param customizedStopwordSet:
        :param generalNPSet:
        :param non_stemming_set:
        :param nlp_spacy:
        :return:
        '''
        sentences = ""
        sentenceDoc = spacy_md(docText)
        for sent in sentenceDoc.sents:
            sentences = sentences + "\n " + self.extract_nounPhrases(sent.text, ngram=ngram,normalized=normalized)

        #print('One document has been processed')
        return sentences.replace("\n", " ").strip()

    def run_NLP_stream(self, dfData, ngram=False, normalized=False):
        '''
        apply NLP tasks for a multi patent texts Panda Dataframe
        :param dfData:
        :param nlp_spacy_model:
        :return:
        '''

        dfData['TEXT'] = dfData['TEXT'].map(
            lambda line: self.preprocess_text(self.remove_accent(line, ngram=ngram, normalized=normalized)))

        return dfData

    def run_NLP(self, doc_txt, ngram=False, normalized=False):
        '''
        apply NLP tasks for a single patent texts
        :param doc_txt:
        :return:
        '''

        return self.preprocess_text(self.remove_accent(doc_txt, ngram=ngram, normalized=normalized))

    def doc_to_pragraph(df, min_paragraph_length=10):
        '''
        convert a text as a panda column into paragraphs (several rows )
        :param nlp_spacy:
        :param min_paragraph_length:
        :return:
        '''
        paragraph_df = pd.DataFrame(columns=['ID', 'paragraph'])
        counter = 0

        for i in range(len(df)):
            ID = df.ID.iloc[i]
            docText = df.TEXT.iloc[i]

            sentenceDoc = spacy_md(docText)
            for sent in sentenceDoc.sents:
                if len(sent.text.split()) > min_paragraph_length:
                    row_list = [ID, sent.text]
                    paragraph_df.loc[counter] = row_list
                    counter += 1

        return paragraph_df


    def get_pure_noun_phrases(self, txt):
        '''
        extrcat all noun phrases from atext
        :param txt:
        :param spacy_model:
        :return:
        '''

        phrases = []
        chunkDoc = spacy_md(txt)
        for chunk in chunkDoc.noun_chunks:
            phrases.append(chunk.text)

        return phrases

    def normalize_np(self, np):
        """
         Function to normalize a noun phrase
        :param np:
        :param nlp_md:
        :return:
        """
        normalized_np = []
        for token in spacy_md(np):
            stopwords = stopwordSet
            if token.text.lower() not in stopwords and not token.is_punct:
                normalized_token = token.lemma_.strip()
                if normalized_token:
                    normalized_np.append(normalized_token.strip())
        np = " ".join(normalized_np)
        np = np.strip()


        if np.lower() in stopwordSet or np.lower() in PATENT_UNITS:
            return ''

        return np.strip()


def soft_text_cleaning(text:str):
    # Remove punctuations
    text = text.replace("\\n", " ")
    text = re.sub('[^a-zA-Z]', ' ', text)
    #text = text.translate({ord(c): " " for c in "!@#$%^&*ω()»¿'[]{};:,./<>?|`~°=\"+"})
    # remove tags
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    # remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)
    # Remove frequent terms
    text = text.replace("DESC", '')
    compiled = re.compile(re.escape('Technical Field'), re.IGNORECASE)
    text = compiled.sub(' ', text)
    compiled = re.compile(re.escape('Field of the Invention'), re.IGNORECASE)
    text = compiled.sub(' ', text)

    return text.strip()


def soft_text_cleaning_stream(df):

    df['text'] = df['text'].map(lambda line: soft_text_cleaning(line))
    return df

