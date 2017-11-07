import os
import nltk
import gensim
import pickle
import string
import re


# Local Vars
user_prefix = r"C:\Users"
user_dir = os.path.join(user_prefix, os.getlogin())

# Stopwords
stopw = set(gensim.parsing.preprocessing.STOPWORDS)
stoplist = set(nltk.corpus.stopwords.words('english'))
stoplist = stopw.intersection(stoplist)

# Punctuation
punctuation = set(string.punctuation)
# Additional punctuation
my_punctuation = ["●", "•", "-"]
for myp in my_punctuation:
    punctuation.add(myp)
# Line, Paragraph breaks, etc.
wht_space = set(string.whitespace)
wht_space.discard(' ')

# Word Tokenizer
# Method for Stanford Tokenizer
os.environ['CLASSPATH'] = os.path.join(user_dir, r"Stanford\stanford-postagger-full-2017-06-09\stanford-postagger.jar")
tokenizer = nltk.tokenize.stanford.StanfordTokenizer()

# Word Stemmer
stemmer = nltk.stem.snowball.SnowballStemmer("english")

# Lemmatizer
lemmatizer = nltk.WordNetLemmatizer()

# WordNet
wordnet = nltk.corpus.wordnet

# POS Tagger
tagger = nltk.tag.StanfordPOSTagger('english-bidirectional-distsim.tagger')

# Phrase Model
path_to_phrase_model = os.path.join(user_dir, r"PycharmProjects\Perseus\dicts_models_etc\unlemquadgram_phrasemodel.model")
path_to_skills = os.path.join(user_dir, r"C:\Users\estasney\Google Drive\IPython Books\Common Models\skills_list.p")
bigram = gensim.models.Phrases.load(path_to_phrase_model)
bigrammer = gensim.models.phrases.Phraser(bigram)

# First Names
name_list_path = os.path.join(user_dir, r"Google Drive\IPython Books\Common Models\name_list.p")
with open(name_list_path, "rb") as p:
    name_list = pickle.load(p)
name_list = set(name_list)

# List of Skills
with open(path_to_skills, "rb") as p:
    skills_list = pickle.load(p)


# Cleaning Functions


def remove_noise(text, sent_mode=False):
    # CV/Resume Specific Cleaning
    # Returns tokens joined with ' ' (String)
    # Sent mode returns list of strings

    regex_email = re.compile(r"((\w|\d|\.)+)(@)(\w+)(\.)(\w{3})")
    regex_dates = re.compile(r"([A-z]+\.? ?\d{2,4}| +- (P|p)resent)|(\d{2}\/\d{2}\/\d{2,4})")
    regex_phone_numbers = re.compile(r"(\d{3}(-|.)){2}(\d{4})")
    regex_three_or_more = re.compile(r"\w*(.)(\1){2,}\w*")  # If a word contains a series of 3 or more identical letters
    regex_bullets = re.compile(r"(•|✓|#|\*|●  *)|(\d[.|:])|( ?- )")
    regex_hyperlinks = re.compile(r"(http)([a-z]|[A-Z]|[\d]|\.|\/|\?|\=|&|:|-)+")
    regex_numbers_only = re.compile(r"\d+[^A-z]")
    regex_punctuation = re.compile(
        r"(!|\"|#|\$|%|&|\||\)|\(|\*|\+|,|-|\.|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|`|\{|\||\}|~|')")

    if sent_mode:  # Returns a list of sentences
        for space in wht_space:
            text.replace(space, "\n")

        # Make a list of sentences
        sent_holder = nltk.tokenize.sent_tokenize(text)
        quiet_sents_clean = []
        for sentence in sent_holder:
            sentence = regex_phone_numbers.sub(" ", sentence)
            sentence = regex_email.sub(" ", sentence)
            sentence = regex_hyperlinks.sub(" ", sentence)
            sentence = regex_three_or_more.sub(" ", sentence)
            sentence = regex_numbers_only.sub(" ", sentence)
            sentence = regex_punctuation.sub(" ", sentence)
            sentence = sentence.lower()
            sentence = tokenizer.tokenize(sentence)
            sentence = ' '.join([word for word in sentence if word not in stoplist and word not in name_list
                                 and word not in punctuation])
            quiet_sents_clean.append(sentence)
        return quiet_sents_clean

    else:
        quiet_text = text
        quiet_text = regex_phone_numbers.sub(" ", quiet_text)
        quiet_text = regex_email.sub(" ", quiet_text)
        quiet_text = regex_hyperlinks.sub(" ", quiet_text)
        quiet_text = regex_dates.sub(" ", quiet_text)
        quiet_text = regex_three_or_more.sub(" ", quiet_text)
        quiet_text = regex_bullets.sub(" ", quiet_text)
        quiet_text = regex_numbers_only.sub(" ", quiet_text)
        quiet_text = regex_punctuation.sub(" ", quiet_text)
        clean_text = quiet_text.lower()
        clean_text = tokenizer.tokenize(clean_text)
        clean_text = ' '.join([word for word in clean_text if word not in stoplist and word not in name_list
                               and word not in punctuation])
    return clean_text


def document_to_tokens(clean_text, lem_tokens):
    tokens = tokenizer.tokenize(clean_text)
    if lem_tokens:
        pos_tokens = tagger.tag([clean_text])
        stemmed_tokens = token_stemmer(pos_tokens)
        return stemmed_tokens
    else:
        tokens = ' '.join(tokens)
        return tokens


def token_stemmer(pos_tagged_tokens):
    stemmed_tokens = []
    for tagged_tuple in pos_tagged_tokens:
        token = tagged_tuple[0]
        pos_tag = get_wordnet_pos(tagged_tuple[1])
        if pos_tag is False:
            lemmed_token = lemmatizer.lemmatize(token)
        else:
            lemmed_token = lemmatizer.lemmatize(token, pos_tag)
        if lemmed_token != token:
            stemmed_tokens.append(lemmed_token)
            continue
        # Try Morphy if no lemma found
        else:
            morphy_token = wordnet.morphy(token)
            if morphy_token is not None:
                stemmed_tokens.append(morphy_token)
            else:
                stemmed_tokens.append(lemmed_token)

    stem_join_tokens = ' '.join(token for token in stemmed_tokens)
    return stem_join_tokens


# https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return False


def clean_it(text, lem_tokens=True, gram_tokens=True, sent_mode=False):
    quiet_text = remove_noise(text, sent_mode=sent_mode)  # sent_mode will return a list of sentences
    if sent_mode:
        sent_holder = []
        for sent in quiet_text:  # String form
            if lem_tokens:
                sent = document_to_tokens(sent, lem_tokens=True)  # returns String form
            if gram_tokens:  # Must be list form
                sent = sent.split()  # List form
                clean_tokens = bigrammer[sent]
                continue_gram = True
                while continue_gram is True:
                    new_tokens = bigrammer[clean_tokens]  # Detect new phrases, must be tokens
                    if clean_tokens != new_tokens:
                        clean_tokens = new_tokens
                    else:
                        sent = new_tokens  # List form
                        sent = ' '.join(new_tokens)  # String rom
                        continue_gram = False
            sent_holder.append(sent)  # sent clean complete, append
            # Sent Holder Ready
        return sent_holder
    elif sent_mode is False:
        if lem_tokens:
            text_tokens = document_to_tokens(quiet_text, lem_tokens=True)
        else:
            text_tokens = document_to_tokens(quiet_text, lem_tokens=False)
        text_tokens = text_tokens.split()
        clean_tokens = ' '.join(
            [token for token in text_tokens if len(token) > 1 and token not in stopw and token not in punctuation])
        clean_tokens = clean_tokens.split()
        if gram_tokens:
            clean_tokens = bigrammer[clean_tokens]
            continue_gram = True
            while continue_gram is True:
                new_tokens = bigrammer[clean_tokens]  # Detect new phrases, must be tokens
                if clean_tokens != new_tokens:
                    clean_tokens = new_tokens
                else:
                    continue_gram = False
        clean_tokens = ' '.join(clean_tokens)
        return clean_tokens