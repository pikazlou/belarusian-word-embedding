import re
import os
import string
from gensim import utils
from gensim.parsing.preprocessing import strip_punctuation, strip_short, strip_numeric, strip_multiple_whitespaces, remove_stopwords
import urllib.request
import zipfile
import lzma
import shutil
import xml.dom.minidom
import json
from itertools import islice

import logging
logging.basicConfig(level=logging.INFO, force = True)
logger = logging.getLogger()
logger.info("Logging initialized")


def exists_nonempty_file(filename):
    return os.path.exists(filename) and os.path.getsize(filename) > 0


def corpus_downloaded():
    return exists_nonempty_file('be.txt')


def download_corpus():
    # Original file found here: https://metatext.io/datasets/cc100-belarusian
    urllib.request.urlretrieve('https://belarus-embedding.s3.eu-central-1.amazonaws.com/be.txt.xz',
                               'be.txt.xz')
    with lzma.open("be.txt.xz", "rb") as fsrc:
        with open("be.txt", "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)


def grammar_downloaded():
    return exists_nonempty_file('GrammarDB-master/N3.xml')


def download_grammar():
    urllib.request.urlretrieve('https://github.com/Belarus/GrammarDB/archive/refs/heads/master.zip',
                               'GrammarDB.zip')
    with zipfile.ZipFile('GrammarDB.zip', 'r') as zip_ref:
        zip_ref.extractall('.')


BASE_FORM_BLACKLIST = [
    'як', # can mean 'bull', but mostly used as particle
    'і' # for some reason listed as noun
] + [chr(ord('а')+delta) for delta in range(0, 32)] # alphabet letters

DERIVED_FORM_BLACKLIST = [
    'але', # can mean geographic place 'Ала', but mostly used as particle 'але'
    'калі', # weird form of 'калій' - 'каль', but used as particle 'калі'
    'вось', # can mean 'axis', but mostly used as particle
    'нам', # can mean short form of 'намеснік', but mostly used as pronoun 'мы'
    'наша', # some weird noun 'наша', but mostly used as pronoun 'мы'
    'нашы', # can be used as noun, but motly used as pronoun 'мы'
    'яму' # can be used as rare noun 'ям', but mostly used as pronoun 'ён'
]


def calculate_mapping_from_forms_to_base(filepath, tag_prefixes=[]):
    xml_doc = xml.dom.minidom.parse(filepath)
    paradigms = xml_doc.getElementsByTagName('Paradigm')
    result = {}
    collision_count = 0
    collisions = set()
    for paradigm in paradigms:
        tag = paradigm.getAttribute('tag')
        if len(tag_prefixes) == 0 or any([tag.startswith(p) for p in tag_prefixes]):
            variants = paradigm.getElementsByTagName('Variant')
            for variant in variants:
                base = variant.getAttribute('lemma').replace("+", "").lower()
                if base not in BASE_FORM_BLACKLIST:
                    forms = variant.getElementsByTagName('Form')
                    local_map = {}
                    citation_count = max([form.getAttribute('slouniki').count(',') for form in forms]) + 1
                    for form in forms:
                        if len(form.childNodes) > 0:
                            word = form.childNodes[0].data.replace("+", "").lower()
                            local_map[word] = (base, citation_count)
                    for k, v in local_map.items():
                        if k in result:
                            if result[k][1] == v[1] and result[k][0] != v[0]:
                                collision_count += 1
                                collisions.add(v[0])
                                collisions.add(result[k][0])
                            elif result[k][1] < v[1]:
                                result[k] = v
                        else:
                            result[k] = v
    logger.info(
        f"Collisions (forms leading to different base word, and having same amount of citation): {collision_count}")
    logger.info(f"Examples of collisions: {list(islice(collisions, 5))}")
    for k in result:
        result[k] = result[k][0]
    return result


def generate_word_mapping(filename, verbs=True, adjectives=True):
    word_map = {}
    if verbs:
        v = calculate_mapping_from_forms_to_base('GrammarDB-master/V.xml')
        word_map.update(v)

    nprop = calculate_mapping_from_forms_to_base('GrammarDB-master/NP.xml', ['NPII'])
    word_map.update(nprop)

    n1 = calculate_mapping_from_forms_to_base('GrammarDB-master/N1.xml')
    n2 = calculate_mapping_from_forms_to_base('GrammarDB-master/N2.xml')
    n3 = calculate_mapping_from_forms_to_base('GrammarDB-master/N3.xml')
    word_map.update(n1)
    word_map.update(n2)
    word_map.update(n3)

    if adjectives:
        adj1 = calculate_mapping_from_forms_to_base('GrammarDB-master/A1.xml', ['ARP', 'AQP'])
        adj2 = calculate_mapping_from_forms_to_base('GrammarDB-master/A2.xml', ['ARP', 'AQP'])
        word_map.update(adj1)
        word_map.update(adj2)

    manual_word_map = {
        'расеі': word_map['расіі'],
        'расея': 'расія',
        'расею': word_map['расію'],
        'расеяй': word_map['расіяй'],
        'ссср': 'ссср',
        'бсср': 'бсср',
        'бнр': 'бнр',
        'вкл': 'вкл',
        'смі': 'смі',
        'шоў': 'шоў',
        'тыс': 'тысяча',
        'млн': 'мільён',
        'вул': 'вуліца',
        'вобл': 'вобласць',
        'тэл': 'тэлефон',
        'км': word_map['кіламетр'],
        'навінаў': word_map['навін'],
        'тысячаў': word_map['тысяч'],
        'прэзыдэнта': word_map['прэзідэнта'],
        'прэзыдэнт': word_map['прэзідэнт'],
        'камэнтары': word_map['каментары'],
        'сыстэму': word_map['сістэму'],
        'сытуацыі': word_map['сітуацыі'],
        'сытуацыя': word_map['сітуацыя'],
        'цэнтар': word_map['цэнтр'],
        'вільня': word_map['вільнюс'],
        'вільню': word_map['вільнюс'],
        'сьмерці': word_map['смерці'],
        'грамадзтва': word_map['грамадства'],
        'эўропы': word_map['еўропы'],
        'сябраў': word_map['сяброў'],
        'апазыцыі': word_map['апазіцыі'],
        'міністар': word_map["міністр"],
        'мэню': word_map["меню"],
        'інтэрвію': word_map["інтэрв'ю"],
        'газэты': word_map["газеты"],
        'дакумэнты': word_map["дакументы"],
        'сытуацыю': word_map["сітуацыю"],
        'разьдзел': word_map["раздзел"],
        'сьмерць': word_map["смерць"],
        'калёніі': word_map["калоніі"],
        'газэта': word_map["газета"],
    }
    word_map.update(manual_word_map)

    if adjectives:
        manual_word_map = {
            'спэцыяльныя': word_map["спецыяльныя"],
            'грамадзкі': word_map["грамадскі"]
        }
        word_map.update(manual_word_map)

    with open(filename, 'w') as f:
        json.dump(word_map, f, ensure_ascii=False, indent=0, sort_keys=True)


def strip_trailing_newline(iterable):
    for i in iterable:
        yield i.rstrip()


# this function is based on gensim.parser.preprocessing.strip_punctuation
# we replace gensim's version to correctly handle symbol ' in words, such as п'еса or кар'ера
RE_PUNCTUATION = re.compile(r'([%s])+' % re.escape(string.punctuation.replace("'","")), re.UNICODE)
def strip_punctuation(s):
    s = utils.to_unicode(s)
    return RE_PUNCTUATION.sub(" ", s)


CHARACTERS_MAP = {'’': '\'', 'ý': 'ў', ' ў': ' у', 'i': 'і', 'ньн': 'нн', 'цьц': 'цц', 'сьц': 'сц', 'сьл':'сл', 'дзьдз': 'ддз', 'сьв': 'св', 'зьв': 'зв', 'сьп': 'сп', 'сьс': 'сс', 'сьн': 'сн', 'разьм': 'разм', 'зьмен': 'змен', 'зьмес': 'змес', 'зьмяс': 'змяс', 'зьмян': 'змян', 'зьн': 'зн', 'зьл': 'зл'}
def lower_and_replace_characters(iterable):
    for s in iterable:
        s = s.lower()
        for k, v in CHARACTERS_MAP.items():
            s = s.replace(k, v)
        yield s


def split_sentences(iterable):
    for i in iterable:
        merged_dots = re.sub("[\.]+", ".", i)
        sentences = merged_dots.split('.')
        for s in sentences:
            yield s


def process_and_filter_word(raw_words, word_map):
    valid_words = []
    removed_words = []
    for w in raw_words:
        w = w.strip("'")
        if w in word_map:
            valid_words.append(word_map[w])
        else:
            removed_words.append(w)
    return valid_words, removed_words


def preprocess_sentences(iterable, derived_form_blacklist, word_map, removed_words_accumulator: list):
    for i in iterable:
        s = strip_multiple_whitespaces(strip_numeric(strip_short(strip_punctuation(i))))
        s = re.sub("[«»“”„…—°′²]", "", s)
        s = remove_stopwords(s, stopwords=derived_form_blacklist)
        valid_words, removed_words = process_and_filter_word(s.split(), word_map)
        s = ' '.join(valid_words)
        removed_words_accumulator.extend(removed_words)
        yield s


def remove_short_lines(iterable):
    for i in iterable:
        if not i.isspace() and len(i) >= 20:
            yield i


def process_corpus(word_map_filename, processed_filename, removed_words_filename, split_sent):
    with open(word_map_filename) as f:
        word_map = json.load(f)

    with open('be.txt', 'r') as original_file:
        with open(processed_filename, 'w') as sentences_file:
            with open(removed_words_filename, 'w') as removed_words_file:
                removed_words = []
                lines = strip_trailing_newline(original_file)
                lines = lower_and_replace_characters(lines)
                if split_sent:
                    lines = split_sentences(lines)
                lines = preprocess_sentences(lines, DERIVED_FORM_BLACKLIST, word_map, removed_words)
                lines = remove_short_lines(lines)
                for s in lines:
                    sentences_file.write(s + "\n")
                    removed_words_file.write(' '.join(removed_words) + "\n")
                    removed_words.clear()


def run_preprocessing(word_mapping_filename,
                      processed_filename,
                      removed_words_filename,
                      verbs,
                      adjectives,
                      split_sent):

    if exists_nonempty_file(word_mapping_filename) and \
            exists_nonempty_file(processed_filename) and \
            exists_nonempty_file(removed_words_filename):
        logger.info(f'Corpus already preprocessed for mapping "{word_mapping_filename}", '
                    f'processed corpus "{processed_filename}" and removed words "{removed_words_filename}"')
        return
    else:
        logger.info(f'Processing corpus for mapping "{word_mapping_filename}", '
                    f'processed corpus "{processed_filename}" and removed words "{removed_words_filename}"')

    if not corpus_downloaded():
        download_corpus()

    if not grammar_downloaded():
        download_grammar()

    if not exists_nonempty_file(word_mapping_filename):
        generate_word_mapping(word_mapping_filename, verbs, adjectives)
    process_corpus(word_mapping_filename, processed_filename, removed_words_filename, split_sent)


if __name__ == "__main__":
    run_preprocessing(word_mapping_filename='word-map.json',
                      processed_filename='processed-corpus.txt',
                      removed_words_filename='removed-words.txt',
                      verbs=True,
                      adjectives=True,
                      split_sent=True)
    run_preprocessing(word_mapping_filename='word-map-only-nouns.json',
                      processed_filename='processed-corpus-only-nouns.txt',
                      removed_words_filename='removed-words-only-nouns.txt',
                      verbs=False,
                      adjectives=False,
                      split_sent=True)
    run_preprocessing(word_mapping_filename='word-map.json',
                      processed_filename='processed-corpus-no-sent-split.txt',
                      removed_words_filename='removed-words-no-sent-split.txt',
                      verbs=True,
                      adjectives=True,
                      split_sent=False)

