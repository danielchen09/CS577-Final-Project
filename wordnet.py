import random
import subprocess
import re


class WordNet:
    def __init__(self, path='./Wordnet/bin/wn.exe', stopword_path='./data/stopwords.txt'):
        self.path = path
        with open(stopword_path, 'r') as f:
            self.stopwords = set(f.readlines())

    def get_random_synonym(self, word):
        syns = self.get_synonyms(word)
        if len(syns) == 0:
            return word
        r = random.randint(0, len(syns) - 1)
        return syns[r]

    def get_synonyms(self, word):
        syns = self._get_sense(word, 1)
        s = []
        for synonym in syns:
            synonym = re.sub('\(.*\)', '', synonym).strip()
            if synonym != word:
                s.append(synonym)
        return s

    def _get_sense(self, word, sense):
        subp_out = self._call_subprocess(word, '-synsa').split('\r\n')
        for i, line in enumerate(subp_out):
            if line == f'Sense {sense}':
                return subp_out[i + 1].split(', ')
        return []

    def _call_subprocess(self, *args):
        return subprocess.run([self.path, *args], stdout=subprocess.PIPE).stdout.decode('utf-8')