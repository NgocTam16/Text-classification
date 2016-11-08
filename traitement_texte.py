#!/usr/bin/env python
# -*- coding: utf8 -*-

# Traitement de texte en python : quelques outils

import re
import string
import numpy as np
import sys  
import string
import importlib
importlib.reload (sys)

import unicodedata 

from nltk.corpus import stopwords

from stop_words import get_stop_words

stop_words = get_stop_words('english')

# Quelques fonctions de nettoyage
def remove_punct(s):
	lettres = [l for l in s if l not in string.punctuation]
	return ''.join(lettres)

def remove_digit(s):
  result = ''.join([l for l in s if not l.isdigit()])
  return result

def lower_string(s):
  return s.lower()

def remove_accents(s):
  nkfd_form = unicodedata.normalize('NFKD', unicode(s))
  return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])

def replace_url(s):
  match_urls = re.compile(r"""((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.‌​][a-z]{2,4}/)(?:[^\s()<>]+|(([^\s()<>]+|(([^\s()<>]+)))*))+(?:(([^\s()<>]+|(‌​([^\s()<>]+)))*)|[^\s`!()[]{};:'".,<>?«»“”‘’]))""", re.DOTALL) 
  return re.sub(match_urls, '', s)

def remove_stopwords(s):
  return ' '.join([i for i in s.split() if i not in stop_words])



