from os import path
from .vocabulary import Vocabulary
from .language_util import LanguageUtil
from tree_sitter import Language, Parser
from pathlib import Path
import glob, os
import numpy as np
import logging
import urllib.request
from urllib3.exceptions import InsecureRequestWarning
from tqdm import tqdm
import zipfile
import shutil
import platform

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
           ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

class ASTParser():
    import logging
    LOGGER = logging.getLogger('ASTParser')
    def __init__(self, language=None):
        # ------------ To initialize for the treesitter parser ------------
        home = str(Path.home())
        cd = os.getcwd()
        plat = platform.system()     
        p = path.join(home, ".tree-sitter", "bin")
        os.chdir(p)
        self.Languages = {}
        for file in glob.glob("*.so"):
          try:
            lang = os.path.splitext(file)[0]
            self.Languages[lang] = Language(path.join(p, file), lang)
          except:
            print("An exception occurred to {}".format(lang))
        os.chdir(cd)
        self.parser = Parser()
        
        self.language = language
        if self.language == None:
            self.LOGGER.info("Cannot find language configuration, using java parser as the default to parse the code into AST")
            self.language = "java"

        lang = self.Languages.get(self.language)
        self.parser.set_language(lang)
        # -----------------------------------------------------------------

    def get_ast_size(self, tree):
        # tree = self.ast_parser.parse(text)
        root = tree.root_node
        num_nodes = 0
        root_type = str(root.type)
        queue = [root]

        while queue:
            current_node = queue.pop(0)
            num_nodes += 1
            for child in current_node.children:
                queue.append(child)
               
        return num_nodes

       
    def parse_with_language(self, code_snippet, language):
        lang = self.Languages.get(language)
        self.parser.set_language(lang)
        return self.parser.parse(code_snippet)

    def parse(self, code_snippet):
        return self.parser.parse(code_snippet)
    
    def set_language(self, language):
        lang = self.Languages.get(language)
        self.parser.set_language(lang)

