from ipykernel.kernelbase import Kernel
import textwrap

import logging

logger = logging.getLogger(__name__)

import json
import requests

class OllamaClient(object):

    base_url = ''
    model = ''
    context = []
    
    def __init__(self,
                 base_url='http://localhost:11434',
                 model='llama2'):
        self.base_url = base_url
        self.model = model

    def generate(self,prompt):

        url = self.base_url + '/api/generate'
        
        r = requests.post(url,
                          json={
                              'model': self.model,
                              'prompt': prompt,
                              'context': self.context,
                          },
                          stream=True)
        r.raise_for_status()

        result = []
        
        first = True

        for line in r.iter_lines():
            body = json.loads(line)
            if 'error' in body:
                raise Exception(body['error'])
            response_part = body.get('response', '')
            if first:
                response_part = response_part.lstrip()
                first = False
            if body.get('done', False):
                if 'context' in body:
                    self.context = body['context']
            yield response_part

from traitlets import Int, Unicode
from traitlets.config.loader import PyFileConfigLoader
import os

from pprint import pformat

class OllamaKernel(Kernel):

    implementation = 'Ollama'
    implementation_version = '1.0'
    language = 'no-op'
    language_version = '0.1'
    language_info = {
        'name': 'ollama',
        'mimetype': 'text/plain',
        'file_extension': '.txt',
    }
    banner = "A kernel for Ollama (a LLM front end)"

    port = Int(11434,help="Port the ollama server listens on").tag(config=True)

    client = None
    hostname = Unicode('localhost',help="Name of the ollama server runs").tag(config=True)
    
    model = Unicode('llama2',help="Name of the model that ollama is to use").tag(config=True)
    base_url = 'http://localhost:11434'

    client_changed = True
    
    config_loaded = False
    config_file = Unicode('').tag(config=True)
    default_config_file = 'ollama_kernel_config.py'


    def load_config_file(self,filename):
        loader = PyFileConfigLoader(filename)
        config = loader.load_config()
        self.update_config(config)

    def load_config(self):
        if not len(self.config_file) > 0:
            self.config_file = self.default_config_file
        paths = [os.path.join("etc","jupyter"),
                 os.path.expanduser(os.path.join('~','.jupyter')),
                 os.getcwd()]
        filenames = [os.path.join(path,self.config_file) for path in paths]
        for fn in filenames:
            self._load_cfg_(fn)

    def _load_cfg_(self,filename):
        if os.path.exists(filename):
            self.load_config_file(filename)


    def out(self,text,stream='stdout'):
            stream_content = {'name': 'stdout', 'text': text}
            self.send_response(self.iopub_socket, 'stream', stream_content)

    def handle_host_magic(self,args):
        if args:
            host = args.strip()
            host = host.split(':')
            self.hostname = host[0]
            if(len(host) > 1):
                self.port = host[1]
            self.base_url = 'http://' + self.hostname + ':' + str(self.port)
            self.out('Setting base_url "%s"' % self.base_url)
        else:
            self.out(self.hostname)

    def handle_model_magic(self,args):
        if args:
            model = args.strip()
            self.model = model
            self.out('Setting model "%s"' % self.model)
        else:
            self.out(self.model)

    def handle_magic(self,magic_line):
        splt = magic_line.split(maxsplit=1)
        magic = splt[0]
        if len(splt) > 1:
            args = splt[1]
        else:
            args = None
        if magic in ['%%hostname','%%host']:
            self.handle_host_magic(args)
        elif magic == '%%model':
            self.handle_model_magic(args)
            
    def filter_magics(self,text):
        lines = text.split('\n')
        magic_lines = []
        prompt = []
        for l in lines:
            if l.startswith('%%'):
                magic_lines.append(l)
            else:
                prompt.append(l)
        if len(magic_lines) > 0:
            for m in magic_lines:
                self.handle_magic(m)
            self.client_changed = True
                
        prompt = '\n'.join(prompt)
        return(prompt)
            
    def do_execute(self, text, silent, store_history=True, user_expressions=None,
                   allow_stdin=False):

        if not self.config_loaded:
            self.load_config()
            self.config_loaded = True
            self.base_url = 'http://' + self.hostname + ':' + str(self.port)

        prompt = self.filter_magics(text)
        if not self.client:
            self.client = OllamaClient(base_url=self.base_url,
                                       model=self.model)
        if self.client_changed:
            self.client.base_url = self.base_url
            self.client.model = self.model

        self.client_changed = False
        errored = False

        if len(prompt) > 0:
            try:
                results = self.client.generate(prompt)
                self.clear_output()
                for r in results:
                    self.wrapped_out(r)

            except Exception as e:
                self.out(pformat(e) + '\n','stderr')
                self.out("Something went wrong. Have you set host adress and model correctly?",'stderr')

        return {'status': 'ok',
                # The base class increments the execution count
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {},
               }

    def clear_output(self):
        self.current_line = ''

    current_line = ''

    width = Int(70,help="Line width of output").tag(config=True)

    def wrapped_out(self,fragment):
        if fragment == '\n':
            if len(self.current_line) > 0:
                self.out('\r' + self.current_line + '\n\n')
            self.current_line = ''
        else: 
            self.current_line += fragment
            wrapped = textwrap.wrap(self.current_line)
            wrapped[0] = '\r' + wrapped[0]
            n = len(wrapped)
            for i in range(n):
                if i < n-1:
                    self.out(wrapped[i] + '\n')
                else:
                    self.out(wrapped[i])
            if n > 1:
                self.current_line = wrapped[n-1]
