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

    def tags(self):
        url = self.base_url + '/api/tags'
        r = requests.get(url,stream=True)
        r.raise_for_status()
        for line in r.iter_lines():
            body = json.loads(line)
            if 'error' in body:
                raise Exception(body['error'])
            models = body.get('models','')
            for model in models:
                yield model
        
    def mod_info(self,model):
        url = self.base_url + '/api/show'
        r = requests.post(url,
                          json={
                              'name' : model
                          },
                          stream=True)
        r.raise_for_status()
        for line in r.iter_lines():
            body = json.loads(line)
            if 'error' in body:
                raise Exception(body['error'])
            yield body

    def pull(self,model):
        url = self.base_url + '/api/pull'
        r = requests.post(url,
                         json={
                             'name': model
                         },
                         stream=True)
        r.raise_for_status()
        for line in r.iter_lines():
            body = json.loads(line)
            if 'error' in body:
                raise Exception(body['error'])
            yield body

    def delete(self,model):
        url = self.base_url + '/api/delete'
        r = requests.delete(url,
                         json={
                             'name': model
                         },
                         stream=True)
        r.raise_for_status()    

from traitlets import Int, Unicode, Bool
from traitlets.config.loader import PyFileConfigLoader
import os

from pprint import pformat

from datetime import datetime, date, time, timezone, timedelta

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

    width = Int(80,help="Line width of output").tag(config=True)

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


    def stream(self,text,stream='stdout'):
            stream_content = {'name': stream, 'text': text}
            self.send_response(self.iopub_socket, 'stream', stream_content)

    def handle_host_magic(self,args):
        if args:
            host = args.strip()
            host = host.split(':')
            self.hostname = host[0]
            if(len(host) > 1):
                self.port = host[1]
            self.base_url = 'http://' + self.hostname + ':' + str(self.port)
            self.stream('Setting base_url "%s"' % self.base_url)
        else:
            self.stream(self.hostname)

    def handle_model_magic(self,args):
        if args:
            model = args.strip()
            self.model = model
            self.stream('Setting model "%s"' % self.model)
        else:
            self.stream(self.model)

    def handle_width_magic(self,args):
        if args:
            width = args.strip()
            try:
                self.width = int(width)
                self.stream('Set width to "%d"' % self.width)
            except:
                self.stream('Error: Width must be integer','stderr')
        else:
            self.stream(self.width)

    def handle_tags_magic(self,args):
        models = self.client.tags()
        for model in models:
            if 'name' in model:
                self.stream('Model: %s\n' % model['name'])
            if 'size' in model:
                size = int(model['size'])
                self.stream('Size: %s\n' % '{:,}'.format(size))
            if 'modified_at' in model:
                mod_date = datetime.fromisoformat(model['modified_at'])
                self.stream('Modified: %s\n\n' % mod_date.strftime('%a, %d %b %Y - %H:%M:%S %Z'))

    def handle_show_magic(self,args):
        if args:
            model = args.strip()
            try:
                mod_infos = self.client.mod_info(model)
                for mi in mod_infos:
                    for n in ['modelfile','license']:
                        N = n.capitalize()
                        if n in mi:
                            self.stream(N + ':\n')
                            self.stream(mi[n] + '\n')
            except Exception as e:
                self.stream(pformat(e) + '\n','stderr')

    def handle_pull_magic(self,args):
        if args:
            model = args.strip()
            try:
                self.stream('Pulling model "%s"\n' % model)
                res = self.client.pull(model)
                b_0 = 0
                t_0 = 0
                perc_compl = 0
                for status in res:
                    if 'total' in status and 'completed' in status:
                        total = status['total']
                        b = status['completed']
                        t = datetime.now()
                        if t_0 == 0:
                            b_0 = b
                            t_0 = t
                        p = (b-b_0)/total
                        last_perc_compl = perc_compl
                        perc_compl = 100*(b/total)
                        if perc_compl - last_perc_compl >= 0.1:
                            if p < 0 or p > 1:
                                p = 1
                            delta_t = t - t_0
                            self.stream('\rPercent completed: ' + ('%3.1F' % perc_compl) + '%')
                            if p > 0:
                                t_remaining = delta_t * (1-p)/p
                                # Round to seconds
                                seconds_remaining = round(t_remaining.total_seconds())
                                t_remaining = timedelta(seconds=seconds_remaining)
                                self.stream(' -- Est. time remaining: ' + str(t_remaining))
                self.stream('\n')
            except Exception as e:
                self.stream(pformat(e) + '\n','stderr')

    def handle_delete_magic(self,args):
        if args:
            model = args.strip()
            self.stream('Deleting model "%s"\n' % model)
            try:
                res = self.client.delete(model)
            except Exception as e:
                self.stream(pformat(e) + '\n','stderr')


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
        elif magic == '%%width':
            self.handle_width_magic(args)
        elif magic in ['%%tags','%%models']:
            self.handle_tags_magic(args)
        elif magic in ['%%show','%%info']:
            self.handle_show_magic(args)
        elif magic == '%%pull':
            self.handle_pull_magic(args)
        elif magic in ['%%delete','%%remove','%%erase']:
            self.handle_delete_magic(args)
            
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

        if not self.client:
            self.client = OllamaClient(base_url=self.base_url,
                                       model=self.model)

        prompt = self.filter_magics(text)
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
                self.stream(pformat(e) + '\n','stderr')
                self.stream("Something went wrong. Have you set host adress and model correctly?",'stderr')

        return {'status': 'ok',
                # The base class increments the execution count
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {},
               }

    def clear_output(self):
        self.current_line = ''

    current_line = ''

    def wrapped_out(self,fragment):
        if fragment == '\n':
            # if len(self.current_line) > 0:
            self.stream('\r' + self.current_line + '\n')
            self.current_line = ''
        else: 
            self.current_line += fragment
            wrapped = textwrap.wrap(self.current_line,width=self.width)
            n = len(wrapped)
            if n > 0:
                wrapped[0] = '\r' + wrapped[0]
                for i in range(n):
                    if i < n-1:
                        self.stream(wrapped[i] + '\n')
                    else:
                        self.stream(wrapped[i].ljust(self.width))
            if n > 1:
                self.current_line = wrapped[n-1]

