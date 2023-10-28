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
        
        for line in r.iter_lines():
            body = json.loads(line)
            if 'error' in body:
                raise Exception(body['error'])
            response_part = body.get('response', '')
            result.append(response_part)
            if body.get('done', False):
                if 'context' in body:
                    self.context = body['context']
                return ''.join(result)

        

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

    port = 11434

    client = None
    hostname = 'localhost'
    
    model = 'llama2'
    base_url = 'http://localhost:11434'

    client_changed = True
    
    def out(self,text,stream='stdout'):
            stream_content = {'name': 'stdout', 'text': text}
            self.send_response(self.iopub_socket, 'stream', stream_content)

    def handle_magic(self,magic):
        if magic.startswith('%%host:'):
            host = magic.removeprefix('%%host:').strip()
            host = host.split(':')
            self.hostname = host[0]
            if(len(host) > 1):
                self.port = host[1]
            self.base_url = 'http://' + self.hostname + ':' + str(self.port)
            self.out('Setting base_url "%s"' % self.base_url)
        elif magic.startswith('%%model:'):
            self.model = magic.removeprefix('%%model:').strip()
            self.out('Setting model "%s"' % self.model)
            
    def filter_magics(self,input):
        lines = input.split('\n')
        magics = []
        prompt = []
        for l in lines:
            if l.startswith('%%'):
                magics.append(l)
            else:
                prompt.append(l)
        if len(magics) > 0:
            for m in magics:
                self.handle_magic(m)
            self.ollama_changed = True
                
        prompt = '\n'.join(prompt)
        return(prompt)
            
    def do_execute(self, input, silent, store_history=True, user_expressions=None,
                   allow_stdin=False):

        prompt = self.filter_magics(input)
        if not self.client:
            self.client = OllamaClient(base_url=self.base_url,
                                       model=self.model)
        if self.client_changed:
            self.client.base_url = self.base_url
            self.client.model = self.model
        
        self.client_changed = False
        # errored = False
        # try:

        output = self.client.generate(prompt)
        paragraphs = output.split('\n')
        paragraphs = [textwrap.fill(para) for para in paragraphs]
        paragraphs[0] = paragraphs[0].lstrip(' ')
        result = '\n\n'.join(paragraphs)
        result = result.replace('\n\n\n','\n')

        # except:
        #     result = "Something went wrong. Have you set host adress and model correctly?"
        #     errored = True
        errored = False
        
        if errored:
            stream = 'stderr'
        else:
            stream = 'stdout'
            
        if not silent:
            self.out(result,stream)

        return {'status': 'ok',
                # The base class increments the execution count
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {},
               }

