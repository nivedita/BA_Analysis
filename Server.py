import logging
import sys
import SocketServer

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import mdp
import csv
import Oger


logging.basicConfig(level=logging.DEBUG,
                    format='%(name)s: %(message)s',
                    )


class MyTCPHandler(SocketServer.StreamRequestHandler):
    
    
    def setFlow(self, flow):
        self.flow = flow

    def handle(self):
        # self.rfile is a file-like object created by the handler;
        # we can now use e.g. readline() instead of raw recv() calls
        self.data = self.rfile.readline().strip()
        while(self.data):
            print "{} wrote:".format(self.client_address[0])
            print self.data
            a = np.array(list(self.data.split(';')), dtype=float)

        # Likewise, self.wfile is a file-like object used to write back
        # to the client


class EchoServer(SocketServer.TCPServer):
    
    def __init__(self, server_address, handler_class=MyTCPHandler):
        self.logger = logging.getLogger('EchoServer')
        self.logger.debug('__init__')
        SocketServer.TCPServer.__init__(self, server_address, handler_class)
        
        return

    def server_activate(self):
        self.logger.debug('server_activate')
        SocketServer.TCPServer.server_activate(self)
        return

    def serve_forever(self):
        self.logger.debug('waiting for request')
        self.logger.info('Handling requests, press <Ctrl-C> to quit')
        while True:
            self.handle_request()
        return

    def handle_request(self):
        self.logger.debug('handle_request')
        return SocketServer.TCPServer.handle_request(self)

    def verify_request(self, request, client_address):
        self.logger.debug('verify_request(%s, %s)', request, client_address)
        return SocketServer.TCPServer.verify_request(self, request, client_address)

    def process_request(self, request, client_address):
        self.logger.debug('process_request(%s, %s)', request, client_address)
        return SocketServer.TCPServer.process_request(self, request, client_address)

    def server_close(self):
        self.logger.debug('server_close')
        return SocketServer.TCPServer.server_close(self)

    def finish_request(self, request, client_address):
        self.logger.debug('finish_request(%s, %s)', request, client_address)
        return SocketServer.TCPServer.finish_request(self, request, client_address)

    def close_request(self, request_address):
        self.logger.debug('close_request(%s)', request_address)
        return SocketServer.TCPServer.close_request(self, request_address)
    
    

if __name__ == '__main__':
    import socket
    import threading

    address = ('192.168.0.115', 11111) # let the kernel give us a port
    server = EchoServer(address, MyTCPHandler)
    ip, port = server.server_address # find out what port we were given

    #t = threading.Thread(target=server.serve_forever)
    #t.setDaemon(True) # don't hang on exit
    #t.start()

    logger = logging.getLogger('client')
    logger.info('Server on %s:%s', ip, port)