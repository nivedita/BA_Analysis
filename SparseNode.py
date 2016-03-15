import numpy as np

from Oger.nodes.reservoir_nodes import ReservoirNode


class SparseNode(ReservoirNode):
	useSparse = False
	connectivity = 0.2
	
	def __init__(self, input_dim=None, output_dim=None, spectral_radius=0.9, \
				 nonlin_func=np.tanh, reset_states=True, bias_scaling=0, input_scaling=1, dtype='float64', _instance=0, \
				 w_in=None, w=None, w_bias=None):
		super(SparseNode, self).__init__(input_dim, output_dim, spectral_radius, \
				 nonlin_func, reset_states, bias_scaling, input_scaling, dtype, _instance, \
				 w_in, w, w_bias)
		
		
	def initialize(self):
		ReservoirNode.initialize(self)
		if self.useSparse:
				self.w_in = np.ones((self.output_dim,self.input_dim))*self.input_scaling
				rand = np.random.random(self.w_in.shape)<1-self.connectivity
				self.w_in[rand]= 0
				#print 'using' +str(self.input_dim) + '  ' + str(self.output_dim) + '  '+str(self.input_scaling)
				for i in range(self.input_dim):
					self.w_in[np.random.randint(self.output_dim),i]=self.input_scaling
