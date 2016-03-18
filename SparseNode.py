import numpy as np

from Oger.nodes.reservoir_nodes import LeakyReservoirNode


class SparseNode(LeakyReservoirNode):
	useSparse = False
	connectivity = 0.1
	inputSignals = 'FGA'
	

		
	def initialize(self):
		LeakyReservoirNode.initialize(self)
		if self.useSparse:
				self.w_in = np.ones((self.output_dim,self.input_dim))*self.input_scaling
				rand = np.random.random(self.w_in.shape)<1-self.connectivity
				self.w_in[rand]= 0
				#print 'using' +str(self.input_dim) + '  ' + str(self.output_dim) + '  '+str(self.input_scaling)
				for i in range(self.input_dim):
					self.w_in[np.random.randint(self.output_dim),i]=self.input_scaling
		if self.inputSignals == 'FGA':
			pass
		elif self.inputSignals == 'FG':
			self.w_in[:,6:9]=0
		elif self.inputSignals == 'FA':
			self.w_in[:,3:6]=0
		elif self.inputSignals == 'GA':
			self.w_in[:,0:3]=0
		elif self.inputSignals == 'F':
			self.w_in[:,3:9]=0
		elif self.inputSignals == 'G':
			self.w_in[:,0:3]=0
			self.w_in[:,6:9]=0
		elif self.inputSignals == 'A':
			self.w_in[:,0:6]=0
			
		
			
		