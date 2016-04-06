import numpy as np

from Oger.nodes.reservoir_nodes import LeakyReservoirNode
from nltk.classify.scikitlearn import SklearnClassifier



class SparseNode(LeakyReservoirNode):

	
	def __init__(self, leak_rate=1, *args, **kwargs):
		LeakyReservoirNode.__init__(self,leak_rate, *args, **kwargs)
		self.useSparse = False
		self.connectivity = 0.1
		self.inputSignals = 'FGA'
		self.useNormalized = 0
		self.colStdFactor = np.array([ 0.19532664, 0.07406439, 0.18426636, 2.57861928,1.19940363,2.51488647,6.37374965,4.49400088,5.75603514])
		self.colMaxFactor = np.array([3.07070231,0.62703943,3.12939386,19.78702,14.564295,20.696224,48.78246,46.557495,49.010956 ])
	
	
	def updateInputScaling(self, dataStep):
		input = np.concatenate([x[0] for x in dataStep],0)
		self.colStdFactor[0:3] = np.std(np.linalg.norm(input[:,0:3], None, 1))
		self.colStdFactor[3:6] = np.std(np.linalg.norm(input[:,3:6], None, 1))
		self.colStdFactor[6:9] = np.std(np.linalg.norm(input[:,6:9], None, 1))

		self.colMaxFactor[0:3] = np.max(np.linalg.norm(input[:,0:3], None, 1))
		self.colMaxFactor[3:6] = np.max(np.linalg.norm(input[:,3:6], None, 1))
		self.colMaxFactor[6:9] = np.max(np.linalg.norm(input[:,6:9], None, 1))
		
		#self.colStdFactor = np.std(input,0)
		print 'Stds: ' + str(self.colStdFactor)
		print 'max Vals: '+str(self.colMaxFactor)
		
	
	def initialize(self):
		LeakyReservoirNode.initialize(self)
		if self.useSparse:
				self.w_in = np.ones((self.output_dim,self.input_dim))*self.input_scaling
				rand = np.random.random(self.w_in.shape)<1-self.connectivity
				self.w_in[rand]= 0
				#print 'using' +str(self.input_dim) + '  ' + str(self.output_dim) + '  '+str(self.input_scaling)
				for i in range(self.input_dim):
					self.w_in[np.random.randint(self.output_dim),i]=self.input_scaling
		if self.useNormalized == 0:
			pass
		elif self.useNormalized == 1:
			self.w_in = self.w_in / self.colStdFactor
		elif self.useNormalized == 2:
			self.w_in = self.w_in / self.colMaxFactor
				
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
		print(self.output_dim)
		print(self.spectral_radius)
			
		
			
		