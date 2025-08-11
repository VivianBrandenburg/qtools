#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import json
import seaborn as sns



class DefaultTracking():
    """Expects to get a dict with a value for each epoch in form 
    {measure1:[val1, val2, ...], measure2:[val1, val2, ...], ...}
    """
    values = None

 
    
    def set_defaults(self, outdir):
        self._outdir = outdir
        self._outfile = outdir + self.filename


    
    def track(self, val):
        self.values = val
        
    def plot(self, to_file=False, plot_live=True):
        for k,v in self.values.items():
            plt.plot(range(len(v)), v, label=k)
        plt.legend()    
        plt.xlabel('epoch')
        plt.ylabel('value')
        if to_file: plt.savefig(self._outfile+'.svg', bbox_inches='tight')
        plt.show() if plot_live else plt.close()    
        
    def write(self):
        outfile = self._outfile + '.json'
        with open(outfile, 'w') as outf:
            json.dump(self.values, outf)
 
    
      
        
# =============================================================================
# measure classes 
# =============================================================================

class Epoche(DefaultTracking): 
    filename = 'epochs'
    
    def __init__(self):
        self.values = []
    
    def track(self, value):
        self.values.append(value)
    
    def plot(self, to_file=False, plot_live=True):
        plt.plot(range(len(self.values)), self.values)
        if to_file: plt.savefig(self._outfile+'.svg', bbox_inches='tight')
        plt.show() if plot_live else plt.close()
    

    



class FeatureVectors(DefaultTracking):    
    filename = 'featureVectors'
    values = {}
    
    def __init__(self, y_names):
        self.y_names = y_names
    
    def track(self,epoch,  val):
        self.values.update({epoch:val})
    
    def select_epoche(self, epoche):
        # case handling if int is given
        if type(epoche) == int: 
            epoche = [epoche]
        # case handling to plot all epochs
        if epoche is None: 
           epoche = self.values.keys()
        # case handling if index -1 was given 
        epoche = [max(self.values.keys()) if x==-1 else x for x in epoche]
        return epoche
    
    
    def plot(self, epoch=None, to_file=False, plot_live=True):
        """
        plot feature vectors as heat map

        Parameters
        ----------
        epoche : int or list of ints, optional
            If None, plot all epochs. The default is -1.
        filename : filename, optional
            If given, plot will be written to outfile. The default is False.

        """
        epoch = self.select_epoche(epoch)
        for e in epoch:
            assert e in self.values.keys(), f'Epoch {e} is missing in the track records and can not be plotted.'
            
            val = self.values[e]
            hm = sns.heatmap(val, cmap='magma_r', mask=(val==0))
            hm.set_yticklabels(self.y_names, rotation=0)
            hm.set_title(f'epoch {e}')
            if to_file: plt.savefig(self._outfile+f'_{e}.svg', bbox_inches='tight')
            plt.show() if plot_live else plt.close()
    
    def write(self):
        value_dump = {k:v.tolist() for k,v in self.values.items()}
        outfile = self._outfile + '.json'
        with open(outfile, 'w') as outf:
            json.dump(value_dump, outf)
 
    

class Scores(DefaultTracking):
    
    def __init__(self):
        self.filename= 'scores'  
        self.values = {'opt':[], 'subopt':[], 'wrong':[]}
    
    def track(self, val):
        for track_v, new_v in zip(self.values.values(), val):
            track_v.append(new_v)
    
  
class Losses(DefaultTracking):
    filename = 'losses' 
    values = None
  
    def track(self, losses):
        """
        expects a dict with {lossname:[loss_values]} or {lossname:lossvalue}
        """
        # compatible with list and float format 
        losses = {k:[v] if type(v)!=list else v for k,v in losses.items()}
        
        # if this is first run we have to initialize the dictionary 
        if not self.values:
            self.values = losses 
        
        # if this is not the first run we append the values
        else: 
            for k,v in losses.items():
                self.values[k].append(v[-1])
             


# =============================================================================
# wrapper 
# =============================================================================
        
class Tracking():
    
    def __init__(self, outdir, y_names):  
        self.y_names = y_names
        
        self.epoche = Epoche()
        self.feature_vectors = FeatureVectors(y_names)
        self.losses = Losses()
        self.scores = Scores()
        
        self.elements = [self.epoche, self.feature_vectors, 
                        self.losses, self.scores]
        self.outdir = outdir
        self.set_defaults()
        self.write_species_names()
    
    def set_defaults(self):
        for el in self.elements:
            el.set_defaults(self.outdir)
    
    def trackall(self, epoch=None, feature_vectors = None, 
                 loss = None, score = None):
        
        self.epoche.track(epoch)
        self.feature_vectors.track(epoch, feature_vectors)
        self.scores.track(score)
        self.losses.track( loss)

    def printall(self):
        print('\nepochs:\n', self.epoche.values)
        print('\nfeature_vectors:\n', self.feature_vectors.values)
        print('\nlosses:\n', self.losses.values)
        print('\nscores:\n', self.scores.values)
        
    def plotall(self,epoch=None, to_file=False, plot_live=True):
        self.feature_vectors.plot(epoch=epoch, to_file=to_file, plot_live=plot_live)
        self.losses.plot(to_file=to_file, plot_live=plot_live)
        self.scores.plot(to_file=to_file, plot_live=plot_live)
    
    def writeall(self):
        for el in self.elements:
            el.write()
    
    def write_species_names(self):
        outfile = self.outdir + 'y_labels.json'
        with open(outfile, 'w') as outf:
            json.dump({'y_labels': self.y_names}, outf)
            
 
        
        
    
    
    




# # =============================================================================
# # testing 
# # =============================================================================
# loss_labels = ['loss1', '2', '3']
# score_labels = ['score', '2', '3']
# y_labels=['y1', 'y2']
# t = Tracking('testing_trackingWrite/', y_labels)


# # testing stacking of feature vectors
# myscores =  {'best': [0.3,1], 'subopt': [0.4,2], 'wrong': [0.2,3]}
# loss =  {'best': [0.3,0], 'subopt': [0.4,1], 'wrong': [0.2,0]}


# vector =  np.array([[1,2,3,4,5 ], [1,2,1,2,1]])
# vector2 = np.array([[2,2,2,2,2 ], [4,5,6,7,8]])

# t.trackall(epoch=0, feature_vectors=vector, loss=loss, score=myscores)

# t.printall()
# t.plotall()
# t.writeall()
