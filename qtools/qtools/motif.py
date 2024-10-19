#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class motif():
    
    import numpy as np
    import pandas as pd
    import logomaker
    import seaborn as sns
    
    
    

    alphabet = ['A', 'C', 'G', 'T']
    
        
    colorscheme = {
        'G': [0.004, 0.125, 0.305],
        'TU': [0.007, 0.514, 0.569],
        'C': [0.965, 0.863, 0.675],
        'A': [0.996, 0.68, 0.44]
    }
    
    
    mcol = '#333333'
    
    sns.set_style(rc={'axes.edgecolor': mcol,
                      'axes.labelcolor': mcol,
                      'xtick.color': mcol,
                      'ytick.color':mcol,
                      'lines.color':mcol,
                    'axes.spines.right': False,
                    'axes.spines.top': False, 
                    'font.family':'sans-serif',
                    'font.sans-serif': 'DejaVu Sans',
                    })
    
    
    def __init__(self, sequences):
        
        self.sequences = [s.upper().replace('U','T') for s in sequences]
        
        self.counts = self.__get_counts(self.sequences)
        self.frequencies = self.__get_frequency(self.counts)
        self.bits = self.__calc_bits(self.counts, self.frequencies)
        
        
    
    def __empty(self):
        return {k:[] for k in self.alphabet}
               
    
            
    def __get_counts(self, seqs):
        """ count absolute occurences for nt in each position of a multiple sequence alignment
        """
        counts = self.__empty()
        for i in range(len(seqs[0])):
            for nt, value in counts.items():
                position = [s[i] for s in seqs]
                value.append(position.count(nt))
        return counts
    
    def __get_frequency(self, counts):
        """ Get realitive frequency of each nt for each position from counts of occurence
        """
        n_seq = len(self.sequences)
        freq = {k:[x/n_seq for x in v] for k,v in counts.items()}
        return freq
    
        
    def R_seq(self, counts):
        """ calculate coseravtion from absolut counts for one position. 
        Equation from weblogo, https://doi.org/10.1101/gr.849004 
        """
        s_max = np.log2(len(counts))
        csum = sum(counts)
        s_observed = [(c/csum)*np.log2(c/csum) for c in counts if c != 0 ]
        return s_max + sum(s_observed)
    
    def __calc_bits(self, counts, frequency):
        """
        """
        # get position-wise counts instead of nucleotide-wise
        counts_transposed = [x for x in zip(*counts.values())]
        # get conservation for each position
        conservation = [self.R_seq(c) for c in counts_transposed]
        # multiply conservation with relative frequency (as in weblogo)
        conversation_frequency = {k:np.multiply(freq,conservation) 
                                  for k,freq in frequency.items()}
        return conversation_frequency
    
    def logo(self):
        
        bits_df = pd.DataFrame(self.bits)
        self.make_logo(bits_df)
        
    @classmethod
    def make_logo(cls, df):
        
        width = len(df) 
        scaling_factor = 9
        figsize = (width/scaling_factor, 6/scaling_factor)
        
        figsize = (4.17,1.08 )
        
        m_logo = logomaker.Logo(df, font_name='Arial Rounded MT Bold',
                                color_scheme=cls.colorscheme,
                                figsize = figsize            )
        # style axes
        m_logo.style_spines(visible=False)
        m_logo.style_spines(spines=['left', 'bottom'], visible=True, color=cls.mcol)
        m_logo.ax.set_ylabel('bits')
    
        
          
