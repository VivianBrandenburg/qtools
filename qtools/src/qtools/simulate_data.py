#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toolbox to mutate new sequences along a given tree
"""

from Bio import Phylo
import json
from copy import deepcopy
import matplotlib.pyplot as plt 
import random


class Seqs():

    alphabet = ['A', 'G', 'C', 'U']
    
    @classmethod
    def random_seq(cls, k):
        return ''.join(random.choices(cls.alphabet, k=k))
    
    @classmethod
    def mutate_seq(cls, seq, k):
        seq = list(seq)
        controlseq = [x.lower() for x in seq]
        idx_to_mut = random.sample(range(len(seq)), k=k)
        for site in idx_to_mut:
            mutated_nt = random.choice([x for x in cls.alphabet if x != seq[site]])
            seq[site] = mutated_nt 
            controlseq[site] = mutated_nt
        return ''.join(seq), ''.join(controlseq)
        



 
class Mutationscheme(dict):
     
    def _update(self, SeqID, scheme):
        if SeqID in self.keys():
            print(f"'{SeqID}' was already used, you are overwriting the last generated sequence")
        self.update({SeqID: scheme})
    
    
    
    def write(self, outdir):
        output = {'scheme': self.as_str(),
                  'names': '_'.join(self.keys()) }
        with open(outdir + 'mutationscheme.json', 'w') as outf:
            json.dump(output, outf)
    
    def as_str(self):
        return '_'.join(self.values())
        
    





class SimTree():
    
   
    mutationscheme = Mutationscheme()
    
    def check_if_prepped(self):
        if not hasattr(self.root, 'seqs_dict'):
            self.prep_seqDicts()
    
    def clear_sequences(self):
        self.prep_seqDicts()
        self.mutationscheme = Mutationscheme()

        
    def prep_seqDicts(self):
        for clade in self.find_clades():
            clade.seqs_dict = {}
            clade.control_dict ={}

    def prune_from_list(self, included):
        for clade in self.get_terminals():
            if clade.name not in included:
                self.prune(clade)
    
    
    
    # =========================================================================
    # functions for walking the tree 
    # =========================================================================
    
    
    
    
    def simulate_evolving(self, SeqID, seq_len, number_mutations):
        """
        Generates a random sequence and mutates it along the tree

        Parameters
        ----------
        SeqID : str. ID to write the sequence to the seq_dict and control_dict. 
        seq_len : int. length of the generated sequence
        number_mutations : int. number of random mutatios  to insert on each node

        Returns:
        -------
        None.
        
        Sequences are written to seq_dict and control_dict at each clade/node 

        """
        root = self.root
        self.check_if_prepped()
        root.seqs_dict.update({SeqID: Seqs.random_seq(seq_len)})
        root.control_dict.update({SeqID: root.seqs_dict[SeqID].lower()})
        self.mutate_along_tree(SeqID, root, number_mutations)
        self.mutationscheme._update(SeqID, f'n{seq_len}m{number_mutations}')
    
    
    
    def mutate_along_tree(self, SeqID, node, number_mutations):
        children = node.clades
        for child in children:
            mutated_seqs = Seqs.mutate_seq(node.seqs_dict[SeqID], 
                                           number_mutations)
            child.seqs_dict.update({SeqID: mutated_seqs[0]})
            child.control_dict.update({SeqID:mutated_seqs[1]})
            if not child.is_terminal():
                self.mutate_along_tree(SeqID, child, number_mutations)
    
    
    
    def simulate_stable(self, SeqID, seq_len):
        """
        Generates a new random sequence for each node

        Parameters
        ----------
        SeqID : str. ID to write the sequence to the seq_dict and control_dict. 
        seq_len : int. length of the generated sequence

        Returns:
        -------
        None.
        
        Sequences are written to seq_dict and control_dict at each clade/node 
        
        """
        randomseq = Seqs.random_seq(seq_len)
        self.check_if_prepped()
        for clade in self.find_clades():
            clade.seqs_dict.update({SeqID: randomseq})
            clade.control_dict.update({SeqID: 's'*len(randomseq)})
        self.mutationscheme._update(SeqID, f's{seq_len}')
        
        
    def simulate_random(self, SeqID, seq_len):
        """
        Generates one random sequence at the root and passes it to each node without changes 

        Parameters
        ----------
        SeqID : str. ID to write the sequence to the seq_dict and control_dict. 
        seq_len : int. length of the generated sequence

        Returns:
        -------
        None.
        
        Sequences are written to seq_dict and control_dict at each clade/node 

        """
        self.check_if_prepped()
        for clade in self.find_clades():
            randomseq = Seqs.random_seq(seq_len)
            clade.seqs_dict.update({SeqID: randomseq})
            clade.control_dict.update({SeqID : 'r'*len(randomseq)})
        self.mutationscheme._update(SeqID, f'r{seq_len}')
    
        
    # =========================================================================
    # functions for writing output 
    # =========================================================================
    
    def plot(self, filename, tree=None):
        plt.rcParams["figure.figsize"] = (5,5)
        if tree == None: 
            tree = self
        drawing = Phylo.draw(tree, do_show=False)
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()


    def write_as_tree(self, outdir):
        """
        Writes newick trees and plots trees with sequences and control sequences as node names 

        Parameters
        ----------
        outdir : str. directory to write to. filenames are appended automatically
        
        """
        
        outtree = deepcopy(self)
        for key in outtree.root.control_dict.keys():
            
            # write single sequences as tree     
            for clade in outtree.find_clades():
                val = clade.seqs_dict[key]
                clade.name=val
            Phylo.write(outtree, outdir+key+'_seq.phy',  'newick')
            self.plot(outdir+key+'_seq.png', outtree)
            
            # write single controlsequences as tree
            for clade in outtree.find_clades():
                val = clade.control_dict[key]
                clade.name=val
            Phylo.write(outtree, outdir+key+'_controlseq.phy',  'newick')
            self.plot(outdir+key+'_controlseq.png', outtree)

    
        # write fused sequences as tree 
        for clade in outtree.find_clades(): 
            val = ''.join(clade.seqs_dict.values())
            clade.name=val
        Phylo.write(outtree, outdir+'fused_seq.phy',  'newick')
        self.plot(outdir+'fused_seq.png', outtree)

            
        # write fused controlsequences as tree 
        for clade in outtree.find_clades(): 
            val = ''.join(clade.control_dict.values())
            clade.name=val
        Phylo.write(outtree, outdir+'fused_controlseq.phy',  'newick')
        self.plot(outdir+'fused_controlseq.png', outtree)
 
        

    
    
    def write_as_csv(self, outdir):
        """
        Writes sequences and control sequences of all taxa (leafs, terminal nodes) to a csv file. 
        Additionally writes current mutation scheme for metadata 

        Parameters
        ----------
        outdir : str. directory to write to. filenames are appended automatically

        """
            
        with open(outdir + 'sequences.csv', 'w') as outf: 
            keys = self.root.seqs_dict.keys()
            header_seqs = ['seq_' + x for x in keys]
            header_controls = ['control_' + x for x in keys]
            outf.write(','.join(['spec', 'seq']+header_seqs+header_controls)+'\n')
            
            for leaf in self.get_terminals(): 
                values = [leaf.name, ''.join(leaf.seqs_dict.values())]
                values += leaf.seqs_dict.values()  
                values += leaf.control_dict.values()
                outf.write(','.join(values)+'\n')
                
        # write the mutation scheme to same path
        self.mutationscheme.write(outdir)
            

     
 
class PatchedTree(Phylo.Newick.Tree, SimTree):
    pass
        


def patch_tree(tree):
    tree.__class__ = PatchedTree
    tree.mutationscheme = Mutationscheme()
    return tree
