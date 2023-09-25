import sys
import pickle
import pdb
from rdkit import Chem
sys.path.append("../")
from private import *


with open('epoch_grammar_4_0.5804753739695949.pkl', 'rb') as f:
    grammar = pickle.load(f)
    for rule_id, rule in enumerate(grammar.prod_rule_list):
        lhs = rule.lhs
        rhs = rule.rhs

        mol_lhs = hg_to_mol_viz(lhs)
        sml_lhs = Chem.MolToSmiles(mol_lhs)

        mol_rhs = hg_to_mol_viz(rhs)
        sml_rhs = Chem.MolToSmiles(mol_rhs)
        
        print("rule: ", sml_lhs, " -> ", sml_rhs)
    
pdb.set_trace()