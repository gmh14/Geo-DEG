import rdkit
import random
import itertools
from rdkit import Chem
from rdkit.Chem import rdFMCS
from collections import defaultdict, deque
from fuseprop.vocab import MAX_VALENCE

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)
    return mol

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None: Chem.Kekulize(mol)
    return mol

def get_smiles(mol, kekuleSmiles=True):
    if kekuleSmiles:
        return Chem.MolToSmiles(mol, kekuleSmiles=True)
    else:
        return Chem.MolToSmiles(mol)

def sanitize(mol, kekulize=True):
    try:
        smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
        mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol

def valence_check(atom, bt):
    cur_val = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
    return cur_val + bt <= MAX_VALENCE[atom.GetSymbol()]

def get_leaves(mol):
    leaf_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetDegree() == 1]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append( set([a1,a2]) )

    rings = [set(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(rings)

    leaf_rings = []
    for r in rings:
        inters = [c for c in clusters if r != c and len(r & c) > 0]
        if len(inters) > 1: continue
        nodes = [i for i in r if mol.GetAtomWithIdx(i).GetDegree() == 2]
        leaf_rings.append( max(nodes) )

    return leaf_atoms + leaf_rings

def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()

def bond_match(mol1, a1, b1, mol2, a2, b2):
    a1,b1 = mol1.GetAtomWithIdx(a1), mol1.GetAtomWithIdx(b1)
    a2,b2 = mol2.GetAtomWithIdx(a2), mol2.GetAtomWithIdx(b2)
    return atom_equal(a1,a2) and atom_equal(b1,b2)

def copy_atom(atom, atommap=True):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    if atommap: 
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

#mol must be RWMol object
def get_sub_mol(mol, sub_atoms):
    new_mol = Chem.RWMol()
    atom_map = {}
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)

    sub_atoms = set(sub_atoms)
    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in sub_atoms: continue
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            if a.GetIdx() < b.GetIdx(): #each bond is enumerated twice
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)

    return new_mol.GetMol()

def find_clusters(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1: #special case
        return [(0,)], [[0]]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append( (a1,a2) )

    ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(ssr)

    atom_cls = [[] for i in range(n_atoms)]
    for i in range(len(clusters)):
        for atom in clusters[i]:
            atom_cls[atom].append(i)

    return clusters, atom_cls

def bfs_select(clusters, atom_cls, start_cls, n_atoms, blocked=[], return_cls=False):
    blocked = set(blocked)
    selected = set()
    selected_atoms = set()
    queue = deque([start_cls]) 

    while len(queue) > 0 and len(selected_atoms) < n_atoms:
        x = queue.popleft()
        selected.add(x)
        selected_atoms.update(clusters[x])
        for a in clusters[x]:
            for y in atom_cls[a]:
                if y in selected or y in blocked: continue
                queue.append(y)

    selected_atoms = [a for cls in selected for a in clusters[cls]]
    selected_atoms = set(selected_atoms)
    if return_cls:
        return selected, selected_atoms
    else:
        return selected_atoms

def random_subgraph(mol, ratio):
    n_atoms = mol.GetNumAtoms()
    clusters, atom_cls = find_clusters(mol)
    start_cls = random.randrange(len(clusters))
    return bfs_select(clusters, atom_cls, start_cls, n_atoms * ratio)

"""
def dual_random_subgraph(mol, n_atoms):
    clusters, atom_cls = find_clusters(mol)
    block_cls = list( range(len(clusters)) )
    random.shuffle(block_cls)

    for k in block_cls:
        blocked = set( [j for a in clusters[k] for j in atom_cls[a]] )
        if len(blocked) == len(clusters): continue
        
        start_cls1 = random.choice( [i for i in range(len(clusters)) if i not in blocked] )
        sg1_cls, sg1_atoms = bfs_select(clusters, atom_cls, start_cls1, n_atoms=1000, blocked=blocked, return_cls=True)
        if len(sg1_atoms) < n_atoms: continue
       
        blocked.update(sg1_cls)
        if len(blocked) == len(clusters): continue

        start_cls2 = random.choice( [i for i in range(len(clusters)) if i not in blocked] )
        sg2_cls, sg2_atoms  = bfs_select(clusters, atom_cls, start_cls2, n_atoms=1000, blocked=blocked, return_cls=True)
        if len(sg2_atoms) < n_atoms: continue

        blocked = set( [j for a in clusters[k] for j in atom_cls[a]] )
        sg1 = bfs_select(clusters, atom_cls, start_cls1, n_atoms, blocked=blocked)
        sg2 = bfs_select(clusters, atom_cls, start_cls2, n_atoms, blocked=blocked)
        return sg1 | sg2
    
    if n_atoms <= 2:
        return None
    else:
        return dual_random_subgraph(mol, int(n_atoms * 0.8))
"""

def dual_random_subgraph(mol, ratio):
    clusters, atom_cls = find_clusters(mol)
    best_size = 0
    best_block_atom = None

    for atom in mol.GetAtoms():
        blocked_cls = set( atom_cls[atom.GetIdx()] )
        blocked_atoms = set( [a for cls in blocked_cls for a in clusters[cls]] )
        if len(blocked_atoms) <= 1: continue

        components = []
        nei_cls = set([cls for a in blocked_atoms for cls in atom_cls[a]]) - blocked_cls
        for start_cls in nei_cls:
            if start_cls in blocked_cls: continue  # blocked_cls is changing
            sg_cls, sg_atoms = bfs_select(clusters, atom_cls, start_cls, n_atoms=1000, blocked=blocked_cls, return_cls=True)
            components.append( (start_cls, sg_cls, sg_atoms) )
            blocked_cls.update(sg_cls)

        if len(components) < 2: continue
        components = sorted(components, key=lambda x:len(x[1]), reverse=True)
        
        if len(components[1][2]) > best_size: # second_component_atoms
            best_size = len(components[1][2])
            best_block_atom = atom.GetIdx()
            best_components = components
    
    if best_block_atom is None:
        return set()

    # recompute with best block atom
    blocked_cls = set( atom_cls[best_block_atom] )
    selected_atoms = set()
    for start_cls, comp_cls, comp_atoms in best_components:
        n_atoms = len(comp_atoms) * ratio
        sg_atoms = bfs_select(clusters, atom_cls, start_cls, n_atoms, blocked=blocked_cls)
        selected_atoms.update(sg_atoms)
        blocked_cls.update(comp_cls)

    return selected_atoms


def enum_subgraph(mol, ratio_list):
    n_atoms = mol.GetNumAtoms()
    clusters, atom_cls = find_clusters(mol)

    selection = []
    for start_cls in range(len(clusters)):
        for ratio in ratio_list:
            x = bfs_select(clusters, atom_cls, start_cls, n_atoms * ratio)
            selection.append(x)
    return selection


def try_extract_subgraph(smiles, selected_atoms, remove_bonds):
    mol = Chem.MolFromSmiles(smiles)
    # Chem.Kekulize(mol)
    return __extract_subgraph_rm_bonds(mol, selected_atoms, remove_bonds)


def __extract_subgraph_rm_bonds(mol, selected_atoms, remove_bonds):
    selected_atoms = set(selected_atoms)
    roots = []
    for idx in selected_atoms:
        atom = mol.GetAtomWithIdx(idx)
        bad_neis = [y for y in atom.GetNeighbors() if y.GetIdx() not in selected_atoms]
        if len(bad_neis) > 0:
            roots.append(idx)

    new_mol = Chem.RWMol(mol)
    for atom in new_mol.GetAtoms():
        atom.SetIntProp('org_idx', atom.GetIdx())

    for atom_idx in roots:
        atom = new_mol.GetAtomWithIdx(atom_idx)
        atom.SetAtomMapNum(1)
        aroma_bonds = [bond for bond in atom.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC]
        aroma_bonds = [bond for bond in aroma_bonds if bond.GetBeginAtom().GetIdx() in selected_atoms and bond.GetEndAtom().GetIdx() in selected_atoms]
        if len(aroma_bonds) == 0:
            atom.SetIsAromatic(False)

    remove_atoms = [atom.GetIdx() for atom in new_mol.GetAtoms() if atom.GetIdx() not in selected_atoms]
    remove_atoms = sorted(remove_atoms, reverse=True)
    for atom in remove_atoms:
        new_mol.RemoveAtom(atom)
    if len(remove_bonds) > 0:
        org_atom_idx = [a.GetIntProp('org_idx') for a in new_mol.GetAtoms()]
        all_bonds = [set([b.GetBeginAtomIdx(), b.GetEndAtomIdx()]) for b in new_mol.GetBonds()]
        # import pdb; pdb.set_trace()
        for bond in remove_bonds:
            bond = list(bond)
            assert(len(bond) == 2)
            if bond[0] not in org_atom_idx or bond[1] not in org_atom_idx:
                continue
            to_rm_bond = set([org_atom_idx.index(bond[0]), org_atom_idx.index(bond[1])])
            if to_rm_bond in all_bonds:
                try:
                    new_mol.RemoveBond(list(to_rm_bond)[0], list(to_rm_bond)[1])
                except:
                    print("ERROR in remove bond")
                    import pdb; pdb.set_trace()
        # print([set([b.GetBeginAtomIdx(), b.GetEndAtomIdx()]) for b in new_mol.GetBonds()])
        # import pdb; pdb.set_trace()

    return new_mol.GetMol(), roots

def __extract_subgraph(mol, selected_atoms):
    selected_atoms = set(selected_atoms)
    roots = []
    for idx in selected_atoms:
        atom = mol.GetAtomWithIdx(idx)
        bad_neis = [y for y in atom.GetNeighbors() if y.GetIdx() not in selected_atoms]
        if len(bad_neis) > 0:
            roots.append(idx)

    new_mol = Chem.RWMol(mol)
    for atom in new_mol.GetAtoms():
        atom.SetIntProp('org_idx', atom.GetIdx())

    for atom_idx in roots:
        atom = new_mol.GetAtomWithIdx(atom_idx)
        atom.SetAtomMapNum(1)
        aroma_bonds = [bond for bond in atom.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC]
        aroma_bonds = [bond for bond in aroma_bonds if bond.GetBeginAtom().GetIdx() in selected_atoms and bond.GetEndAtom().GetIdx() in selected_atoms]
        if len(aroma_bonds) == 0:
            atom.SetIsAromatic(False)

    remove_atoms = [atom.GetIdx() for atom in new_mol.GetAtoms() if atom.GetIdx() not in selected_atoms]
    remove_atoms = sorted(remove_atoms, reverse=True)
    for atom in remove_atoms:
        new_mol.RemoveAtom(atom)

    return new_mol.GetMol(), roots

def extract_subgraph(smiles, selected_atoms): 
    # try with kekulization
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    subgraph_mapped, roots = __extract_subgraph(mol, selected_atoms) 
    subgraph = Chem.MolToSmiles(subgraph_mapped, kekuleSmiles=True)
    assert '.' not in subgraph
    subgraph = Chem.MolFromSmiles(subgraph)

    mol = Chem.MolFromSmiles(smiles)  # de-kekulize
    if subgraph is not None and mol.HasSubstructMatch(subgraph):
        return subgraph, subgraph_mapped, roots

    # If fails, try without kekulization
    subgraph_mapped, roots = __extract_subgraph(mol, selected_atoms) 
    subgraph = Chem.MolToSmiles(subgraph_mapped)
    assert '.' not in subgraph
    subgraph = Chem.MolFromSmiles(subgraph)
    if subgraph is not None:
        return subgraph, subgraph_mapped, roots
    else:
        return None, subgraph_mapped, None

def find_fragments(mol):
    new_mol = Chem.RWMol(mol)
    for atom in new_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())

    for bond in mol.GetBonds():
        if bond.IsInRing(): continue
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        if a1.IsInRing() and a2.IsInRing():
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
            new_idx = new_mol.AddAtom(copy_atom(a1))
            new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a1.GetIdx())
            new_mol.AddBond(new_idx, a2.GetIdx(), bond.GetBondType())

        elif a1.IsInRing() and a2.GetDegree() > 1:
            new_idx = new_mol.AddAtom(copy_atom(a1))
            new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a1.GetIdx())
            new_mol.AddBond(new_idx, a2.GetIdx(), bond.GetBondType())
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())

        elif a2.IsInRing() and a1.GetDegree() > 1:
            new_idx = new_mol.AddAtom(copy_atom(a2))
            new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a2.GetIdx())
            new_mol.AddBond(new_idx, a1.GetIdx(), bond.GetBondType())
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
    
    new_mol = new_mol.GetMol()
    new_smiles = Chem.MolToSmiles(new_mol)

    hopts = []
    for fragment in new_smiles.split('.'):
        fmol = Chem.MolFromSmiles(fragment)
        indices = set([atom.GetAtomMapNum() for atom in fmol.GetAtoms()])
        fmol = get_clique_mol(mol, indices)
        fmol = sanitize(fmol, kekulize=False)
        fsmiles = Chem.MolToSmiles(fmol)
        hopts.append((fsmiles, indices))
    
    return hopts

def find_fragments_with_scaffold(mol, smiles):
    # scaffold_sml = ['Oc1c(O[*])cccc1', 'N#Cc1c([*])c([*])c(C#N)c2c1Oc3ccccc3O2',
    scaffold_sml = ['Oc1c(O)cccc1', 'N#Cc1ccc(C#N)c2c1Oc3ccccc3O2',
                    'Oc1ccc(S(=O)(c2ccc(O)cc2)=O)cc1', 'O=C(c1c(C2=O)cccc1)N2',
                    'Oc1ccc(C(C)(C)c2ccc(O)cc2)cc1', 'O=C(N1)c2ccccc2C1=O', 'FC(c1cc(c2ccccc2)ccc1)(F)F',
                    'Oc1ccc2c(C3(CC(C)2C)CC(C)(C)c4ccc(O)cc43)c1', 'c1ccc(c2cccc(C(F)(F)F)c2)cc1',
                    'Oc1ccc(C2(c3ccc(O)cc3)c4ccccc4C(N2c5ccccc5)=O)cc1', 'FC(F)(F)CC(F)(F)F',
                    'Oc1ccc(c2ccc(c3ccc(c4ccc(O)c(C(F)(F)F)c4)cc3)cc2)cc1C(F)(F)F',
                    'Oc1c(C(F)(F)F)cc(c2ccc(c3cc(C(F)(F)F)c([O])cc3)cc2)cc1', 
                    'Oc1c(C(F)(F)F)cc(c2sc(c3cc(C(F)(F)F)c(O)cc3)cc2)cc1',
                    'Oc1c(C(F)(F)F)cc(c2nc(c3cc(C(F)(F)F)c(O)cc3)ccc2)cc1',
                    'Oc1cc(C)c(c2c(C)cc(O)cc2)cc1',
                    ###
                    'NC(c1cccc(N(C)C)c1)=O', 'Oc2ccc(C(N)=O)cc2', 
                    'COc1cc2cncnc2cc1OCCCN3CCOCC3', 'Nc1cnc(nc1)NC(c2ccccc2)=O',
                    'Nc1sc(CC(Nc2cccc(F)c2)=O)cn1', 'COc1c(OC)cc2c(cncn2)c1',
                    'CC[C@@]1(C[C@H]2CN(C1)CCC3=C([C@@H](C2)C(OC)=O)NC4=C3C=CC=C4)O',
                    'COC1=CC(N([C@H]2[C@@](C(OC)=O)([C@@H]([C@@]3(C=CCN4CC[C@]52[C@@H]43)CC)OC(C)=O)O)C=O)=C5C=C1']
                    
    scaffold = [Chem.MolFromSmarts(sm) for sm in scaffold_sml]

    scaffold_atom_id = []
    scaffold_bond_id = []
    for i, scaffold_i in enumerate(scaffold):
        hit_ats = list(Chem.MolFromSmiles(smiles).GetSubstructMatch(scaffold_i))
        scaffold_atom_id.extend(hit_ats)
        if len(hit_ats) == 0:
            continue
        for bond in scaffold_i.GetBonds():
            aid1 = hit_ats[bond.GetBeginAtomIdx()]
            aid2 = hit_ats[bond.GetEndAtomIdx()]
            scaffold_bond_id.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

    new_mol = Chem.RWMol(mol)
    for atom in new_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())

    for bond in mol.GetBonds():

        if len(scaffold_atom_id) == 0:
            if bond.IsInRing(): continue
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if a1.IsInRing() and a2.IsInRing():
                new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
                new_idx = new_mol.AddAtom(copy_atom(a1))
                new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a1.GetIdx())
                new_mol.AddBond(new_idx, a2.GetIdx(), bond.GetBondType())

            elif a1.IsInRing() and a2.GetDegree() > 1:
                new_idx = new_mol.AddAtom(copy_atom(a1))
                new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a1.GetIdx())
                new_mol.AddBond(new_idx, a2.GetIdx(), bond.GetBondType())
                new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())

            elif a2.IsInRing() and a1.GetDegree() > 1:
                new_idx = new_mol.AddAtom(copy_atom(a2))
                new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a2.GetIdx())
                new_mol.AddBond(new_idx, a1.GetIdx(), bond.GetBondType())
                new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
        else:
            if bond.GetIdx() in scaffold_bond_id or bond.IsInRing(): continue
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if a1.GetIdx() in scaffold_atom_id and a2.GetIdx() in scaffold_atom_id:
                new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
                new_idx = new_mol.AddAtom(copy_atom(a1))
                new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a1.GetIdx())
                new_mol.AddBond(new_idx, a2.GetIdx(), bond.GetBondType())

            elif a1.GetIdx() in scaffold_atom_id and a2.GetDegree() > 1:
                new_idx = new_mol.AddAtom(copy_atom(a1))
                new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a1.GetIdx())
                new_mol.AddBond(new_idx, a2.GetIdx(), bond.GetBondType())
                new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())

            elif a2.GetIdx() in scaffold_atom_id and a1.GetDegree() > 1:
                new_idx = new_mol.AddAtom(copy_atom(a2))
                new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a2.GetIdx())
                new_mol.AddBond(new_idx, a1.GetIdx(), bond.GetBondType())
                new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())

            elif a1.IsInRing() and a2.IsInRing():
                new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
                new_idx = new_mol.AddAtom(copy_atom(a1))
                new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a1.GetIdx())
                new_mol.AddBond(new_idx, a2.GetIdx(), bond.GetBondType())

            elif a1.IsInRing() and a2.GetDegree() > 1:
                new_idx = new_mol.AddAtom(copy_atom(a1))
                new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a1.GetIdx())
                new_mol.AddBond(new_idx, a2.GetIdx(), bond.GetBondType())
                new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())

            elif a2.IsInRing() and a1.GetDegree() > 1:
                new_idx = new_mol.AddAtom(copy_atom(a2))
                new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a2.GetIdx())
                new_mol.AddBond(new_idx, a1.GetIdx(), bond.GetBondType())
                new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
    
    new_mol = new_mol.GetMol()
    new_smiles = Chem.MolToSmiles(new_mol)

    if new_smiles.count('.') > 5 and len(scaffold_atom_id) > 0:
        new_mol = Chem.RWMol(mol)
        for atom in new_mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        for bond in mol.GetBonds():
            if bond.GetIdx() in scaffold_bond_id: continue
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if a1.GetIdx() in scaffold_atom_id and a2.GetIdx() in scaffold_atom_id:
                new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
                new_idx = new_mol.AddAtom(copy_atom(a1))
                new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a1.GetIdx())
                new_mol.AddBond(new_idx, a2.GetIdx(), bond.GetBondType())

            elif a1.GetIdx() in scaffold_atom_id and a2.GetDegree() > 1:
                new_idx = new_mol.AddAtom(copy_atom(a1))
                new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a1.GetIdx())
                new_mol.AddBond(new_idx, a2.GetIdx(), bond.GetBondType())
                new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())

            elif a2.GetIdx() in scaffold_atom_id and a1.GetDegree() > 1:
                new_idx = new_mol.AddAtom(copy_atom(a2))
                new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a2.GetIdx())
                new_mol.AddBond(new_idx, a1.GetIdx(), bond.GetBondType())
                new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())

        new_mol = new_mol.GetMol()
        new_smiles = Chem.MolToSmiles(new_mol)

    hopts = []
    for fragment in new_smiles.split('.'):
        fmol = Chem.MolFromSmiles(fragment)
        indices = set([atom.GetAtomMapNum() for atom in fmol.GetAtoms()])
        fmol = get_clique_mol(mol, indices)
        fmol = sanitize(fmol, kekulize=False)
        fsmiles = Chem.MolToSmiles(fmol)
        hopts.append((fsmiles, indices))
    
    return hopts, scaffold_atom_id

def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol) 
    #if tmp_mol is not None: new_mol = tmp_mol
    return new_mol

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
        #if bt == Chem.rdchem.BondType.AROMATIC and not aromatic:
        #    bt = Chem.rdchem.BondType.SINGLE
    return new_mol

def enum_root(smiles, num_decode):
    mol = Chem.MolFromSmiles(smiles)
    roots = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0]
    outputs = []
    for perm_roots in itertools.permutations(roots):
        if len(outputs) >= num_decode: break
        mol = Chem.MolFromSmiles(smiles)
        for i,a in enumerate(perm_roots):
            mol.GetAtomWithIdx(a).SetAtomMapNum(i + 1)
        outputs.append(Chem.MolToSmiles(mol))

    while len(outputs) < num_decode:
        outputs = outputs + outputs
    return outputs[:num_decode]

def unique_rationales(smiles_list):
    visited = set()
    unique = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        root_atoms = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() > 0:
                root_atoms += 1
                atom.SetAtomMapNum(1)

        smiles = Chem.MolToSmiles(mol)
        if smiles not in visited and root_atoms > 0:
            visited.add(smiles)
            unique.append(smiles)

    return unique

def merge_rationales(x, y):
    xmol = Chem.MolFromSmiles(x)
    ymol = Chem.MolFromSmiles(y)

    mcs = rdFMCS.FindMCS([xmol, ymol], ringMatchesRingOnly=True, completeRingsOnly=True, timeout=1)
    if mcs.numAtoms == 0: return []

    mcs = Chem.MolFromSmarts(mcs.smartsString)
    xmatch = xmol.GetSubstructMatches(mcs, uniquify=False)
    ymatch = ymol.GetSubstructMatches(mcs, uniquify=False)
    
    joined = [__merge_molecules(xmol, ymol, mx, my) for mx in xmatch for my in ymatch]
    joined = [Chem.MolToSmiles(new_mol) for new_mol in joined if new_mol]
    return list(set(joined))


def __merge_molecules(xmol, ymol, mx, my):
    new_mol = Chem.RWMol(xmol)
    for i in mx:  # remove atom maps where overlap happens
        atom = new_mol.GetAtomWithIdx(i)
        atom.SetAtomMapNum(0)

    atom_map = {}
    for atom in ymol.GetAtoms():
        idx = atom.GetIdx()
        if idx in my:
            atom_map[idx] = mx[my.index(idx)]
        else:
            atom_map[idx] = new_mol.AddAtom( copy_atom(atom) )

    for bond in ymol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        bt = bond.GetBondType()
        a1, a2 = atom_map[a1.GetIdx()], atom_map[a2.GetIdx()]
        if new_mol.GetBondBetweenAtoms(a1, a2) is None:
            new_mol.AddBond(a1, a2, bt)

    new_mol = new_mol.GetMol()
    new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))
    if new_mol and new_mol.HasSubstructMatch(xmol) and new_mol.HasSubstructMatch(ymol):
        return new_mol
    else:
        return None

