from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# ============================================================
# Chemical Graph
# ============================================================

class GraphNode:
    def __init__(self):
        self.nodes = {}        # node_id -> atom symbol
        self.node_tags = {}    # node_id -> set(tags)
        self.edges = {}        # i -> j -> {"bond": int, "tags": set}
        self._next_id = 0

    # ---------- Nodes ----------
    def add_node(self, atom, tags=None):
        idx = self._next_id
        self.nodes[idx] = atom
        self.node_tags[idx] = set(tags) if tags else set()
        self.edges[idx] = {}
        self._next_id += 1
        return idx

    # ---------- Edges ----------
    def add_edge(self, i, j, bond=1, tags=None):
        if bond not in (1, 2, 3):
            raise ValueError("Bond must be 1, 2, or 3")
        data = {"bond": bond, "tags": set(tags) if tags else set()}
        self.edges[i][j] = data
        self.edges[j][i] = data

    # ---------- Cycle Detection ----------
    def find_cycle(self):
        """
        Find a single cycle in the graph using DFS.
        Returns list of node IDs forming the cycle, or None if acyclic.
        """
        visited = set()
        parent = {}
        
        def dfs(v, p):
            visited.add(v)
            parent[v] = p
            
            for neighbor in self.edges[v]:
                if neighbor == p:  # skip parent edge
                    continue
                if neighbor in visited:
                    # Found cycle - reconstruct it
                    cycle = [neighbor]
                    curr = v
                    while curr != neighbor:
                        cycle.append(curr)
                        curr = parent[curr]
                    return cycle
                else:
                    result = dfs(neighbor, v)
                    if result:
                        return result
            return None
        
        # Try from each unvisited node
        for node in self.nodes:
            if node not in visited:
                cycle = dfs(node, None)
                if cycle:
                    return cycle
        return None

    def has_cycle(self):
        """Check if graph contains a cycle"""
        return self.find_cycle() is not None

    # ---------- Main chain detection ----------
    def tag_mainchain(self, atom="C", tag="mainchain"):
        """
        Detect and orient the IUPAC main chain.
        Returns:
            mainchain: list of atom IDs in order
            numbering: dict {atom_id: position in chain (1-based)}
        """
        carbons = [i for i, a in self.nodes.items() if a == atom]
        raw_chains = []

        def dfs(v, visited, path):
            visited.add(v)
            path.append(v)
            extended = False
            for n in self.edges[v]:
                if n not in visited and self.nodes[n] == atom:
                    dfs(n, visited, path)
                    extended = True
            if not extended:
                raw_chains.append(path.copy())
            path.pop()
            visited.remove(v)

        for c in carbons:
            dfs(c, set(), [])

        def bonds_of(chain):
            return [self.edges[chain[i]][chain[i + 1]].get("bond", 1)
                    for i in range(len(chain) - 1)]

        def multiple_bond_count(bonds):
            return sum(1 for b in bonds if b > 1)

        def double_bond_count(bonds):
            return sum(1 for b in bonds if b == 2)

        def locant_score(bonds):
            double_locs = [i + 1 for i, b in enumerate(bonds) if b == 2]
            triple_locs = [i + 1 for i, b in enumerate(bonds) if b == 3]
            def s(l):
                return sum(l) if l else float("inf")
            return (s(double_locs), s(triple_locs))

        candidates = []
        max_len = max(len(c) for c in raw_chains)
        for chain in raw_chains:
            if len(chain) != max_len:
                continue
            for oriented in (chain, chain[::-1]):
                bonds = bonds_of(oriented)
                score = (
                    len(oriented),
                    multiple_bond_count(bonds),
                    double_bond_count(bonds),
                    tuple(-x for x in locant_score(bonds)),
                    sum(bonds)
                )
                candidates.append((score, oriented))

        candidates.sort(reverse=True)
        mainchain = candidates[0][1]

        # Tag atoms and create numbering
        numbering = {}
        for pos, atom_id in enumerate(mainchain, 1):
            self.node_tags[atom_id].add(tag)
            numbering[atom_id] = pos

        return mainchain, numbering

    def collect_subgraph(self, start_node, exclude=None):
        """
        Recursively collect all nodes connected to start_node, excluding nodes in `exclude`.
        """
        if exclude is None:
            exclude = set()
        seen = set()

        def dfs(node):
            if node in seen or node in exclude:
                return
            seen.add(node)
            for nbr in self.edges[node]:
                dfs(nbr)

        dfs(start_node)
        return list(seen)

    # ---------- Subgraph extraction ----------
    def subgraph(self, node_ids):
        sub = GraphNode()
        m = {}

        for i in node_ids:
            m[i] = sub.add_node(self.nodes[i], self.node_tags[i])

        for i in node_ids:
            for j, e in self.edges[i].items():
                if j in node_ids and m[i] < m[j]:
                    sub.add_edge(m[i], m[j], e["bond"], e["tags"])

        return sub

    def get_substituents(self, mainchain):
        """
        Return a dictionary mapping each main-chain atom to a list of subgraphs
        representing substituents (everything attached to that atom that's not on mainchain).
        """
        attachments = {}
        main_set = set(mainchain)

        for atom in mainchain:
            subs = []

            for neighbor, edge_data in self.edges[atom].items():
                if neighbor in main_set:
                    continue  # skip main chain atoms

                # Collect all atoms in this substituent using DFS
                visited = set()
                stack = [neighbor]
                sub_nodes = set()

                while stack:
                    n = stack.pop()
                    if n in visited or n in main_set:
                        continue
                    visited.add(n)
                    sub_nodes.add(n)
                    for nn in self.edges[n]:
                        if nn not in visited and nn not in main_set:
                            stack.append(nn)

                # Create a subgraph for this substituent
                subgraph = self.subgraph(sub_nodes)
                subs.append(subgraph)

            if subs:
                attachments[atom] = subs

        return attachments


# ============================================================
# Tree Node (Chemical AST)
# ============================================================

class TreeNode:
    def __init__(self, pos, chain_length, nodes=None, label="", bonds=None, is_cyclic=False):
        """
        pos: position on parent chain
        chain_length: length of this chain segment
        nodes: list of node indices
        label: "mainchain", "substituent", or "cycle"
        bonds: list of bond orders between consecutive nodes
        is_cyclic: True if this represents a ring structure
        """
        self.pos = pos
        self.chain_length = chain_length
        self.nodes = nodes or []
        self.label = label
        self.bonds = bonds or [1] * (len(self.nodes) - 1)
        self.is_cyclic = is_cyclic
        self.children = []

    def add_child(self, c):
        self.children.append(c)

    def __repr__(self, level=0):
        ind = "  " * level
        s = f"{ind}TreeNode(pos={self.pos}, chain_length={self.chain_length}"
        if self.label:
            s += f", label={self.label}"
        if self.is_cyclic:
            s += f", cyclic=True"
        if self.nodes:
            s += f", nodes={self.nodes}"
        if self.bonds:
            s += f", bonds={self.bonds}"
        s += ")"
        for c in self.children:
            s += "\n" + c.__repr__(level + 1)
        return s


# ============================================================
# IUPAC NAMING CONSTANTS
# ============================================================

ALKANE = {
    1: "meth",
    2: "eth",
    3: "prop",
    4: "but",
    5: "pent",
    6: "hex",
    7: "hept",
    8: "oct",
    9: "non",
    10: "dec"
}

MULTIPLIER = {
    2: "di",
    3: "tri",
    4: "tetra",
    5: "penta",
    6: "hexa"
}


# ============================================================
# Tree Building Functions
# ============================================================

def _build_substituent_tree(graph, attach_atom, start_atom, mainchain_set, visited=None):
    """
    Build a substituent TreeNode starting from start_atom,
    excluding mainchain atoms and avoiding cycles.
    """
    if visited is None:
        visited = set()

    visited.add(start_atom)

    # Build linear chain first (longest path)
    def dfs_chain(v, parent):
        best = [v]
        for n in graph.edges[v]:
            if n == parent or n in visited or n in mainchain_set:
                continue
            if graph.nodes[n] != "C":
                continue

            path = [v] + dfs_chain(n, v)
            if len(path) > len(best):
                best = path
        return best

    chain = dfs_chain(start_atom, attach_atom)
    chain_set = set(chain)

    # Bond orders along substituent chain
    bonds = [graph.edges[chain[i]][chain[i + 1]].get("bond", 1)
             for i in range(len(chain) - 1)]

    node = TreeNode(
        pos=0,
        chain_length=len(chain),
        nodes=chain,
        label="substituent",
        bonds=bonds
    )

    # Attach branches to substituent chain
    for i, atom in enumerate(chain):
        for n in graph.edges[atom]:
            if n in chain_set or n in visited or n in mainchain_set:
                continue
            if graph.nodes[n] != "C":
                continue

            child = _build_substituent_tree(
                graph,
                atom,
                n,
                mainchain_set,
                visited
            )
            child.pos = i + 1
            node.add_child(child)

    return node


def build_tree_recursive(graph: GraphNode) -> TreeNode:
    """
    Build a TreeNode recursively from a GraphNode.
    Handles both acyclic and monocyclic structures.
    """
    # Check for cycle
    cycle = graph.find_cycle()
    
    if cycle:
        # Monocyclic structure
        return _build_cyclic_tree(graph, cycle)
    else:
        # Acyclic structure
        return _build_acyclic_tree(graph)


def _build_acyclic_tree(graph: GraphNode) -> TreeNode:
    """Build tree for acyclic structure (original logic)"""
    mainchain, numbering = graph.tag_mainchain()
    L = len(mainchain)

    bonds = [graph.edges[mainchain[i]].get(mainchain[i + 1], {"bond": 1}).get("bond", 1)
             for i in range(L - 1)]

    root = TreeNode(
        pos=0,
        chain_length=L,
        nodes=mainchain,
        label="mainchain",
        bonds=bonds
    )

    main_set = set(mainchain)
    attachments = graph.get_substituents(mainchain)

    for atom_id, subgraphs in attachments.items():
        for subgraph in subgraphs:
            visited = set()
            sub_root = _build_substituent_tree(
                subgraph,
                attach_atom=None,
                start_atom=list(subgraph.nodes.keys())[0],
                mainchain_set=set()
            )
            sub_root.pos = numbering[atom_id]
            root.add_child(sub_root)

    root.children.sort(key=lambda x: x.pos)
    return root


def _build_cyclic_tree(graph: GraphNode, cycle: list) -> TreeNode:
    """
    Build tree for monocyclic structure.
    The cycle becomes the main chain, substituents attach to it.
    """
    # Find substituents first to help with orientation
    cycle_set = set(cycle)
    substituents_dict = {}
    
    for atom_id in cycle:
        for neighbor in graph.edges[atom_id]:
            if neighbor not in cycle_set:
                substituents_dict[atom_id] = True
                break
    
    # Orient cycle to minimize unsaturation and substituent locants
    cycle = _orient_cycle(graph, cycle, substituents_dict)
    
    L = len(cycle)
    
    # Get bonds around the cycle
    bonds = [graph.edges[cycle[i]][cycle[(i + 1) % L]].get("bond", 1)
             for i in range(L)]

    root = TreeNode(
        pos=0,
        chain_length=L,
        nodes=cycle,
        label="cycle",
        bonds=bonds,
        is_cyclic=True
    )

    # Find substituents attached to cycle
    attachments = graph.get_substituents(cycle)

    for atom_id, subgraphs in attachments.items():
        for subgraph in subgraphs:
            visited = set()
            sub_root = _build_substituent_tree(
                subgraph,
                attach_atom=None,
                start_atom=list(subgraph.nodes.keys())[0],
                mainchain_set=set()
            )
            # Position is 1-indexed around the cycle
            sub_root.pos = cycle.index(atom_id) + 1
            root.add_child(sub_root)

    root.children.sort(key=lambda x: x.pos)
    return root


def _orient_cycle(graph: GraphNode, cycle: list, substituents_dict) -> list:
    """
    Orient the cycle to give lowest locants to unsaturated bonds, then substituents.
    IUPAC rule: 
    1. Number to minimize positions of double/triple bonds
    2. Then minimize substituent positions
    """
    L = len(cycle)
    
    def get_cycle_bonds(oriented_cycle):
        """Get bond orders around the cycle"""
        return [graph.edges[oriented_cycle[i]][oriented_cycle[(i + 1) % L]].get("bond", 1)
                for i in range(L)]
    
    def get_substituent_positions(oriented_cycle):
        """Get positions of substituents in this orientation"""
        positions = []
        for i, atom_id in enumerate(oriented_cycle):
            if atom_id in substituents_dict:
                positions.append(i + 1)
        return positions
    
    def score_orientation(bonds, sub_positions):
        """Score based on locants - lower is better"""
        double_locs = [i + 1 for i, b in enumerate(bonds) if b == 2]
        triple_locs = [i + 1 for i, b in enumerate(bonds) if b == 3]
        
        # Return tuple for comparison (lower is better, so we negate)
        return (
            tuple(double_locs) if double_locs else tuple(),
            tuple(triple_locs) if triple_locs else tuple(),
            tuple(sub_positions) if sub_positions else tuple()
        )
    
    best = cycle
    best_score = score_orientation(get_cycle_bonds(cycle), get_substituent_positions(cycle))
    
    # Try all starting points and both directions
    for start in range(L):
        for direction in [1, -1]:
            oriented = []
            for i in range(L):
                idx = (start + direction * i) % L
                oriented.append(cycle[idx])
            
            bonds = get_cycle_bonds(oriented)
            sub_pos = get_substituent_positions(oriented)
            score = score_orientation(bonds, sub_pos)
            
            if score < best_score:  # Lower is better
                best_score = score
                best = oriented
    
    return best


# ============================================================
# IUPAC Naming Functions
# ============================================================

def needs_parentheses(name):
    """Check if substituent name needs parentheses"""
    return "-" in name or "," in name


def substituent_name(subroot):
    """
    Convert a substituent TreeNode to proper IUPAC substituent name.
    """
    if subroot.chain_length not in ALKANE:
        raise ValueError("Unsupported substituent length")

    base = ALKANE[subroot.chain_length]

    double_locs = [i + 1 for i, b in enumerate(subroot.bonds) if b == 2]
    triple_locs = [i + 1 for i, b in enumerate(subroot.bonds) if b == 3]

    suffix_parts = []

    if double_locs:
        count = len(double_locs)
        prefix = MULTIPLIER.get(count, str(count)) if count > 1 else ""
        suffix_parts.append(f"{','.join(map(str, double_locs))}-{prefix}enyl")

    if triple_locs:
        count = len(triple_locs)
        prefix = MULTIPLIER.get(count, str(count)) if count > 1 else ""
        suffix_parts.append(f"{','.join(map(str, triple_locs))}-{prefix}ynyl")

    if not suffix_parts:
        return base + "yl"

    final_suffix = []
    for i, part in enumerate(suffix_parts):
        if i > 0 and final_suffix[-1].endswith('e') and part.startswith('e'):
            part = part[1:]
        final_suffix.append(part)

    return base + "-" + "-".join(final_suffix)


def tree_to_iupac(root):
    """
    Convert TreeNode to IUPAC name.
    Handles both acyclic and cyclic structures.
    """
    if root.is_cyclic:
        return _cyclic_to_iupac(root)
    else:
        return _acyclic_to_iupac(root)


def _acyclic_to_iupac(root):
    """IUPAC naming for acyclic compounds (original logic)"""
    if root.chain_length not in ALKANE:
        raise ValueError("Unsupported chain length")

    base_chain = ALKANE[root.chain_length]

    # Collect multiple bonds
    double_locs = []
    triple_locs = []

    for i, b in enumerate(root.bonds):
        if b == 2:
            double_locs.append(i + 1)
        elif b == 3:
            triple_locs.append(i + 1)

    # Build suffix
    suffix_parts = []

    if double_locs:
        n = len(double_locs)
        mult = MULTIPLIER.get(n, "") if n > 1 else ""
        suffix_parts.append({
            "type": "ene",
            "text": f"{','.join(map(str, double_locs))}-{mult}ene"
        })

    if triple_locs:
        n = len(triple_locs)
        mult = MULTIPLIER.get(n, "") if n > 1 else ""
        suffix_parts.append({
            "type": "yne",
            "text": f"{','.join(map(str, triple_locs))}-{mult}yne"
        })

    if len(suffix_parts) == 2:
        if suffix_parts[0]["type"] == "ene" and suffix_parts[1]["type"] == "yne":
            suffix_parts[0]["text"] = suffix_parts[0]["text"].replace("ene", "en")

    suffix = "-".join(p["text"] for p in suffix_parts) if suffix_parts else "ane"

    # Substituents
    subs = []
    for c in root.children:
        name = substituent_name(c)
        display = f"({name})" if needs_parentheses(name) else name
        subs.append((c.pos, name, display))

    grouped = {}
    for pos, base, disp in subs:
        grouped.setdefault(base, {"positions": [], "display": disp})
        grouped[base]["positions"].append(pos)

    pieces = []
    for base in sorted(grouped, key=str.lower):
        data = grouped[base]
        locs = sorted(data["positions"])
        mult = MULTIPLIER.get(len(locs), "") if len(locs) > 1 else ""
        pieces.append(f"{','.join(map(str, locs))}-{mult}{data['display']}")

    # Final assembly
    if pieces:
        if suffix == "ane":
            return "-".join(pieces) + base_chain + "ane"
        else:
            return "-".join(pieces) + base_chain + "-" + suffix
    else:
        if suffix == "ane":
            return base_chain + "ane"
        else:
            return base_chain + "-" + suffix


def _cyclic_to_iupac(root):
    """
    IUPAC naming for cyclic compounds.
    Format: [substituents]cyclo[base][unsaturation suffix]
    
    Rules:
    - If only one double bond and no substituents, omit the locant (cyclopentene, not cyclopent-1-ene)
    - If multiple double bonds or substituents present, include locants for unsaturation
    """
    if root.chain_length not in ALKANE:
        raise ValueError("Unsupported ring size")

    base = ALKANE[root.chain_length]

    # Collect unsaturation in the ring
    double_locs = []
    triple_locs = []

    for i, b in enumerate(root.bonds):
        if b == 2:
            double_locs.append(i + 1)
        elif b == 3:
            triple_locs.append(i + 1)

    # Check if we have substituents
    has_substituents = len(root.children) > 0

    # Build suffix for ring
    suffix_parts = []

    if double_locs:
        n = len(double_locs)
        mult = MULTIPLIER.get(n, "") if n > 1 else ""
        
        # Only include locants if: multiple double bonds OR substituents present
        if n > 1 or has_substituents or triple_locs:
            suffix_parts.append({
                "type": "ene",
                "text": f"{','.join(map(str, double_locs))}-{mult}ene"
            })
        else:
            # Single double bond, no substituents - omit locant
            suffix_parts.append({
                "type": "ene",
                "text": f"{mult}ene"
            })

    if triple_locs:
        n = len(triple_locs)
        mult = MULTIPLIER.get(n, "") if n > 1 else ""
        
        # Always include locants for triple bonds when present
        suffix_parts.append({
            "type": "yne",
            "text": f"{','.join(map(str, triple_locs))}-{mult}yne"
        })

    # Apply elision rule
    if len(suffix_parts) == 2:
        if suffix_parts[0]["type"] == "ene" and suffix_parts[1]["type"] == "yne":
            suffix_parts[0]["text"] = suffix_parts[0]["text"].replace("ene", "en")

    suffix = "-".join(p["text"] for p in suffix_parts) if suffix_parts else "ane"

    # Handle substituents
    subs = []
    for c in root.children:
        name = substituent_name(c)
        display = f"({name})" if needs_parentheses(name) else name
        subs.append((c.pos, name, display))

    grouped = {}
    for pos, base_name, disp in subs:
        grouped.setdefault(base_name, {"positions": [], "display": disp})
        grouped[base_name]["positions"].append(pos)

    pieces = []
    for base_name in sorted(grouped, key=str.lower):
        data = grouped[base_name]
        locs = sorted(data["positions"])
        mult = MULTIPLIER.get(len(locs), "") if len(locs) > 1 else ""
        pieces.append(f"{','.join(map(str, locs))}-{mult}{data['display']}")

    # Assemble final name
    def assemble_cyclic_name(prefix, base, suffix):
        if suffix == "ane":
            ring = f"cyclo{base}ane"
        else:
            # INSERT HYPHEN if suffix starts with a digit
            if suffix[0].isdigit():
                ring = f"cyclo{base}-{suffix}"
            else:
                ring = f"cyclo{base}{suffix}"

        return f"{prefix}{ring}" if prefix else ring

    prefix = "-".join(pieces) if pieces else ""

    return assemble_cyclic_name(prefix, base, suffix)



# ============================================================
# RDKit Conversion Functions
# ============================================================

def graphnode_to_rdkit_mol(graph):
    rw_mol = Chem.RWMol()
    id_map = {}

    for node_id, atom_symbol in graph.nodes.items():
        atom = Chem.Atom(atom_symbol)
        idx = rw_mol.AddAtom(atom)
        id_map[node_id] = idx

    added = set()
    for i, neighbors in graph.edges.items():
        for j, data in neighbors.items():
            if (j, i) in added:
                continue

            bond_order = data.get("bond", 1)
            if bond_order == 1:
                bond_type = Chem.BondType.SINGLE
            elif bond_order == 2:
                bond_type = Chem.BondType.DOUBLE
            elif bond_order == 3:
                bond_type = Chem.BondType.TRIPLE
            else:
                raise ValueError(f"Invalid bond order: {bond_order}")

            rw_mol.AddBond(id_map[i], id_map[j], bond_type)
            added.add((i, j))

    mol = rw_mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        raise ValueError(f"Invalid chemical structure: {e}")

    return mol


def graphnode_to_smiles(graph, canonical=True):
    mol = graphnode_to_rdkit_mol(graph)
    return Chem.MolToSmiles(mol, canonical=canonical)


def smiles_to_graphnode(smiles: str) -> GraphNode:
    """Convert a SMILES string into a GraphNode structure, handling aromaticity."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    Chem.Kekulize(mol, clearAromaticFlags=False)  # preserve aromatic info if needed

    graph = GraphNode()
    idx_map = {}

    # Add atoms
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        # Optional: mark aromatic atoms
        if atom.GetIsAromatic():
            symbol = symbol.lower()  # lowercase to indicate aromatic (e.g., 'c' for benzene)
        node_id = graph.add_node(symbol)
        idx_map[atom.GetIdx()] = node_id

    # Add bonds
    for bond in mol.GetBonds():
        i = idx_map[bond.GetBeginAtomIdx()]
        j = idx_map[bond.GetEndAtomIdx()]

        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:
            order = 1
        elif bt == Chem.BondType.DOUBLE:
            order = 2
        elif bt == Chem.BondType.TRIPLE:
            order = 3
        elif bt == Chem.BondType.AROMATIC:
            order = 1  # Treat aromatic bonds as single for GraphNode; can add flag if needed
        else:
            order = 1

        tags = set()
        if bond.GetIsAromatic():
            tags.add("aromatic")

        graph.add_edge(i, j, bond=order, tags=tags)

    return graph



def draw_graph_with_rdkit(graph, filename="compound.png", size=(600, 400)):
    rw_mol = Chem.RWMol()
    atom_map = {}

    for node_id, atom_symbol in graph.nodes.items():
        # Convert lowercase aromatic symbols to normal uppercase
        symbol = atom_symbol.upper()
        atom = Chem.Atom(symbol)
        # Mark aromatic atom if symbol is lowercase in GraphNode
        if atom_symbol.islower():
            atom.SetIsAromatic(True)
        atom_map[node_id] = rw_mol.AddAtom(atom)

    added = set()
    for i, nbrs in graph.edges.items():
        for j, data in nbrs.items():
            key = tuple(sorted((i, j)))
            if key in added:
                continue

            bond_order = data.get("bond", 1)
            # Map bond order, mark aromatic if bond has "aromatic" tag
            if "aromatic" in data.get("tags", set()):
                bond_type = Chem.BondType.AROMATIC
            else:
                bond_type = {1: Chem.BondType.SINGLE,
                             2: Chem.BondType.DOUBLE,
                             3: Chem.BondType.TRIPLE}.get(bond_order, Chem.BondType.SINGLE)

            rw_mol.AddBond(atom_map[i], atom_map[j], bond_type)
            added.add(key)

    mol = rw_mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        print("Sanitization failed:", e)

    AllChem.Compute2DCoords(mol)

    img = Draw.MolToImage(mol, size=size, kekulize=False, wedgeBonds=True)
    img.save(filename)
    print(f"Saved {filename}")
