import string
import copy
import re

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from collections import defaultdict
from collections import deque
from collections import Counter

# ============================================================
# Chemical Graph
# ============================================================

class GraphNode:
    def __init__(self):
        self.nodes = {}          # node_id -> atom symbol
        self.node_tags = {}      # node_id -> set(tags)
        self.edges = {}          # i -> j -> {"bond": int, "tags": set}

        self.charges = {}        # node_id -> formal charge
        self.radicals = {}       # node_id -> radical electrons (0 or 1)
        self.lone_pairs = {}     # node_id -> lone pairs
        self.attached_h = {}     # node_id -> implicit hydrogens

        self._next_id = 0

    def add_node(self, atom, tags=None, charge=0, radical=0):
        idx = self._next_id
        self.nodes[idx] = atom
        self.node_tags[idx] = set(tags) if tags else set()
        self.edges[idx] = {}

        self.charges[idx] = charge
        self.radicals[idx] = radical
        self.lone_pairs[idx] = 0
        self.attached_h[idx] = 0

        self._next_id += 1
        return idx
    def copy(self):
        return copy.deepcopy(self)

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
    def remove_node(self, node_id):
        # Remove edges pointing TO this node
        for nbr in list(self.edges.get(node_id, {})):
            self.edges[nbr].pop(node_id, None)

        # Remove node entry
        self.edges.pop(node_id, None)
        self.nodes.pop(node_id, None)
        self.node_tags.pop(node_id, None)

        # Remove electronic properties safely
        if hasattr(self, "charges"):
            self.charges.pop(node_id, None)
        if hasattr(self, "radicals"):
            self.radicals.pop(node_id, None)
        if hasattr(self, "lone_pairs"):
            self.lone_pairs.pop(node_id, None)
    def has_cycle(self):
        """Check if graph contains a cycle"""
        return self.find_cycle() is not None


    def tag_mainchain(self, atom="C", tag="mainchain"):
        """
        Tag the principal chain for IUPAC naming using enumerate_acyclic_mainchains.
        Priority:
          1) Longest chain
          2) Functional groups earliest
          3) Most unsaturation
          4) Lowest sum of substituent locants (halogens + alkyls only)
        """

        # --------------------------
        # 1️⃣ Identify functional groups
        # --------------------------
        acid_carbons, aldehyde_carbons, ketone_carbons, alcohol_carbons = set(), set(), set(), set()

        for o_id, sym in self.nodes.items():
            if sym != "O":
                continue
            for c_id, edge in self.edges[o_id].items():
                if self.nodes.get(c_id) != "C":
                    continue
                bond = edge.get("bond", 1)
                carbon_neighbors = [n for n in self.edges[c_id] if self.nodes.get(n) == "C"]
                if bond == 2:
                    if len(carbon_neighbors) == 1:
                        aldehyde_carbons.add(c_id)
                    else:
                        ketone_carbons.add(c_id)
                else:
                    alcohol_carbons.add(c_id)

        # --------------------------
        # 2️⃣ Enumerate candidate chains
        # --------------------------
        all_numberings = enumerate_acyclic_mainchains(self, atom)
        if not all_numberings:
            return [], {}

        # --------------------------
        # 3️⃣ Scoring function (HALOGEN FIX)
        # --------------------------
        def score_chain(chain):
            length = len(chain)

            # Unsaturation
            bonds = [
                self.edges[chain[i]][chain[i + 1]].get("bond", 1)
                for i in range(length - 1)
            ]
            unsat = sum(1 for b in bonds if b > 1)

            # Functional group locants (earliest wins)
            fg_positions = []
            for group in (acid_carbons, aldehyde_carbons, ketone_carbons, alcohol_carbons):
                fg_positions.extend(i + 1 for i, c in enumerate(chain) if c in group)
                if fg_positions:
                    break
            fg_positions = fg_positions or [length + 1]

            # ✅ Substituent locants (ONLY halogens + alkyls)
            substituent_locs = []
            for i, c in enumerate(chain):
                for n in self.edges[c]:
                    if n in chain:
                        continue

                    sym = self.nodes.get(n)

                    # halogens count
                    if sym in HALOGEN:
                        substituent_locs.append(i + 1)
                        break

                    # alkyl branches count
                    if sym == "C":
                        substituent_locs.append(i + 1)
                        break

            sum_sub_locs = sum(substituent_locs) if substituent_locs else 0

            return (
                -length,          # longest chain
                fg_positions,     # FG earliest
                -unsat,           # most unsaturation
                sum_sub_locs      # LOWEST sum of substituent locants
            )

        # --------------------------
        # 4️⃣ Select best chain
        # --------------------------
        best_chain = None
        best_score = None

        for chain, _ in all_numberings:
            sc = score_chain(chain)
            if best_score is None or sc < best_score:
                best_score = sc
                best_chain = chain

        # --------------------------
        # 5️⃣ Assign tag and numbering
        # --------------------------
        numbering = {atom_id: pos for pos, atom_id in enumerate(best_chain, 1)}
        for atom_id in best_chain:
            self.node_tags.setdefault(atom_id, set()).add(tag)

        return best_chain, numbering


    def __repr__(self):
        return graphnode_to_smiles(self)
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
        sub.original_id = {}  # map new ID -> original ID
        m = {}

        # Add nodes
        for i in node_ids:
            new_id = sub.add_node(self.nodes[i], self.node_tags[i])
            m[i] = new_id
            sub.original_id[new_id] = i  # store mapping

        # Add edges
        for i in node_ids:
            for j, e in self.edges[i].items():
                if j in node_ids and m[i] < m[j]:
                    sub.add_edge(m[i], m[j], e["bond"], e["tags"])

        return sub


    def get_substituents(self, mainchain):
        """
        Return a dictionary mapping each main-chain atom to a list of subgraphs
        representing substituents (everything attached to that atom that's not on mainchain),
        including cyclic substituents.

        mainchain: should be the oriented cycle if ring, or main chain for acyclic
        """
        attachments = {}
        main_set = set(mainchain)  # exclude all main-chain atoms

        for atom in mainchain:  # iterate in oriented order
            subs = []

            for neighbor in self.edges[atom]:
                if neighbor in main_set:
                    continue  # skip main-chain atoms

                # Collect full connected subgraph starting from this neighbor
                sub_nodes = self.collect_subgraph(neighbor, exclude=main_set)
                if not sub_nodes:
                    continue

                subgraph = self.subgraph(sub_nodes)
                subs.append(subgraph)

            if subs:
                attachments[atom] = subs

        return attachments

VALENCE_ELECTRONS = {
    "C": 4, "N": 5, "O": 6, "S": 6,
    "P": 5, "F": 7, "Cl": 7, "Br": 7, "I": 7,
}

MAX_VALENCE = {
    "C": 4, "N": 3, "O": 2, "S": 2,
    "F": 1, "Cl": 1, "Br": 1, "I": 1,
}

def calculate_electron_state(graph: GraphNode):
    for node, atom in graph.nodes.items():
        atom_u = atom.upper()
        if atom_u not in VALENCE_ELECTRONS:
            continue

        valence_e = VALENCE_ELECTRONS[atom_u]
        charge = graph.charges.get(node, 0)

        bonding = sum(e["bond"] for e in graph.edges[node].values())

        remaining = valence_e - bonding - charge
        if remaining < 0:
            remaining = 0

        radical = graph.radicals.get(node, 0)
        lone_pairs = max(0, (remaining - radical) // 2)

        graph.lone_pairs[node] = lone_pairs

        max_v = MAX_VALENCE.get(atom_u, 0)
        graph.attached_h[node] = max(0, max_v - bonding)

# ============================================================
# Tree Node (Chemical AST)
# ============================================================

class TreeNode:
    def __init__(self, pos, chain_length, nodes=None, label="", bonds=None, is_cyclic=False, atom=None, exo_bond=None, charge=0):
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
        self.atom = atom
        self.exo_bond = exo_bond
        self.charge = charge
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
        if self.charge != 0:
            s += f", charge={self.charge}"
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
    6: "hexa",
    7: "hepta"
}

HALOGEN = {
    "F": "fluoro",
    "Cl": "chloro",
    "Br": "bromo",
    "I": "iodo"
}
HETERO = {
    "O": "oxy"
}
FUNCTIONAL_GROUP_LABELS = {
    "carboxylic_acid",
    "aldehyde",
    "ketone",
    "alcohol",
    "cyano",
    "nitro",
    "halogen",
}


def enumerate_acyclic_mainchains(graph: GraphNode, atom="C"):
    # 1️⃣ Detect cycle nodes
    cycle = graph.find_cycle()  # returns a list of node IDs forming the cycle, or None
    cycle_nodes = set(cycle) if cycle else set()


    # 2️⃣ Identify valid starting nodes (terminal or near-terminal acyclic carbons)
    potential_starts = []
    for nid, sym in graph.nodes.items():
        if sym != atom or nid in cycle_nodes:
            continue
        carbon_neighbors = [nbr for nbr in graph.edges[nid] 
                            if graph.nodes[nbr] == atom and nbr not in cycle_nodes]
        if len(carbon_neighbors) <= 1:  # terminal or near-terminal
            potential_starts.append(nid)

    # 3️⃣ DFS to enumerate paths
    raw_chains = []

    def dfs(node, visited, path):
        visited.add(node)
        path.append(node)
        extended = False

        for nbr in graph.edges[node]:
            if nbr in visited:
                continue
            if graph.nodes[nbr] != atom:
                continue
            if nbr in cycle_nodes:
                continue  # skip cycles
            dfs(nbr, visited, path)
            extended = True

        if not extended:
            raw_chains.append(path.copy())

        path.pop()
        visited.remove(node)

    # 4️⃣ Start DFS from each valid start
    for start in potential_starts:
        dfs(start, set(), [])

    # 5️⃣ Generate numbering dictionaries
    all_numberings = []
    for chain in raw_chains:
        numbering = {nid: pos for pos, nid in enumerate(chain, 1)}
        all_numberings.append((chain, numbering))

    return all_numberings
def has_single_carbon_attachment_with_halogen_or_oxygen(
    graph: GraphNode,
    cycle: list
) -> bool:
    """
    Returns True if any cycle carbon:
    1) Has exactly ONE attachment outside the cycle
    2) That attachment contains at least one carbon
    3) AND contains at least one halogen OR oxygen anywhere in that attachment
    """
    cycle_set = set(cycle)

    for c in cycle:
        # nodes directly attached outside the cycle
        external = [n for n in graph.edges[c] if n not in cycle_set]

        if len(external) != 1:
            continue

        start = external[0]

        # Traverse the entire attachment subgraph
        stack = [start]
        visited = {c} | cycle_set

        found_carbon = False
        found_hetero = False

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            sym = graph.nodes.get(node)

            if sym == "C":
                found_carbon = True
            elif sym == "O" or sym in HALOGEN:
                found_hetero = True

            for nbr in graph.edges[node]:
                if nbr not in visited:
                    stack.append(nbr)

        if found_carbon and found_hetero:
            return c

    return None

def children_only_ketone_or_halogen(node: "TreeNode") -> bool:
    """
    Return True if all descendants (children at any depth) of `node`
    are ketones or halogens. The node itself is NOT checked.
    """

    allowed = {
        "ketone",
        "halogen",
        "fluoro",
        "chloro",
        "bromo",
        "iodo",
        "aldehyde"
    }

    for child in node.children:
        # Check this child
        if child.label in allowed:
            return True

        # Recursively check its children
        if children_only_ketone_or_halogen(child):
            return True

    return False


def build_tree_recursive(graph: GraphNode,start_atom=None) -> TreeNode:
    def has_carbon(g: GraphNode) -> bool:
        return any(sym in ["c","C"] for sym in g.nodes.values())
    
    if not has_carbon(graph):
        return None  # skip this graph entirely
    cycle = graph.find_cycle()
    if cycle:
        out2 = _build_cyclic_tree(graph, cycle, start_atom)
        convert_carbaldehyde_nodes(out2)
        
        if not children_only_ketone_or_halogen(out2):
            return out2
        out = has_single_carbon_attachment_with_halogen_or_oxygen(graph, cycle)
        if out:
            return _build_acyclic_tree(graph, out)
        return _build_cyclic_tree(graph, cycle, start_atom)
    # All other cases: acyclic chain (includes -OH if present)
    return _build_acyclic_tree(graph, start_atom)
def normalize_carboxylic_acids(root: TreeNode):
    """
    Convert aldehyde + alcohol at same position into carboxylic acid.
    """

    # group children by position
    by_pos = defaultdict(list)
    for child in root.children:
        by_pos[child.pos].append(child)

    new_children = []

    for pos, nodes in by_pos.items():
        labels = {n.label for n in nodes}

        # 🔴 aldehyde + alcohol → carboxylic acid
        if "aldehyde" in labels and "alcohol" in labels:
            new_children.append(
                TreeNode(
                    pos=pos,
                    chain_length=1,
                    nodes=[pos],   # symbolic; same as aldehyde logic
                    label="carboxylic_acid",
                    bonds=[]
                )
            )
        else:
            # keep nodes unchanged
            new_children.extend(nodes)

    root.children = sorted(new_children, key=lambda x: (x.pos, x.label))
def _build_acyclic_tree(graph: GraphNode, start_atom=None) -> TreeNode:
    """
    Recursive version of _build_acyclic_tree.
    ALL original features preserved.
    Alcohols (-OH), halogens, and nitroso (-N=O) are detected and added as nodes.
    """

    # ============================================================
    # 1️⃣ Identify main chain
    # ============================================================
    mainchain, numbering = graph.tag_mainchain()
    if not mainchain:
        raise ValueError("No main chain found")

    L = len(mainchain)
    bonds = [
        graph.edges[mainchain[i]][mainchain[i + 1]].get("bond", 1)
        for i in range(L - 1)
    ]

    root = TreeNode(
        pos=0,
        chain_length=L,
        nodes=mainchain[:],
        label="mainchain",
        bonds=bonds
    )

    # ============================================================
    # 2️⃣ Detect carbonyls on main chain (C=O)
    # ============================================================
    carbonyl_pairs = []
    for c in mainchain:
        for nbr, edge in graph.edges[c].items():
            if graph.nodes.get(nbr) == "O" and edge.get("bond") == 2:
                carbonyl_pairs.append((c, nbr))

    # ============================================================
    # 3️⃣ Detect alcohols on main chain (C–OH)
    # ============================================================
    alcohol_nodes = []
    for c in mainchain:
        for nbr, edge in graph.edges[c].items():
            if graph.nodes.get(nbr) == "O" and edge.get("bond", 1) == 1:
                alcohol_nodes.append((c, nbr))

    # ============================================================
    # 3️⃣b Detect halogens on main chain (C–X)
    # ============================================================
    halogen_nodes = []
    for c in mainchain:
        for nbr, edge in graph.edges[c].items():
            if graph.nodes.get(nbr) in HALOGEN and edge.get("bond", 1) == 1:
                halogen_nodes.append((c, nbr))

    # ============================================================
    # 🆕 3️⃣c Detect nitroso groups on main chain (C–N=O)
    # ============================================================

    nitro_nodes = []
    for c in mainchain:
        for nbr, edge in graph.edges[c].items():
            if graph.nodes.get(nbr) == "N":
                oxy_count = 0
                for n2, e2 in graph.edges[nbr].items():
                    if n2 != c and graph.nodes.get(n2) == "O" and e2.get("bond") in (1, 2):
                        oxy_count += 1
                if oxy_count == 2:
                    nitro_nodes.append((c, nbr))
    # ============================================================
    # 🆕 3️⃣d Detect cyano groups on main chain (C–C≡N)
    # ============================================================
    cyano_nodes = []
    for c in mainchain:
        for c2, edge_cc in graph.edges[c].items():
            if graph.nodes.get(c2) != "C" or edge_cc.get("bond") != 1:
                continue

            # Check for C≡N
            for n, edge_cn in graph.edges[c2].items():
                if graph.nodes.get(n) == "N" and edge_cn.get("bond") == 3:
                    cyano_nodes.append((c, c2))
                    break
    # ============================================================
    # Add charged-carbon nodes on main chain
    # ============================================================
    for i, atom in enumerate(mainchain):
        charge = graph.charges.get(atom, 0)
        if charge != 0 and graph.nodes.get(atom) == "C":
            root.add_child(
                TreeNode(
                    pos=i + 1,                # mainchain position
                    chain_length=1,
                    nodes=[atom],
                    label="charged_carbon",
                    bonds=[],
                    charge=charge            # store charge on TreeNode
                )
            )


    # ============================================================
    # 5️⃣ Recursively build substituents
    # ============================================================
    attachments = graph.get_substituents(mainchain)

    for atom in mainchain:
        pos = numbering[atom]
        for subgraph in attachments.get(atom, []):
            if not subgraph.nodes:
                continue

            sub_root = build_tree_recursive(subgraph, start_atom)
            if sub_root:
                sub_root.pos = pos
                root.add_child(sub_root)

    # ============================================================
    # 6️⃣ Add carbonyl nodes (aldehyde vs ketone)
    # ============================================================
    terminal_carbons = {mainchain[0], mainchain[-1]}
    if start_atom is not None:
        terminal_carbons = terminal_carbons -set(graph.edges[start_atom].keys())

    for c, _ in carbonyl_pairs:
        label = "aldehyde" if c in terminal_carbons else "ketone"
        root.add_child(
            TreeNode(
                pos=numbering[c],
                chain_length=1,
                nodes=[c],
                label=label,
                bonds=[]
            )
        )

    # ============================================================
    # 7️⃣ Add alcohol nodes
    # ============================================================
    for c, o in alcohol_nodes:
        root.add_child(
            TreeNode(
                pos=numbering[c],
                chain_length=1,
                nodes=[o],
                label="alcohol",
                bonds=[]
            )
        )

    # ============================================================
    # 7️⃣b Add halogen nodes
    # ============================================================
    for c, x in halogen_nodes:
        root.add_child(
            TreeNode(
                pos=numbering[c],
                chain_length=1,
                nodes=[x],
                label="halogen",
                atom=graph.nodes[x],
                bonds=[]
            )
        )

    # ============================================================
    # 🆕 7️⃣c Add nitroso nodes
    # ============================================================

    for c, n in nitro_nodes:
        root.add_child(TreeNode(pos=numbering[c], chain_length=1, nodes=[n], label="nitro", bonds=[]))

    # ============================================================
    # 🆕 7️⃣d Add cyano nodes
    # ============================================================
    for c, c2 in cyano_nodes:
        root.add_child(
            TreeNode(
                pos=numbering[c],
                chain_length=1,
                nodes=[c2],
                label="cyano",
                bonds=[]
            )
        )


    # ============================================================
    # 9️⃣ Final normalization
    # ============================================================
    root.children.sort(key=lambda x: (x.pos, x.label))

    return root


def _build_cyclic_tree(graph: GraphNode, cycle: list, start_atom=None) -> TreeNode:
    """
    Recursive cyclic tree builder.
    Detects alcohols, ketones, and halogens on the ring.
    Phenols are ignored.
    """
    L = len(cycle)
    cycle_set = set(cycle)

    # ============================================================
    # Aromatic detection
    # ============================================================
    ring_bonds = [graph.edges[cycle[i]][cycle[(i + 1) % L]].get("bond", 1) for i in range(L)]
    ring_tags = [graph.edges[cycle[i]][cycle[(i + 1) % L]].get("tags", set()) for i in range(L)]
    is_aromatic = all("aromatic" in t for t in ring_tags) or (ring_bonds.count(2) == 3 and ring_bonds.count(1) == 3)

    # ============================================================
    # Substituent positions
    # ============================================================
    substituents_dict = {}
    for atom in cycle:
        for nbr in graph.edges[atom]:
            if nbr not in cycle_set:
                substituents_dict[atom] = True
                break

    # ============================================================
    # Detect ketones, alcohols, and halogens
    # ============================================================
    ketone_pairs = []
    alcohol_nodes = []
    halogen_nodes = []
    carbaldehyde_carbons = set()
    carbaldehyde_nodes = []


    for atom in cycle:
        for nbr, edge in graph.edges[atom].items():
            if nbr in cycle_set:
                continue
            sym = graph.nodes.get(nbr)

            if sym == "O":
                if edge.get("bond", 1) == 2:
                    ketone_pairs.append((atom, nbr))
                elif edge.get("bond", 1) == 1:
                    alcohol_nodes.append((atom, nbr))

            elif sym == "C":
                # detect –CHO (carbaldehyde) without explicit hydrogens
                bonds = graph.edges[nbr]

                # must have exactly one double-bonded oxygen
                double_o = [
                    x for x, e in bonds.items()
                    if graph.nodes.get(x) == "O" and e.get("bond") == 2
                ]

                # heavy atom degree (exclude ring atom check later)
                heavy_neighbors = [
                    x for x in bonds
                    if graph.nodes.get(x) != "H"
                ]

                # conditions for aldehyde carbon
                if (
                    len(double_o) == 1
                    and len(heavy_neighbors) == 2  # ring C + O
                    and atom in bonds              # bonded to ring carbon
                ):
                    carbaldehyde_carbons.add(atom)
                    carbaldehyde_nodes.append((atom, nbr))


            elif sym in {"F", "Cl", "Br", "I"}:
                halogen_nodes.append((atom, sym))

    # ============================================================
    # Orient cycle (existing logic)
    # ============================================================

    oriented_cycle = _orient_cycle(
        graph,
        cycle,
        substituents_dict,
        is_aromatic,
        ketone_carbons={c for c, _ in ketone_pairs},
        carbaldehyde_carbons=carbaldehyde_carbons,
        start_atom=start_atom
    )

    
    bonds = [graph.edges[oriented_cycle[i]][oriented_cycle[(i + 1) % L]].get("bond", 1) for i in range(L)]

    root = TreeNode(
        pos=0,
        chain_length=L,
        nodes=oriented_cycle,
        label="cycle",
        bonds=bonds,
        is_cyclic=True
    )

    # ============================================================
    # Recursive attachment of substituents
    # ============================================================
    attachments = graph.get_substituents(oriented_cycle)
    for atom, subgraphs in attachments.items():
        pos = oriented_cycle.index(atom) + 1
        for subgraph in subgraphs:
            if not subgraph.nodes:
                continue

            attach_atom = None
            for n in subgraph.nodes:
                orig_n = getattr(subgraph, "original_id", {}).get(n, n)
                if atom in graph.edges.get(orig_n, {}):
                    attach_atom = orig_n
                    break

            if attach_atom is None:
                continue

            bond_order = graph.edges[atom][attach_atom].get("bond", 1)

                        
            # 🔹 Recursive tree for the substituent
            sub_root = build_tree_recursive(subgraph, start_atom)
            if not sub_root:
                continue

            # 🔹 Exocyclic unsaturation
            if (
                bond_order in (2, 3)
                and sub_root.label == "mainchain"
                and not sub_root.children
                and all(b == 1 for b in sub_root.bonds)
            ):
                root.add_child(
                    TreeNode(
                        pos=pos,
                        chain_length=sub_root.chain_length,
                        nodes=sub_root.nodes,
                        label="exocyclic_unsat",
                        bonds=[],
                        exo_bond=bond_order
                    )
                )
                continue

            # 🔹 Normal attachment
            sub_root.pos = pos
            root.add_child(sub_root)

    # ============================================================
    # Add ketone nodes
    # ============================================================
    for c, _ in ketone_pairs:
        root.add_child(
            TreeNode(
                pos=oriented_cycle.index(c) + 1,
                chain_length=1,
                nodes=[_],
                label="ketone",
                bonds=[]
            )
        )

    # ============================================================
    # Add alcohol nodes
    # ============================================================
    for atom, nbr in alcohol_nodes:
        root.add_child(
            TreeNode(
                pos=oriented_cycle.index(atom) + 1,
                chain_length=1,
                nodes=[nbr],
                label="alcohol",
                bonds=[]
            )
        )

    # ============================================================
    # Add halogen nodes
    # ============================================================
    for atom, sym in halogen_nodes:
        root.add_child(
            TreeNode(
                pos=oriented_cycle.index(atom) + 1,
                chain_length=1,
                nodes=[atom],
                label="halogen",
                bonds=[],
                atom=sym  # store halogen type
            )
        )

    root.children.sort(key=lambda x: (x.pos, x.label))
    return root

def enumerate_cycle_numberings(cycle, start_atom=None):
    """
    Return all possible numberings of a cycle.
    Each numbering is a list of atom IDs.
    Includes both directions.
    If start_atom is given, numbering starts only from that atom.
    """
    L = len(cycle)
    numberings = []

    if start_atom is None:
        starts = range(L)
    else:
        if start_atom not in cycle:
            raise ValueError("start_atom not in cycle")
        starts = [cycle.index(start_atom)]

    for start in starts:
        # clockwise
        numberings.append([cycle[(start + i) % L] for i in range(L)])
        # anticlockwise
        numberings.append([cycle[(start - i) % L] for i in range(L)])

    return numberings



def _orient_cycle(
    graph: GraphNode,
    cycle: list,
    substituents_dict: dict,
    is_aromatic: bool = False,
    ketone_carbons=None,
    carbaldehyde_carbons=None,
    start_atom=None
):

    """
    Orient a cyclic structure for IUPAC naming.
    Handles ketones, alcohols, and halogens.
    Halogens have higher priority than alkyls.
    Returns the best oriented list of atoms around the ring.
    """
    ketone_carbons = ketone_carbons or set()
    carbaldehyde_carbons = carbaldehyde_carbons or set()

    def get_cycle_bonds(oriented):
        L = len(oriented)
        return [
            graph.edges[oriented[i]][oriented[(i + 1) % L]].get("bond", 1)
            for i in range(L)
        ]

    def substituent_locants(oriented):
        return tuple(
            i + 1 for i, a in enumerate(oriented)
            if a in substituents_dict
        )

    def substituent_alpha_sequence(oriented):
        """
        Collect substituents outside the ring.
        Halogens have higher priority than alkyls.
        """
        seq = []
        for i, atom in enumerate(oriented):
            if atom in substituents_dict:
                for nbr in graph.edges[atom]:
                    if nbr not in oriented:
                        sym = graph.nodes[nbr]
                        name = HALOGEN.get(sym, sym)
                        priority = 0 if sym in HALOGEN else 1
                        seq.append((priority, name, i + 1))
        seq.sort(key=lambda x: (x[0], x[1], x[2]))
        return tuple((pos, name) for _, name, pos in seq)

    best_oriented = None
    best_score = None

    # 🔁 Use enumerated cycle numberings
    for oriented in enumerate_cycle_numberings(cycle, start_atom):
        score = (
            tuple(i + 1 for i, a in enumerate(oriented) if a in carbaldehyde_carbons),
            tuple(i + 1 for i, a in enumerate(oriented) if a in ketone_carbons),
            substituent_locants(oriented),
            substituent_alpha_sequence(oriented),
        )


        if best_score is None or score < best_score:
            best_score = score
            best_oriented = oriented

    return best_oriented


# ============================================================
# IUPAC Naming Functions
# ============================================================

def needs_parentheses(name: str) -> bool:
    """
    Check if a substituent name needs parentheses in the IUPAC name.
    
    According to IUPAC recommendations:
    - Parentheses are required when the substituent name contains locants
      (commas or hyphens for numbers) or is itself a complex name with hyphens.
    - Simple alkyl (ethyl, propyl) or single-word prefixes do not need them.
    - Unsaturated substituents like "prop-1-en-1-yl" need them.
    - "hydroxymethyl" does NOT need parentheses (treated as simple prefix).
    
    Returns True if parentheses are needed.
    """
    if name == "hydroxymethyl":
        return False  # special case: no parentheses for hydroxymethyl
    
    # Needs parentheses if:
    # - Contains a comma (multiple locants inside, e.g., "1,1-dichloroethyl")
    # - Contains a hyphen followed by digit (unsaturation locant: "prop-1-enyl")
    # - Contains hyphen but not just a multiplier (e.g., "di" or "tri" alone is okay, but "1-enyl" is not)
    if "," in name:
        return True
    if "-" in name:
        # Split to check if any part after hyphen is numeric (locant)
        parts = name.split("-")
        if any(part.isdigit() or (len(part) > 1 and part[0].isdigit()) for part in parts):
            return True
        # If it has hyphen but no digits, it's likely a complex base like "cyclohexyl" — no parens needed
        # But unsaturated always have digits → already covered
        return False
    
    return False

VOWEL_STARTING_SUFFIXES = (
    "ol", "al", "one", "oic", "amine", "amide", "thiol", "hydroxy"
)

def elide_unsaturation_e(name: str) -> str:
    """
    Removes the terminal 'e' from 'ene' or 'yne' ONLY when
    followed by a vowel-starting suffix (IUPAC vowel elision).
    
    Examples:
    - prop-2-ene-1-ol  -> prop-2-en-1-ol
    - but-1-yne-3-ol   -> but-1-yn-3-ol
    - prop-1-ene       -> unchanged
    - benzene          -> unchanged
    """

    # Never touch benzene or substituted benzenes
    if "benzene" in name:
        return name

    for suf in VOWEL_STARTING_SUFFIXES:
        # ene → en
        name = re.sub(
            rf"ene(-\d+)?-{suf}",
            lambda m: f"en{m.group(1) or ''}-{suf}",
            name
        )

        # yne → yn
        name = re.sub(
            rf"yne(-\d+)?-{suf}",
            lambda m: f"yn{m.group(1) or ''}-{suf}",
            name
        )

    return name


def tree_to_iupac(root):
    """
    Convert TreeNode to IUPAC name.
    Handles both acyclic and cyclic structures.
    """
    return elide_unsaturation_e(iupac_name(root))


# Constants
HALOGEN = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
MULTIPLIER = {2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa', 7: 'hepta', 8: 'octa', 9: 'nona', 10: 'deca'}
ALKANE_STEM = {
    1: 'meth', 2: 'eth', 3: 'prop', 4: 'but', 5: 'pent',
    6: 'hex', 7: 'hept', 8: 'oct', 9: 'non', 10: 'dec'
}

def _build_substituent_name(child: "TreeNode", graph: "GraphNode" = None) -> str:
    """
    Recursively build the name of a substituent (halogens, alkyl, cycles, etc.)
    with parentheses for IUPAC naming.
    """

    hal_count = defaultdict(list)
    other_children = []

    # 1️⃣ Count halogens and collect other children
    for grand in getattr(child, "children", []):
        if grand.label == "halogen":
            # Determine element symbol
            if hasattr(grand, "atom"):
                element = grand.atom
            elif graph is not None:
                element = graph.nodes[grand.nodes[0]]
            else:
                raise ValueError("Cannot determine halogen element. Pass `graph` or set grand.atom.")
            hal_count[element].append(grand.pos)
        else:
            other_children.append(grand)

    # 2️⃣ Build halogen prefix with positions
    hal_parts = []
    for element in sorted(hal_count, key=lambda x: HALOGEN[x]):
        positions = sorted(hal_count[element])
        count = len(positions)
        mult = MULTIPLIER[count] if count > 1 else ""
        pos_str = ','.join(map(str, positions))
        hal_parts.append(f"{pos_str}-{mult}{HALOGEN[element]}")
    hal_prefix = "".join(hal_parts)

    # 3️⃣ Determine base name
    if child.label == "cycle":
        if child.chain_length == 6 and getattr(child, "is_cyclic", False):
            if len(child.bonds) == 6 and all(b in (1, 2) for b in child.bonds) and child.bonds.count(2) == 3:
                base = "phenyl"
            else:
                base = f"cyclo{ALKANE_STEM[child.chain_length]}yl"
        else:
            base = f"cyclo{ALKANE_STEM[child.chain_length]}yl"
    elif child.chain_length == 1 and child.label == "mainchain":
        base = "methyl"
    else:
        base = f"{ALKANE_STEM[child.chain_length]}yl"

    name = hal_prefix + base if hal_prefix else base

    # 4️⃣ Recursively handle other children (alkyl, cycles)
    if other_children:
        inner_parts = [_build_substituent_name(inner, graph) for inner in other_children]
        # Wrap this entire substituent in parentheses
        name = f"({name}){''.join(inner_parts)}"
    else:
        # Even halogens on cycles get parentheses if hal_prefix exists
        if hal_prefix and child.label == "cycle":
            name = f"({name})"

    return name


def iupac_name(root: "TreeNode") -> str:
    is_cyclic = getattr(root, "is_cyclic", False)

    is_benzene = (
        is_cyclic and root.chain_length == 6 and
        all(b in (1, 2) for b in root.bonds) and root.bonds.count(2) == 3
    )

    # Unsaturation
    double_pos = sorted(i + 1 for i, b in enumerate(root.bonds) if b == 2)
    triple_pos = sorted(i + 1 for i, b in enumerate(root.bonds) if b == 3)
    unsat_parts = []
    if not is_benzene:
        if double_pos:
            mult = MULTIPLIER[len(double_pos)] if len(double_pos) > 1 else ""
            unsat_parts.append(f"{','.join(map(str, double_pos))}-{mult}en")
        if triple_pos:
            mult = MULTIPLIER[len(triple_pos)] if len(triple_pos) > 1 else ""
            unsat_parts.append(f"{','.join(map(str, triple_pos))}-{mult}yn")
    unsaturation = "-".join(unsat_parts) if unsat_parts else ""

    # Functional groups
    acid_children = [c for c in root.children if c.label == "carboxylic_acid"]
    aldehyde_children = [c for c in root.children if c.label in ("aldehyde")]
    carbaldehyde_children = [c for c in root.children if c.label in ("carbaldehyde")]
    ketone_children = [c for c in root.children if c.label == "ketone"]
    alcohol_children = [c for c in root.children if c.label == "alcohol"]
    cyano_children = [c for c in root.children if c.label == "cyano"]

    acid_pos = sorted(c.pos for c in acid_children)
    aldehyde_pos = sorted(c.pos for c in aldehyde_children)
    carbaldehyde_pos = sorted(c.pos for c in carbaldehyde_children)
    ketone_pos = sorted(c.pos for c in ketone_children)
    alcohol_pos = sorted(c.pos for c in alcohol_children)
    cyano_pos = sorted(c.pos for c in cyano_children)
    
    has_acid = bool(acid_pos)
    has_higher = has_acid or bool(aldehyde_pos)

    alcohol_is_prefix = bool(alcohol_pos) and (is_benzene or has_higher or bool(ketone_pos))

    # Prefixes
    prefix_dict = defaultdict(list)

    if alcohol_is_prefix:
        prefix_dict["hydroxy"].extend(alcohol_pos)

    # Direct halogens on main chain
    for child in root.children:
        if child.label == "halogen":
            prefix_dict[HALOGEN[child.atom]].append(child.pos)
            
    # Cyano groups (always prefix)
    for pos in cyano_pos:
        prefix_dict["cyano"].append(pos)

    for child in root.children:
        if child.label == "nitro":
            prefix_dict["nitro"].append(child.pos)

    # Exocyclic unsaturation
    for child in root.children:
        if child.label == "exocyclic_unsat":
            bond = getattr(child, "exo_bond", 1)
            stem = ALKANE_STEM.get(child.chain_length, "alk")  # e.g., meth, eth, prop...
            if bond == 2:
                prefix_name = f"{stem}ylidene"
            elif bond == 3:
                prefix_name = f"{stem}ylidyne"
            else:
                prefix_name = f"{stem}yl"  # fallback, single bond
            prefix_dict[prefix_name].append(child.pos)

    
    # Substituents (alkyl, cycle)
    for child in root.children:
        if child.label in ("mainchain", "cycle"):
            sub_name = _build_substituent_name(child)
            prefix_dict[sub_name].append(child.pos)

    # Build prefixes
    prefix_parts = []
    for name in sorted(prefix_dict, key=str.lower):
        pos_list = sorted(prefix_dict[name])
        mult = MULTIPLIER[len(pos_list)] if len(pos_list) > 1 else ""
        prefix_parts.append(f"{','.join(map(str, pos_list))}-{mult}{name}")

    prefixes = "-".join(prefix_parts)
    if is_benzene and len(prefix_parts) == 1 and prefix_parts[0].startswith("1-"):
        prefixes = prefix_parts[0][2:]

    # Core construction - FIXED STEM LOGIC
    core_parts = []

    if is_benzene:
        core_parts.append("benzene")
    else:
        has_suffix = bool(acid_pos or aldehyde_pos or ketone_pos or (alcohol_pos and not alcohol_is_prefix))
        if is_cyclic:
            cyclo_prefix = "cyclo"
            if has_suffix or unsaturation:
                if unsaturation:
                    stem = f"{cyclo_prefix}{ALKANE_STEM[root.chain_length]}"
                else:
                    stem = f"{cyclo_prefix}{ALKANE_STEM[root.chain_length]}an"
            else:
                stem = f"{cyclo_prefix}{ALKANE_STEM[root.chain_length]}ane"
            core_parts.append(stem)
        else:
            if has_suffix or unsaturation:
                if unsaturation:
                    stem = ALKANE_STEM[root.chain_length]
                else:
                    stem = ALKANE_STEM[root.chain_length] + "an"
            else:
                stem = ALKANE_STEM[root.chain_length] + "ane"
            core_parts.append(stem)

        if unsaturation:
            core_parts.append(unsaturation)

    # Suffix without leading hyphen
    suffix = ""
    if has_acid:
        # Special for acid: assume single terminal for now, no locant
        if len(acid_pos) == 1:
            suffix = "oic acid"
        else:
            mult = MULTIPLIER[len(acid_pos)] if len(acid_pos) > 1 else ""
            locs = ','.join(map(str, acid_pos))
            suffix = f"{locs}-{mult}dioic acid"
    elif aldehyde_pos:
        mult = MULTIPLIER[len(aldehyde_pos)] if len(aldehyde_pos) > 1 else ""
        if len(aldehyde_pos) == 1 and aldehyde_pos[0] == 1:
            suffix = mult + "al"
        else:
            locs = ','.join(map(str, aldehyde_pos))
            suffix = f"{locs}-{mult}al"
    elif ketone_pos:
        mult = MULTIPLIER[len(ketone_pos)] if len(ketone_pos) > 1 else ""
        locs = ','.join(map(str, ketone_pos))
        suffix = f"{locs}-{mult}one"
    elif alcohol_pos and not alcohol_is_prefix:
        mult = MULTIPLIER[len(alcohol_pos)] if len(alcohol_pos) > 1 else ""
        locs = ','.join(map(str, alcohol_pos))
        suffix = f"{locs}-{mult}ol"
    elif carbaldehyde_pos:
        mult = MULTIPLIER[len(carbaldehyde_pos)] if len(carbaldehyde_pos) > 1 else ""
        if len(carbaldehyde_pos) == 1 and carbaldehyde_pos[0] == 1:
            suffix = mult + "carbaldehyde"
        else:
            locs = ','.join(map(str, carbaldehyde_pos))
            suffix = f"{locs}-{mult}carbaldehyde"
    core = "-".join(core_parts) + (f"-{suffix}" if suffix else "")

    # Vowel elision
    core = core.replace("en-al", "enal").replace("yn-al", "ynal").replace("en-one", "enone").replace("yn-one", "ynone")

    # Final name
    if prefixes:
        return f"{prefixes}-{core}"
    return core

def remove_unnecessary_hyphens(name: str) -> str:
    parts = name.split("-")
    if len(parts) == 1:
        return name

    out = parts[0]

    for i in range(1, len(parts)):
        left = parts[i - 1]
        right = parts[i]

        # keep hyphen ONLY if either side has a digit
        if any(c.isdigit() for c in left) or any(c.isdigit() for c in right):
            out += "-" + right
        else:
            out += right

    return out


def convert_carbaldehyde_nodes(root: TreeNode):
    """
    Recursively convert nodes like:
        mainchain(1) -> aldehyde
    into a single carbaldehyde node attached to the parent.
    """
    new_children = []

    for child in root.children:
        # Recursively process lower levels first
        convert_carbaldehyde_nodes(child)

        # Detect carbaldehyde pattern
        if child.label == "mainchain" and child.chain_length == 1:
            aldehyde_child = None
            for gc in child.children:
                if gc.label == "aldehyde":
                    aldehyde_child = gc
                    break
            if aldehyde_child:
                # Create a new carbaldehyde node
                carbal_node = TreeNode(
                    pos=child.pos,
                    chain_length=1,
                    nodes=child.nodes[:],
                    label="carbaldehyde",
                    bonds=[]
                )
                new_children.append(carbal_node)
                continue  # Skip adding original mainchain node

        # Otherwise, keep the child as is
        new_children.append(child)

    root.children = new_children
    
# ============================================================
# RDKit Conversion Functions
# ============================================================

def graphnode_to_rdkit_mol(graph):
    rw_mol = Chem.RWMol()
    id_map = {}

    charges = getattr(graph, "charges", {})
    radicals = getattr(graph, "radical", {})

    # ---------- Add atoms ----------
    for node_id, atom_symbol in graph.nodes.items():
        atom = Chem.Atom(atom_symbol)

        # 🔹 charges
        ch = charges.get(node_id, 0)
        if ch != 0:
            atom.SetFormalCharge(int(ch))

        # 🔹 radicals
        rad = radicals.get(node_id, 0)
        if rad:
            atom.SetNumRadicalElectrons(int(rad))

        # 🔹 CRITICAL: do NOT show hydrogens
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)   # allow implicit Hs

        idx = rw_mol.AddAtom(atom)
        id_map[node_id] = idx

    # ---------- Add bonds ----------
    added = set()
    for i, neighbors in graph.edges.items():
        for j, data in neighbors.items():
            if (j, i) in added:
                continue

            bond_order = data.get("bond", 1)
            bond_type = {
                1: Chem.BondType.SINGLE,
                2: Chem.BondType.DOUBLE,
                3: Chem.BondType.TRIPLE
            }.get(bond_order, Chem.BondType.SINGLE)

            rw_mol.AddBond(id_map[i], id_map[j], bond_type)
            added.add((i, j))

    mol = rw_mol.GetMol()

    # ---------- Controlled sanitization ----------
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        )
    except Exception as e:
        print("Warning: skipped valence sanitization:", e)

    return mol

def graphnode_to_smiles(graph, canonical=True):
    mol = graphnode_to_rdkit_mol(graph)
    return Chem.MolToSmiles(
        mol,
        canonical=canonical,
        allHsExplicit=False   # ✅ this is enough
    )

def smiles_to_graphnode(smiles: str) -> GraphNode:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Kekulize to get explicit single/double bonds for aromatic rings
    Chem.Kekulize(mol, clearAromaticFlags=True)

    graph = GraphNode()
    idx_map = {}

    # ---------- Add atoms ----------
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()  # keep C uppercase for benzene
        charge = atom.GetFormalCharge()
        radical = atom.GetNumRadicalElectrons()

        node_id = graph.add_node(
            atom=symbol,
            tags=set(),
            charge=charge,
            radical=radical
        )

        idx_map[atom.GetIdx()] = node_id

    # ---------- Add bonds ----------
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
        else:
            order = 1  # aromatic treated as Kekulized

        tags = set()
        if bond.GetIsAromatic():
            tags.add("aromatic")  # optional, not changing symbol

        graph.add_edge(i, j, bond=order, tags=tags)

    # ---------- Calculate lone pairs, radicals, hydrogens ----------
    calculate_electron_state(graph)

    return graph



def draw_graph_with_rdkit(graph, filename="compound.png", size=(600, 400)):
    rw_mol = Chem.RWMol()
    atom_map = {}

    for node_id, atom_symbol in graph.nodes.items():
        # Keep halogens properly capitalized
        symbol = atom_symbol if atom_symbol in {"Cl", "Br", "I", "F"} else atom_symbol.upper()
        atom = Chem.Atom(symbol)
        # Mark aromatic atom if symbol is lowercase in GraphNode
        if atom_symbol.islower() and atom_symbol not in {"c", "n", "o"}:  # only carbons/hetero
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

def functional_group_distances(root: "TreeNode", target_label: str):
    """
    Correct functional group distances using backbone positions
    when a common parent chain exists.
    """

    FUNCTIONAL_GROUP_LABELS = {
        "carboxylic_acid",
        "aldehyde",
        "ketone",
        "alcohol",
        "cyano",
        "nitro",
        "chloro",
        "fluoro",
        "charged_carbon_1",
        "charged_carbon_-1"
    }

    parent = {}

    def build_parent(node):
        for child in getattr(node, "children", []):
            parent[child] = node
            build_parent(child)

    build_parent(root)

    functional_nodes = []

    def collect(node):
        if helper(node) in FUNCTIONAL_GROUP_LABELS:
            functional_nodes.append(node)
        for c in getattr(node, "children", []):
            collect(c)

    collect(root)

    targets = [n for n in functional_nodes if helper(n) == target_label]
    results = []

    def path_to_root(node):
        p = []
        while node:
            p.append(node)
            node = parent.get(node)
        return p

    for t in targets:
        path_t = path_to_root(t)

        for other in functional_nodes:
            if other is t:
                continue

            path_o = path_to_root(other)

            # Find lowest common ancestor
            lca = next((n for n in path_t if n in path_o), None)

            if lca and lca.label in ("mainchain", "cycle"):
                # Backbone-based distance
                dist = abs(t.pos - other.pos) + 1
            else:
                # Pure tree distance fallback
                dist = path_t.index(lca) + path_o.index(lca)
            
            results.append({
                "to_label": helper(other),
                "distance": dist
            })

    return results
def helper(x):
    if x.label == "halogen":
        return HALOGEN[x.atom]
    return "charged_carbon_" + str(x.charge) if x.label == "charged_carbon" else x.label
def group_halogens(fg_distances):
    """
    Group halogen atoms for identification (NOT IUPAC).

    Input:
        (acid_label, atom_symbol, distance)

    Output:
        (acid_label, grouped_label, distance)

    Example:
        ('COOH', 'Cl', 2), ('COOH', 'Cl', 2)
        → ('COOH', '2-Cl2', 2)
    """
    from collections import defaultdict

    grouped = defaultdict(int)

    # group by (acid, symbol, distance)
    for acid, symbol, dist in fg_distances:
        grouped[(acid, symbol, dist)] += 1

    result = []

    for (acid, symbol, dist), count in grouped.items():
        new_label = symbol
        if count > 1:
            new_label = f"{count}-{symbol}"
        result.append((acid, new_label, dist))

    return result


def build_tree(graph):
    tmp = build_tree_recursive(graph)
    normalize_carboxylic_acids(tmp)
    convert_carbaldehyde_nodes(tmp)
    return tmp
def all_atoms_neutral(graph):
    if not hasattr(graph, "charges"):
        return True  # no charge info → assume neutral

    return all(charge == 0 for charge in graph.charges.values())
def count_pi_bonds(graph):
    """
    Count the total number of pi bonds in a GraphNode compound.

    Double bond = 1 pi bond
    Triple bond = 2 pi bonds
    """
    counted = set()
    pi_count = 0

    for i, neighbors in graph.edges.items():
        for j, data in neighbors.items():
            if (j, i) in counted:
                continue  # avoid double-counting bonds
            bond_order = data.get("bond", 1)
            if bond_order == 2:
                pi_count += 1
            elif bond_order == 3:
                pi_count += 2
            counted.add((i, j))

    return pi_count
def get_charged_atoms(graph):
    """
    Returns a list of tuples for atoms with nonzero charge.
    Each tuple: (atom_symbol, charge)
    """
    charged_atoms = []
    for atom_id, atom_symbol in graph.nodes.items():
        ch = graph.charges.get(atom_id, 0)
        if ch != 0:
            charged_atoms.append((atom_symbol, ch))
    return charged_atoms

def find_internal_charge_pairs(graph: "GraphNode"):
    """
    Finds all internal electrostatic / lone-pair related pairs in a GraphNode.

    Returns:
      (distance, atom1_symbol, atom2_symbol, value1, value2, pair_type)
    """

    pairs = []
    seen = set()

    # ✅ FIX 1: correct attribute name
    charges = getattr(graph, "charges", {})
    lone_pairs = getattr(graph, "lone_pairs", {})

    positives = []
    negatives = []
    lp_atoms = []

    for atom_id, sym in graph.nodes.items():
        ch = charges.get(atom_id, 0)
        lp = lone_pairs.get(atom_id, 0)

        if ch > 0:
            positives.append((atom_id, sym, ch))
        elif ch < 0:
            negatives.append((atom_id, sym, ch))

        # ignore lone pairs on carbon
        if lp > 0 and sym.upper() != "C":
            lp_atoms.append((atom_id, sym, lp))


    # -------- BFS (cycle-safe) --------
    def bfs_distance(start, goal):
        visited = set()
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            if node == goal:
                return dist
            if node in visited:
                continue
            visited.add(node)
            for nbr in graph.edges.get(node, {}):
                if nbr not in visited:
                    queue.append((nbr, dist + 1))
        return float("inf")

    # -------- Pair generator --------
    def add_pairs(list1, list2, label):
        for a_id, a_sym, a_val in list1:
            for b_id, b_sym, b_val in list2:
                if a_id == b_id:
                    continue

                key = tuple(sorted((a_id, b_id))) + (label,)
                if key in seen:
                    continue
                seen.add(key)

                dist = bfs_distance(a_id, b_id)
                pairs.append((dist, a_sym, b_sym, a_val, b_val, label))

    # -------- All combinations --------
    add_pairs(positives, negatives, "positive_negative")
    add_pairs(positives, lp_atoms, "positive_lone_pair")
    add_pairs(positives, positives, "positive_positive")
    add_pairs(negatives, negatives, "negative_negative")
    add_pairs(negatives, lp_atoms, "negative_lone_pair")
    add_pairs(lp_atoms, lp_atoms, "lone_pair_lone_pair")

    pairs.sort(key=lambda x: x[0])
    return pairs

def condense_functional_groups(graph: "GraphNode", allow=[True,True,True,True,True]) -> "GraphNode":
    """
    Condenses specific functional groups into single pseudo-atoms.

    Supported:
      - COOH
      - OH
      - NO2
      - CCl3
      - OCH3

    Assumes:
      - implicit hydrogens
      - graph is cyclic-safe
    """

    def condense_oh(g):
        for o_id, sym in list(g.nodes.items()):
            if sym != "O":
                continue

            neighbors = list(g.edges.get(o_id, {}).items())
            if len(neighbors) != 1:
                continue

            ext, data = neighbors[0]
            if data.get("bond", 1) != 1:
                continue

            if g.charges.get(o_id, 0) != 0:
                continue

            new = g.add_node("OH", tags={"condensed"})
            g.charges[new] = 0
            g.lone_pairs[new] = 0
            g.radicals[new] = 0

            g.add_edge(new, ext, bond=1)
            g.remove_node(o_id)

    def condense_cooh(g):
        for c_id, sym in list(g.nodes.items()):
            if sym != "C":
                continue

            o_double = None
            o_single = None
            external = None

            for nbr, e in g.edges.get(c_id, {}).items():
                bond = e.get("bond", 1)
                if g.nodes[nbr] == "O" and bond == 2:
                    o_double = nbr
                elif g.nodes[nbr] == "O" and bond == 1:
                    o_single = nbr
                else:
                    external = nbr

            if not (o_double and o_single and external):
                continue

            if g.charges.get(o_single, 0) != 0:
                continue

            new = g.add_node("COOH", tags={"condensed"})
            g.charges[new] = 0
            g.lone_pairs[new] = 0
            g.radicals[new] = 0

            g.add_edge(new, external, bond=g.edges[c_id][external]["bond"])

            for x in (c_id, o_double, o_single):
                g.remove_node(x)

    def find_nitro_groups(g):
        out = []
        for n_id, sym in g.nodes.items():
            if sym != "N":
                continue

            o_ids = []
            orders = []

            for nbr, e in g.edges[n_id].items():
                if g.nodes[nbr] == "O":
                    o_ids.append(nbr)
                    orders.append(e.get("bond", 1))

            if len(o_ids) == 2 and sorted(orders) == [1, 2]:
                out.append((n_id, o_ids))
        return out

    def condense_no2(g):
        for n_id, o_ids in find_nitro_groups(g):
            external = None
            for nbr in g.edges[n_id]:
                if nbr not in o_ids:
                    external = nbr
                    break

            new = g.add_node("NO2", tags={"condensed"})
            g.charges[new] = 0
            g.lone_pairs[new] = 0
            g.radicals[new] = 0

            if external is not None:
                bond = g.edges[n_id][external]["bond"]
                g.add_edge(new, external, bond=bond)

            for x in o_ids + [n_id]:
                g.remove_node(x)

    def condense_ccl3(g):
        for c_id, sym in list(g.nodes.items()):
            if sym != "C":
                continue

            cl = []
            external = None

            for nbr, e in g.edges.get(c_id, {}).items():
                if g.nodes[nbr] == "Cl" and e.get("bond", 1) == 1:
                    cl.append(nbr)
                else:
                    external = nbr

            if len(cl) != 3:
                continue

            new = g.add_node("CCl3", tags={"condensed"})
            g.charges[new] = 0
            g.lone_pairs[new] = 0
            g.radicals[new] = 0

            if external is not None:
                bond = g.edges[c_id][external]["bond"]
                g.add_edge(new, external, bond=bond)

            for x in cl + [c_id]:
                g.remove_node(x)

    def condense_och3(g):
        """
        Condense terminal methoxy groups -OCH3 into a single 'OCH3' node.
        """
        for o_id, sym in list(g.nodes.items()):
            if sym != "O":
                continue

            neighbors = list(g.edges.get(o_id, {}).items())
            if len(neighbors) != 2:
                continue  # O must have exactly two neighbors: one main chain, one methyl C

            main_ext = None
            methyl_c = None
            for nbr, data in neighbors:
                if g.nodes[nbr] == "C" and len(g.edges.get(nbr, {})) == 1:
                    methyl_c = nbr
                else:
                    main_ext = nbr

            if not (main_ext and methyl_c):
                continue

            new = g.add_node("OCH3", tags={"condensed"})
            g.charges[new] = 0
            g.lone_pairs[new] = 0
            g.radicals[new] = 0

            g.add_edge(new, main_ext, bond=g.edges[o_id][main_ext]["bond"])
            g.remove_node(o_id)
            g.remove_node(methyl_c)

    # ---------- Iterative condensation ----------
    g = graph.copy()
    changed = True

    while changed:
        before = len(g.nodes)
        if allow[0]:
            condense_cooh(g)
        if allow[1]:
            condense_no2(g)
        if allow[2]:
            condense_ccl3(g)
        if allow[3]:
            condense_oh(g)
        if len(allow) > 4 and allow[4]:
            condense_och3(g)

        changed = len(g.nodes) != before

    return g


def distances_from_acidic_groups(graph: "GraphNode"):
    """
    For each COOH or OH atom, compute distances to:
      - other condensed functional groups
      - halogen atoms

    Returns a list of tuples:
      (source_atom_symbol, target_atom_symbol, distance)
    """

    results = []

    # -------- Identify reference atoms --------
    sources = [
        (aid, sym)
        for aid, sym in graph.nodes.items()
        if sym in {"COOH", "OH"}
    ]

    if not sources:
        return []

    # -------- Identify target atoms --------
    halogens = {"F", "Cl", "Br", "I"}

    targets = []
    for aid, sym in graph.nodes.items():
        if sym in halogens:
            targets.append((aid, sym))
        elif "condensed" in graph.node_tags.get(aid, set()):
            targets.append((aid, sym))

    # remove self-pairs later
    target_ids = {aid for aid, _ in targets}

    # -------- BFS distance (cycle-safe) --------
    def bfs_all_distances(start):
        dist = {start: 0}
        q = deque([start])

        while q:
            u = q.popleft()
            for v in graph.edges.get(u, {}):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    # -------- Compute distances --------
    for src_id, src_sym in sources:
        dist_map = bfs_all_distances(src_id)

        for tgt_id, tgt_sym in targets:
            if tgt_id == src_id:
                continue
            if tgt_id in dist_map:
                results.append(
                    (src_sym, tgt_sym, dist_map[tgt_id])
                )

    # sort by source then distance
    results.sort(key=lambda x: (x[0], x[2]))
    return results

def print_graphnode(graph: "GraphNode"):
    """
    Print a detailed view of a GraphNode.
    Shows:
      - Node ID, symbol, charge, radical, lone pairs, tags
      - Edges with bond orders
    """
    print("=== GraphNode ===")
    print("Nodes:")
    for node_id, sym in graph.nodes.items():
        ch = getattr(graph, "charges", {}).get(node_id, 0)
        rad = getattr(graph, "radicals", {}).get(node_id, 0)
        lp = getattr(graph, "lone_pairs", {}).get(node_id, 0)
        tags = graph.node_tags.get(node_id, set())
        print(f"  {node_id}: {sym}, charge={ch}, radical={rad}, lone_pairs={lp}, tags={tags}")

    print("\nEdges:")
    for i, nbrs in graph.edges.items():
        for j, data in nbrs.items():
            bond = data.get("bond", 1)
            tags = data.get("tags", set())
            print(f"  {i} - {j}, bond={bond}, tags={tags}")
    print("================\n")

def identify_ortho_meta_para_graph(graph: "GraphNode"):
    """
    Identify ortho/meta/para substituents relative to OH or COOH
    ONLY if the ring is benzene.
    Returns: list of (relation, substituent_symbol)
    """
    # -----------------------------
    # 1. Detect benzene ring
    # -----------------------------
    ring = None
    for n in graph.nodes:
        if graph.nodes[n] != "C":
            continue
        # get neighbors in ring (must have 2)
        neighbors = [nbr for nbr in graph.edges[n] if graph.nodes[nbr] == "C"]
        if len(neighbors) != 2:
            continue
        # naive cycle check: 6 connected carbons
        visited = [n]
        current = neighbors[0]
        prev = n
        while len(visited) < 6:
            visited.append(current)
            next_c = [nbr for nbr in graph.edges[current] if graph.nodes[nbr] == "C" and nbr != prev]
            if not next_c:
                break
            prev, current = current, next_c[0]
        if len(visited) == 6 and all(graph.nodes[x] == "C" for x in visited):
            ring = visited
            break
    if not ring:
        return None
    
    ring_set = set(ring)
    ring_len = 6

    # -----------------------------
    # 2. Find reference OH / COOH
    # -----------------------------
    ref_atom = None
    for sub_id, sym in graph.nodes.items():
        if sym not in {"OH", "COOH"}:
            continue
        for nbr in graph.edges[sub_id]:
            if nbr in ring_set:
                ref_atom = nbr
                break
        if ref_atom is not None:
            break
    if ref_atom is None:
        return []

    # -----------------------------
    # 3. Ring-only BFS distances
    # -----------------------------
    distances = {ref_atom: 0}
    queue = deque([ref_atom])
    while queue:
        u = queue.popleft()
        for v in graph.edges[u]:
            if v in ring_set and v not in distances:
                distances[v] = distances[u] + 1
                queue.append(v)

    # -----------------------------
    # 4. Identify substituents
    # -----------------------------
    results = []
    for ring_atom in ring:
        if ring_atom == ref_atom:
            continue
        for nbr in graph.edges[ring_atom]:
            if nbr in ring_set:
                continue
            sym = graph.nodes[nbr]
            if sym in {"Cl", "Br", "F", "I"} or "condensed" in graph.node_tags.get(nbr, set()):
                d = distances.get(ring_atom)
                if d is None:
                    continue
                d = min(d, ring_len - d)
                if d == 1:
                    rel = "ortho"
                elif d == 2:
                    rel = "meta"
                elif d == 3:
                    rel = "para"
                else:
                    continue
                results.append((rel, sym))
    return results

def condense_ch3(graph: "GraphNode") -> "GraphNode":
    """
    Condense terminal methyl groups (-CH3) into a single 'CH3' node.

    Assumptions:
    - No explicit hydrogen atoms in the graph
    - CH3 carbon has exactly ONE neighbor
    """

    g = graph.copy()

    for c_id, sym in list(g.nodes.items()):
        if sym != "C":
            continue

        neighbors = list(g.edges.get(c_id, {}).items())

        # CH3 must be terminal
        if len(neighbors) != 1:
            continue

        ext, data = neighbors[0]

        # must be single bonded
        if data.get("bond", 1) != 1:
            continue

        # do NOT condense if attached to heteroatom directly
        # (prevents methanol carbon collapsing incorrectly)
        if g.nodes[ext] in {"O", "N", "S"}:
            continue

        # --- create condensed node ---
        new_id = g.add_node("CH3", tags={"condensed"})
        g.charges[new_id] = 0
        g.radicals[new_id] = 0
        g.lone_pairs[new_id] = 0

        # reconnect
        g.add_edge(new_id, ext, bond=1)

        # remove original carbon
        g.remove_node(c_id)

    return g

def inductive_effect_acid(fg_dist_a, fg_dist_b):
    INDUCTIVE_STRENGTH = {
        "NO2": 5,
        "CN": 4,
        "F": 3,
        "Cl": 2,
        "Br": 1.5,
        "I": 1,
        "2-Cl": 3.25,
        "3-Cl": 3.5,
        "CCl3": 3.5
    }
    sgn = 1
    if len(fg_dist_a)+len(fg_dist_b) not in [1,2] or abs(len(fg_dist_a)-len(fg_dist_b)) > 1:
        return 0
    if len(fg_dist_a)==len(fg_dist_b):
        if fg_dist_a[0][2] == fg_dist_b[0][2]:
            if fg_dist_a[0][1] == fg_dist_b[0][1]:
                return 0
            if INDUCTIVE_STRENGTH.get(fg_dist_a[0][1], 1) > INDUCTIVE_STRENGTH.get(fg_dist_b[0][1], 1):
                return 1
            return -1
        elif fg_dist_a[0][1] == fg_dist_b[0][1]:
            sgn2 = 1 if INDUCTIVE_STRENGTH.get(fg_dist_a[0][1], 1) > 0 else -1
            if fg_dist_a[0][2] > fg_dist_b[0][2]:
                return sgn2
            return -sgn2
        else:
            return 0
    elif len(fg_dist_a)==1:
        if INDUCTIVE_STRENGTH.get(fg_dist_a[0][1], 1) > 0:
            return 1
        return -1
    elif len(fg_dist_b)==1:
        if INDUCTIVE_STRENGTH.get(fg_dist_b[0][1], 1) > 0:
            return -1
        return 1
    return 0
def compare_acidic_strength(graph_a: "GraphNode", graph_b: "GraphNode") -> int:
    con_a = condense_functional_groups(graph_a)
    con_b = condense_functional_groups(graph_b)
    orig_a = con_a.copy()
    orig_b = con_b.copy()
    con_a = condense_ch3(con_a)
    con_b = condense_ch3(con_b)
    
    fg_dist_a = group_halogens(distances_from_acidic_groups(con_a))
    fg_dist_b = group_halogens(distances_from_acidic_groups(con_b))

    cy_a = identify_ortho_meta_para_graph(con_a)
    cy_b = identify_ortho_meta_para_graph(con_b)
    
    if cy_a is not None and cy_b is not None:
        cy = Counter(cy_a) & Counter(cy_b)
        cy_a = list(Counter(cy_a) - cy)
        cy_b = list(Counter(cy_b) - cy)

        if len(cy_a) == 1 and len(cy_b) == 1 and cy_a[0][0] == cy_b[0][0] and cy_a[0][0] != "meta" and cy_a[0][1] in ["F", "Cl", "Br", "I"] and\
           cy_b[0][1] in ["F", "Cl", "Br", "I"]:
            return -inductive_effect_acid(fg_dist_a, fg_dist_b)
        if len(cy_a) == 1 and len(cy_b) == 1 and cy_a[0][1] == cy_b[0][1]:
            if cy_a[0][1] == "OCH3":
                score = {"meta":3, "ortho":2, "para":1}
                s_a = score[cy_a[0][0]]
                s_b = score[cy_b[0][0]]
                if s_a > s_b:
                    return 1
                if s_a < s_b:
                    return -1
            if cy_a[0][0] == "ortho":
                return 1
            if cy_b[0][0] == "ortho":
                return -1
        lst = []
        for item in [cy_a, cy_b]:
            other = (None, None)
            for item2 in item:
                if item2[0] != "meta":
                    if item2[1] == "NO2":
                        other = ("-M", "-I")
                    elif item2[1] == "CCl3":
                        other = ("-I", "-M")
                    elif item2[1] == "CH3":
                        other = ("+H", "+I")
                    elif item2[1] == "OCH3":
                        other = ("+M", "-I")
                    elif item2[1] in ["Cl", "F", "Br", "I"]:
                        other = ("-I", "+M")
                else:
                    if item2[1] == "NO2":
                        other = ("-I", None)
                    elif item2[1] == "CCl3":
                        other = ("-I", None)
                    elif item2[1] == "CH3":
                        other = ("+H", "+I")
                    elif item2[1] == "OCH3":
                        other = ("-I", None)
                    elif item2[1] in ["Cl", "F", "Br", "I"]:
                        other = ("-I", None)
            lst.append(other)
            
        effect_a = lst[0]
        effect_b = lst[1]
        def score(x):
            if x is None:
                return 0
            sgn = -1
            if x[0] == "-":
                sgn = 1
            return sgn * "k I H M".split(" ").index(x[1])
        for i in range(2):
            if score(effect_a[i]) > score(effect_b[i]):
                return 1
            elif score(effect_a[i]) < score(effect_b[i]):
                return -1
            
            if effect_a[i] in ["+I", "-I"]:
                
                return inductive_effect_acid(fg_dist_a, fg_dist_b)
            if i == 0 and effect_a[i] == "-M" and len(cy_a) == 1 and len(cy_b) == 1 and cy_a[0][1] == "NO2" and cy_b[0][1] == "NO2":
                if cy_a[0][0] == "para":
                    return 1
                if cy_b[0][0] == "para":
                    return -1
    fg_dist_a = group_halogens(distances_from_acidic_groups(orig_a))
    fg_dist_b = group_halogens(distances_from_acidic_groups(orig_b))
        
    return inductive_effect_acid(fg_dist_a, fg_dist_b)
def compare_stability(graph_a: "GraphNode", graph_b: "GraphNode") -> int:

    if all_atoms_neutral(graph_a) and not all_atoms_neutral(graph_b):
        return 1
    if not all_atoms_neutral(graph_a) and all_atoms_neutral(graph_b):
        return -1
    if count_pi_bonds(graph_a) > count_pi_bonds(graph_b):
        return 1
    if count_pi_bonds(graph_a) < count_pi_bonds(graph_b):
        return -1
    ca = set(get_charged_atoms(graph_a))
    cb = set(get_charged_atoms(graph_b))
    
    c = ca & cb
    ca = list(ca - c)
    cb = list(cb - c)
    stability_score = {"C":1, "N":2, "O":3, "S":4}
    def c_score(ca, cb, index_a, index_b):
        if ca[index_a][1] == cb[index_b][1] and ca[index_a][0] != cb[index_b][0]:
            if ca[index_a][1] == 0:
                return 0
            sgn = -1 if ca[index_a][1] < 0 else 1
            if stability_score[ca[index_a][0]] > stability_score[cb[index_b][0]]:
                return -sgn
            if stability_score[ca[index_a][0]] < stability_score[cb[index_b][0]]:
                return sgn
        return 0
    if len(ca) == len(cb):
        if len(ca) == 2:
            for item in [((1,1),(2,2)),((1,2),(2,1))]:
                tmp = set([c_score(ca, cb, item2[0]-1, item2[1]-1) for item2 in item])
                if 0 not in tmp and len(tmp) == 1:
                    return list(tmp)[0]
        elif len(ca) == 1:
            tmp = c_score(ca, cb, 0, 0)
            if tmp != 0:
                return tmp
            
    for item in ["positive_negative","positive_lone_pair","positive_positive","negative_negative","negative_lone_pair","lone_pair_lone_pair"]:
        
        sa = find_internal_charge_pairs(graph_a)
        sb = find_internal_charge_pairs(graph_b)
        
        sa, sb = [set([item3 for item3 in item2 if item3[-1] == item]) for item2 in [sa, sb]]
        
        s = sa & sb
        
        sa = list(sa - s)
        sb = list(sb - s)
        
        if len(sa) == 1 and len(sb) == 1 and all(sa[0][i] == sb[0][i] for i in range(1,len(sa[0]))):
            sgn = 1
            if item in ["positive_negative","positive_lone_pair"]:
                sgn = -1
            if sa[0][0] < sb[0][0]:
                return sgn
            if sa[0][0] > sb[0][0]:
                return -sgn
    
    ha = count_hyperconjugation(graph_a)
    hb = count_hyperconjugation(graph_b)
    if ha > hb:
        return 1
    elif ha < hb:
        return -1
    
    tree_a = build_tree(graph_a)
    tree_b = build_tree(graph_b)
    
    acid_labels = {"charged_carbon"}
    acids_a = [c for c in tree_a.children if c.label in acid_labels]
    acids_b = [c for c in tree_b.children if c.label in acid_labels]

    if not acids_a or not acids_b:
        return 0

    type_a = helper(acids_a[0])
    type_b = helper(acids_b[0])
    if type_a != type_b:
        return 0
    
    fg_dist_a = group_halogens(functional_group_distances(tree_a, target_label=type_a))
    fg_dist_b = group_halogens(functional_group_distances(tree_b, target_label=type_b))
    
    INDUCTIVE_STRENGTH = {
        "nitro": 5,
        "cyano": 4,
        "fluoro": 3,
        "chloro": 2,
        "bromo": 1.5,
        "iodo": 1,
        "dichloro": 3.25
    }
    sgn = 1
    if type_a == "charged_carbon_-1":
        sgn = -1
    if len(fg_dist_a)+len(fg_dist_b) not in [1,2] or abs(len(fg_dist_a)-len(fg_dist_b)) > 1:
        return 0
    if len(fg_dist_a)==len(fg_dist_b):
        if fg_dist_a[0]["distance"] == fg_dist_b[0]["distance"]:
            if fg_dist_a[0]["to_label"] == fg_dist_b[0]["to_label"]:
                return 0
            if INDUCTIVE_STRENGTH.get(fg_dist_a[0]["to_label"], 1) > INDUCTIVE_STRENGTH.get(fg_dist_b[0]["to_label"], 1):
                return -sgn
            return sgn
        elif fg_dist_a[0]["to_label"] == fg_dist_b[0]["to_label"]:
            sgn2 = 1 if INDUCTIVE_STRENGTH.get(fg_dist_a[0]["to_label"], 1) > 0 else -1
            if fg_dist_a[0]["distance"] > fg_dist_b[0]["distance"]:
                return sgn * sgn2
            return -sgn * sgn2
        else:
            return 0

def is_carbon(graph, n):
    return graph.nodes[n].upper() == "C"

def is_acceptor_center(graph, n):
    # carbocation or radical
    if graph.charges.get(n, 0) > 0:
        return True
    if graph.radicals.get(n, 0) > 0:
        return True

    # π systems
    for nbr, data in graph.edges[n].items():
        if data.get("bond", 1) >= 2:
            return True
    return False
def count_hyperconjugation(graph, include_CC=False):
    """
    Counts hyperconjugative interactions (σ-donation possibilities),
    not resonance structures.
    """

    count = 0

    for acc in graph.nodes:
        if not is_carbon(graph, acc):
            continue
        if not is_acceptor_center(graph, acc):
            continue

        # look at adjacent carbons (β-carbons)
        for beta, data in graph.edges[acc].items():
            if not is_carbon(graph, beta):
                continue

            # --- C–H hyperconjugation ---
            h_count = graph.attached_h.get(beta, 0)
            count += h_count

            # --- optional C–C hyperconjugation ---
            if include_CC:
                for nbr2, data2 in graph.edges[beta].items():
                    if nbr2 == acc:
                        continue
                    if is_carbon(graph, nbr2) and data2.get("bond", 1) == 1:
                        count += 1

    return count

def custom_sort(items, cmp):
    # object → letter
    labels = {obj: string.ascii_lowercase[i] for i, obj in enumerate(items)}

    score = {obj: 0 for obj in items}

    for a in items:
        for b in items:
            if a is b:
                continue

            c = cmp(a, b)

            if c == 1:
                score[a] += 1
                score[b] -= 1
            elif c == -1:
                score[a] -= 1
                score[b] += 1
            # c == 0 → ignore

    # group by score
    buckets = {}
    for obj, s in score.items():
        buckets.setdefault(s, []).append(labels[obj])

    # highest dominance first
    return [buckets[s] for s in sorted(buckets, reverse=True)]

def iupac(graph, debug=False):
    tmp = build_tree(graph)
    if debug:
        print(tmp)
    return remove_unnecessary_hyphens(tree_to_iupac(tmp))
def smiles(string):
    return smiles_to_graphnode(string)
def draw(graph, filename="compound.png", size=(300, 200)):
    draw_graph_with_rdkit(graph, filename, size)
