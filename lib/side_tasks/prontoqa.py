import random
import re

from lib import prontoqa

from .base import SideTaskDefinition, partition_split


# "relevant" distractors add unreachable ontology branches and near-miss rules connected to the
# same concepts, so the proof chain is not the whole context: the model must actually reason about
# which edges are reachable from the explored region. Premise indices then span the full (variable-
# sized) premise list, so proof_trace writes each index in minimal-width binary delimited by 2 -
# the alphabet stays [0, 1, 2] regardless of context size.
# "random" ordering is required for proof_trace to be non-trivial: with the default postorder the
# proof-path premises land in a predictable position, which leaks the answer.
_DISTRACTORS = "relevant"
_ORDERING = "random"


def _split_premises(question: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.]) ", question.strip()) if s.strip()]


def _answer_bit(example: dict) -> int:
    return 1 if example["answer"] == "True" else 0


def _encode_indices(indices: list[int]) -> list[int]:
    """Each premise index as a minimal-width binary number (MSB first, no leading zeros),
    consecutive indices separated by the delimiter value 2. E.g. [5, 2] -> [1, 0, 1, 2, 1, 0]."""
    out = []
    for j, idx in enumerate(indices):
        if j:
            out.append(2)
        out.extend(int(b) for b in format(idx, "b"))
    return out


def _explore_trace(question: str, chain_of_thought: list[str]) -> list[int]:
    """The chain-of-thought interleaves derived facts (even positions) with the rule applied to
    reach them (odd positions). Exploration starts at the base fact (chain_of_thought[0]) and then
    applies one rule per proof step; the base fact and every applied rule appear verbatim among the
    premises. The trace binary-encodes, in explore order, the premise index of the base fact (the
    first edge explored) followed by each applied rule, delimiting entries with 2."""
    premises = _split_premises(question)
    explored = [chain_of_thought[0]] + chain_of_thought[1::2]
    return _encode_indices([premises.index(step) for step in explored])


def _generate(hops: int, n_samples: int, seed: int, ordering: str) -> list[dict]:
    examples = prontoqa.generate(
        hops=hops, num_trials=n_samples, few_shot_examples=0,
        distractors=_DISTRACTORS, test_distractors=_DISTRACTORS, ordering=ordering, seed=seed)
    return [e["test_example"] for e in examples]


def _numbered_input(question: str, query: str) -> str:
    premises = "\n".join(f"{i}. {s}" for i, s in enumerate(_split_premises(question)))
    return f"{premises}\n\n{query}"


def _answer_only_data(hops: int, n_samples: int, seed: int) -> list[dict]:
    data = []
    for ex in _generate(hops, n_samples, seed, _ORDERING):
        bit = _answer_bit(ex)
        data.append({"input": f"{ex['question']}\n\n{ex['query']}", "reasoning": [bit], "output": [bit]})
    return data


def _proof_trace_data(hops: int, n_samples: int, seed: int) -> list[dict]:
    data = []
    for ex in _generate(hops, n_samples, seed, _ORDERING):
        trace = _explore_trace(ex["question"], ex["chain_of_thought"])
        # base fact + hops rules = hops+1 entries, so hops delimiters between them
        assert trace.count(2) == hops, f"expected {hops} delimiters, got {trace.count(2)}"
        bit = _answer_bit(ex)
        # the reasoning concludes with the answer bit after a final delimiter, so the trace
        # carries both the exploration and the True/False conclusion it reaches
        data.append({
            "input": _numbered_input(ex["question"], ex["query"]),
            "reasoning": trace + [2, bit],
            "output": [bit],
        })
    return data


def _answer_only_explanation() -> str:
    return (
        "You will receive a set of premises followed by a query of the form "
        '"True or false: ...". Determine whether the queried statement follows from the '
        "premises by logical deduction. Encode a single value as both your reasoning and "
        "output: 1 if the statement is true, 0 if it is false."
    )


def _proof_trace_explanation(hops: int) -> str:
    example = _encode_indices([5, 2])  # illustrative: base fact premise 5, then first rule premise 2
    return (
        "You will receive a list of numbered premises followed by a query of the form "
        '"True or false: ...". Exactly one premise is a base fact about a single entity '
        '(e.g. "Alex is a wumpus."); the rest are universally-quantified rules '
        '(e.g. "Every wumpus is a jompus."), and some of the rules are distractors. Treat the '
        'premises as a directed graph whose nodes are categories and whose edges are the rules (a '
        'rule "Every A is B" is an edge from A to B); the base fact places the entity at one '
        "starting node. Prove the query by exploring this graph outward from that node: at each "
        "step, consider the states already reached and tracked in your reasoning so far, look at "
        "the edges (rules) leaving that explored region, and apply one such reachable rule to reach "
        f"a new state, for {hops} steps in total. Ignore distractor premises that are not reachable "
        "from your starting node. Emit your reasoning as premise numbers in the order you explore "
        "them, each written in binary (most significant bit first, no leading zeros) and separated "
        "by the delimiter value 2: first the number of the base-fact premise (the first edge you "
        f"explore, fixing your starting node), then the number of the rule you apply at each of the "
        f"{hops} steps. Conclude your reasoning with one more delimiter 2 followed by the answer "
        "reached: 1 if the queried statement is true, 0 if it is false. Then output that same single "
        "value (1 if true, 0 if false). For example, if the base fact is premise 5 and the first "
        f"rule you apply is premise 2, your reasoning begins {example} (premise 5 = binary "
        f"{format(5, 'b')}, delimiter 2, premise 2 = binary {format(2, 'b')}), continuing with the "
        "remaining steps and ending with a delimiter 2 and the answer bit."
    )


# The chain strategy replaces the vendored generator's rejection sampling (whose acceptance
# rate collapses beyond ~5 hops) with a direct construction: n_chains disjoint rule chains
# of length hops, the entity's base fact attached to one of them, and the query naming the
# endpoint of either the entity's chain (answer 1) or another chain (answer 0). The two
# cases are structurally identical — every endpoint terminates a length-hops chain and is
# mentioned exactly once, names are fresh nonsense words, premise order is shuffled — so no
# local statistic separates them: the answer requires connecting the entity to an endpoint.
# Out-degree is at most 1 everywhere, so the entity's walk is deterministic and the gold
# trace (base-fact index, then each followed rule's index) is unique.
_CHAIN_ENTITIES = ["Alex", "Wren", "Sam", "Max", "Polly", "Stella", "Fae", "Rex"]
_CHAIN_NAME_PARTS = (
    ["wor", "yer", "zil", "drom", "quim", "bler", "snor", "grum", "tral", "clop",
     "brin", "stel", "frap", "glin", "hax", "jun", "kel", "lum", "mip", "ned",
     "pog", "quat", "rup", "sib", "tev", "vun", "wex", "yob", "zam", "crul"],
    ["", "al", "or", "in", "um", "ee", "ath"],
    ["pus", "ple", "git", "per", "pist", "pant", "por", "pee", "pin"],
)


def _chain_name_pool() -> list[str]:
    a, b, c = _CHAIN_NAME_PARTS
    return [x + y + z for x in a for y in b for z in c]


def _chain_example(rng: random.Random, hops: int, n_chains: int, pool: list[str]) -> dict:
    names = rng.sample(pool, n_chains * (hops + 1))
    chains = [names[i * (hops + 1):(i + 1) * (hops + 1)] for i in range(n_chains)]
    entity = rng.choice(_CHAIN_ENTITIES)
    answer = rng.randint(0, 1)
    target = chains[0][-1] if answer else rng.choice(chains[1:])[-1]

    base = f"{entity} is a {chains[0][0]}."
    rules = {(c[i], c[i + 1]): f"Every {c[i]} is a {c[i + 1]}."
             for c in chains for i in range(hops)}
    premises = [base] + list(rules.values())
    rng.shuffle(premises)
    followed = [base] + [rules[(chains[0][i], chains[0][i + 1])] for i in range(hops)]
    trace = _encode_indices([premises.index(p) for p in followed])

    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(premises))
    query = f"True or false: {entity} is a {target}."
    return {
        "input": f"{numbered}\n\n{query}",
        "reasoning": trace + [2, answer],
        "output": [answer],
    }


def _chain_data(hops: int, n_chains: int, n_samples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pool = _chain_name_pool()
    return [_chain_example(rng, hops, n_chains, pool) for _ in range(n_samples)]


def _chain_explanation(hops: int) -> str:
    example = _encode_indices([5, 2])
    return (
        "You will receive a list of numbered premises followed by a query of the form "
        '"True or false: ...". Exactly one premise is a base fact placing an entity in a '
        'category (e.g. "Alex is a wumpus."); the rest are rules of the form "Every A is a '
        'B.". Each category has at most one rule leading out of it, so from the entity\'s '
        f"starting category there is exactly one chain of {hops} rules to follow; the other "
        "rules form separate chains that the entity's chain never touches. Follow the "
        "entity's chain to its final category. The queried category is the final category "
        "of one of the chains; the answer is 1 if it is the final category of the entity's "
        "chain, 0 if it belongs to a different chain. Emit your reasoning as premise "
        "numbers in the order you use them, each written in binary (most significant bit "
        "first, no leading zeros) and separated by the delimiter value 2: first the number "
        f"of the base-fact premise, then the number of each rule you follow, for {hops} "
        "rules in total. Conclude your reasoning with one more delimiter 2 followed by the "
        "answer: 1 if the queried category is the entity's final category, 0 if it is not. "
        "Then output that same single value. For example, if the base fact is premise 5 and "
        f"the first rule you follow is premise 2, your reasoning begins {example} (premise "
        f"5 = binary {format(5, 'b')}, delimiter 2, premise 2 = binary {format(2, 'b')}), "
        "continuing with the remaining rules and ending with a delimiter 2 and the answer."
    )


# DESIGN RATIONALE for the horn task (conjunctive forward chaining over a shared category
# pool). Each property below removes a specific shortcut; full analysis in
# experiments/theory/problems/horn/.
# (a) Rule bodies are TWO-premise conjunctions ("Everything that is a A and a B is a C.").
#     Single-premise rules make the task a reachability chain — L-class, solvable in log-span
#     by path doubling; the conjunction is the jump to the P-complete format.
# (b) The free body leg of the rule deriving c_i reaches back over the WHOLE chain (j uniform
#     over the valid values in [0, i-2]), never a bounded window: any bounded window bounds
#     the derivation DAG's bandwidth, so a constant-size frontier automaton squared O(log t)
#     times recovers the whole trace (frontier doubling) and the task collapses to ~log t.
# (c) Uniformly random UNRESTRICTED back-references alone are also wrong: they form a skip
#     list, so a backward greedy chase from the target reaches the base in O(log t) steps.
#     What allows far references safely is the pair-privacy promise P1 plus the shared pool:
#     every rule body pair {c_j, c_{i-1}} lies in exactly one chain's category set (enforced
#     by rejection sampling over j), while every single category lies on exactly `rho`
#     chains — a backward chase branches rho ways per level, and pruning a branch means
#     deciding co-derivability, i.e. the original problem.
# (d) Categories are shared across chains, every rule genuinely fires for some entity, and
#     there are NO distractor rules: dead rules are syntactically filterable, and per-chain
#     namespaces make chain membership a recoverable partition of categories. Sharing also
#     makes the premise graph one component and puts every terminal in every entity's
#     one-leg-relaxed closure, so connectivity and AND-as-OR relaxation decide nothing. Only
#     the terminals c_{hops+1} are private, symmetrically: each heads exactly one rule,
#     appears in no body, and is mentioned exactly once.
# (e) Pair-privacy is also what makes the derivation order unique WITHOUT distractors: a rule
#     can only fire for its own chain's entity, and pinning the newest category c_{i-1} keeps
#     exactly one rule applicable per step, so the gold trace is well defined.
# (f) Constraints: hops >= 16 (the P1 rejection sampling converges only there, and frontier
#     doubling still partially bites at hops=8); n_chains > rho (at n_chains == rho every
#     body pair lies in every chain, so P1 is unsatisfiable); rho >= 2 (rho = 1 partitions
#     the categories per chain); n_chains * (hops + 1) % rho == 0 (every non-terminal on
#     exactly rho chains); tested default n_chains=4 (n_chains=3's rho-overlap block
#     structure is clusterable).
# Honest residual: this is a promise family — the design removes the known shortcut classes,
# not every sub-linear-span computation in principle.
#
# SURFACE. The logical structure above is rendered by one of two surfaces. "prose" is the
# original English rendering. "compact" says exactly the same thing in a symbolic notation over
# single-token category words, which is a pure rendering change: same structure, same P1
# sampling, same trace, same answer distribution — only the characters differ.
_HORN_SURFACES = {
    "prose": ("{e} is a {c}.", "Everything that is a {a} and a {b} is a {h}.",
              "True or false: {e} is a {t}."),
    "compact": ("{e} : {c}", "{a} {b} > {h}", "? {e} : {t}"),
}

# Common concrete English nouns that the Qwen tokenizer encodes as a SINGLE token when they
# appear mid-line, i.e. with a leading space (" snow" -> 1 token, while the generated nonsense
# name " wumpus" costs 3). Guarded by test_compact_categories_are_single_tokens.
_COMPACT_CATEGORIES = [
    "snow", "wind", "fire", "gold", "iron", "steel", "copper", "rock", "sand", "clay",
    "dust", "ash", "coal", "salt", "water", "ice", "steam", "smoke", "rain", "storm",
    "thunder", "fog", "mist", "dew", "sun", "moon", "star", "earth", "soil", "grass", "leaf",
    "branch", "root", "flower", "vine", "moss", "oak", "pine", "maple", "cedar", "palm",
    "apple", "pear", "plum", "peach", "grape", "lemon", "lime", "orange", "berry", "olive",
    "fig", "bread", "cheese", "butter", "cream", "honey", "jam", "pie", "bun", "roll",
    "soup", "stew", "rice", "meat", "fish", "beef", "pork", "chicken", "egg", "bacon", "ham",
    "pepper", "garlic", "onion", "tomato", "carrot", "bean", "corn", "wheat", "oat", "tea",
    "coffee", "juice", "beer", "cider", "dog", "cat", "horse", "cow", "sheep", "goat",
    "duck", "hen", "mouse", "rat", "fox", "wolf", "bear", "lion", "tiger", "leopard",
    "camel", "ape", "whale", "shark", "dolphin", "eagle", "hawk", "owl", "robin", "crane",
    "snake", "lizard", "frog", "crab", "lobster", "shrimp", "jelly", "worm", "bee", "ant",
    "fly", "beetle", "spider", "house", "home", "room", "door", "window", "wall", "roof",
    "ceiling", "stair", "gate", "fence", "yard", "barn", "shed", "garage", "bridge",
    "tunnel", "road", "path", "lane", "track", "city", "town", "village", "field", "forest",
    "wood", "mountain", "valley", "river", "sea", "ocean", "beach", "shore", "cliff",
    "desert", "swamp", "table", "chair", "bench", "shelf", "bed", "couch", "lamp", "clock",
    "mirror", "curtain", "pillow", "cup", "mug", "plate", "bowl", "knife", "spoon", "pot",
    "kettle", "jar", "bottle", "can", "bag", "basket", "bucket", "book", "page", "paper",
    "pen", "ink", "brush", "paint", "chalk", "note", "card", "map", "knot", "rope", "chain",
    "wire", "screw", "bolt", "hammer", "saw", "drill", "blade", "needle", "thread", "cloth",
    "cotton", "silk", "leather", "glass", "plastic", "metal", "shirt", "coat", "hat", "boot",
    "shoe", "sock", "belt", "scarf", "dress", "suit", "vest", "car", "truck", "bus", "train",
    "ship", "plane", "bike", "wagon", "sled", "canoe", "key", "lock", "ring", "bill",
    "purse", "wallet", "band", "pin", "badge", "flag", "torch", "candle", "drum", "flute",
    "horn", "whistle", "guitar", "piano", "bone", "skin", "hair", "eye", "nose", "mouth",
    "tooth", "neck", "arm", "hand", "finger", "foot", "knee", "back", "heart", "brain",
    "blood", "nerve", "game", "ball", "bat", "net", "race", "prize", "medal", "team",
    "spice", "mint", "basil", "sage", "gem", "pearl", "ruby", "amber", "coral", "marble",
    "granite", "bank", "shop", "store", "mall", "bar", "cafe", "inn", "school", "church",
    "castle", "palace", "fort", "port", "dock", "pier", "wave", "tide", "flood", "quake",
    "flame", "spark", "glow", "shade", "shadow", "dark", "dawn", "dusk", "day", "week",
    "month", "hour", "minute", "king", "queen", "prince", "guard", "scout", "hunter",
    "baker", "smith", "miner", "pilot", "judge", "chef", "clerk", "guest", "letter", "word",
    "name", "dot", "mark", "sign", "list", "plan", "form", "shape", "curve",
    "square", "angle", "point", "edge", "side", "bubble", "foam", "drop", "stream", "creek",
    "spring", "pond", "canal", "grain", "flour", "dough", "crust", "slice", "tail", "wing",
    "claw", "scale", "feather", "shell", "hide", "paw", "nest", "hive", "web", "ribbon",
    "lace", "patch", "button", "zip", "seam", "cuff", "sleeve", "tool", "crank", "gear",
    "axle", "valve", "pipe", "tank", "pump", "motor", "bud", "stem", "bark", "pollen", "pod",
]


def _horn_bodies(rng: random.Random, chain: list[str], members: list[set[str]],
                  hops: int) -> list[int] | None:
    """Free-leg index j for each rule i in 2..hops+1, uniform over the j whose body pair
    {chain[j], chain[i-1]} lies in exactly one chain (P1). None if some rule has no valid j."""
    js = []
    for i in range(2, hops + 2):
        for j in rng.sample(range(i - 1), i - 1):
            if sum({chain[j], chain[i - 1]} <= m for m in members) == 1:
                js.append(j)
                break
        else:
            return None
    return js


def _horn_example(rng: random.Random, hops: int, n_chains: int, rho: int,
                   pool: list[str], surface: str) -> dict:
    fact_fmt, rule_fmt, query_fmt = _HORN_SURFACES[surface]
    span = hops + 1  # non-terminal positions 0..hops; the private terminal sits at hops+1
    n_cats = n_chains * span // rho
    names = rng.sample(pool, n_cats + n_chains)
    terminals = names[n_cats:]
    owners = [[] for _ in range(n_chains)]
    room = [span] * n_chains
    for c in names[:n_cats]:
        # give each category to the rho chains with the most room left (ties broken at random):
        # the greedy that always meets the degree sequence, so every chain ends up with exactly
        # `span` non-terminals and every category sits on exactly rho chains
        for e in sorted(range(n_chains), key=lambda e: (-room[e], rng.random()))[:rho]:
            owners[e].append(c)
            room[e] -= 1
    assert set(room) == {0}
    members = [set(o) | {t} for o, t in zip(owners, terminals)]

    chains, bodies = [], []
    for e in range(n_chains):
        for _ in range(200):
            chain = owners[e][:]
            rng.shuffle(chain)
            chain.append(terminals[e])
            js = _horn_bodies(rng, chain, members, hops)
            if js is not None:
                break
        else:
            raise AssertionError(f"horn: P1 rejection sampling did not converge (hops={hops}, "
                                 f"n_chains={n_chains}, rho={rho}) — lower rho")
        chains.append(chain)
        bodies.append(js)

    entities = rng.sample(_CHAIN_ENTITIES, n_chains)
    answer = rng.randint(0, 1)
    target = terminals[0] if answer else rng.choice(terminals[1:])

    facts = {(e, c): fact_fmt.format(e=e, c=c) for e, ch in zip(entities, chains) for c in ch[:2]}
    rules = {}
    for k, (ch, js) in enumerate(zip(chains, bodies)):
        for i in range(2, hops + 2):
            body = [ch[js[i - 2]], ch[i - 1]]
            rng.shuffle(body)
            rules[(k, i)] = rule_fmt.format(a=body[0], b=body[1], h=ch[i])
    premises = list(facts.values()) + list(rules.values())
    rng.shuffle(premises)
    # the two base facts are interchangeable — the first rule's body is shuffled, so nothing in
    # the input distinguishes c_0 from c_1 — so emit them in premise order rather than in chain
    # order, which would put an unrecoverable coin flip in the gold trace
    fired = sorted(premises.index(facts[(entities[0], c)]) for c in chains[0][:2])
    fired += [premises.index(rules[(0, i)]) for i in range(2, hops + 2)]
    trace = _encode_indices(fired)

    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(premises))
    query = query_fmt.format(e=entities[0], t=target)
    return {
        "input": f"{numbered}\n\n{query}",
        "reasoning": trace + [2, answer],
        "output": [answer],
    }


def _horn_data(hops: int, n_chains: int, rho: int, n_samples: int, seed: int,
               surface: str) -> list[dict]:
    rng = random.Random(seed)
    pool = _COMPACT_CATEGORIES if surface == "compact" else _chain_name_pool()
    needed = n_chains * (hops + 1) // rho + n_chains
    assert len(pool) >= needed, (
        f"horn surface={surface}: the category pool holds {len(pool)} words but hops={hops}, "
        f"n_chains={n_chains}, rho={rho} needs {needed} distinct categories per instance — "
        "lower hops/n_chains, raise rho, or extend the pool")
    return [_horn_example(rng, hops, n_chains, rho, pool, surface) for _ in range(n_samples)]


def _horn_explanation(hops: int, surface: str) -> str:
    if surface == "compact":
        return _horn_compact_explanation(hops)
    example = _encode_indices([2, 5])
    return (
        "You will receive a list of numbered premises followed by a query of the form "
        '"True or false: ...". Some premises are base facts placing an entity in a category '
        '(e.g. "Alex is a wumpus."); the rest are conjunctive rules of the form "Everything '
        'that is a A and a B is a C.". A rule fires for an entity only once BOTH of its body '
        "categories A and B have already been established for that entity, and then "
        "establishes C for it. Categories are shared: the same category is established by "
        "several different entities, so you cannot tell which entity a rule belongs to by "
        "looking at its categories one at a time — only the PAIR of body categories is "
        "specific to one entity, and every rule fires for exactly one of them. The queried "
        "entity has exactly two base facts; starting from those two categories, fire the "
        f"rules as they become applicable, {hops} rules in total, and follow the queried "
        "entity's derivation to its final category. Exactly one rule is applicable at each "
        "step: it is the rule whose body pairs the category you established last with some "
        "category established earlier, which may be arbitrarily far back — always check both "
        "body categories against everything established so far. The queried category may "
        "instead be the final category of another entity's derivation: the answer is whether "
        "it is derivable FOR THE QUERIED ENTITY — 1 if it is the queried entity's final "
        "category, 0 if it is only derivable for a different entity. Emit your reasoning as "
        "premise numbers in the order you use them, each written in binary (most significant "
        "bit first, no leading zeros) and separated by the delimiter value 2: first the "
        "numbers of the queried entity's two base-fact premises in increasing order, then the "
        f"number of each rule you fire in derivation order, for {hops} rules in total. "
        "Conclude your reasoning with one more delimiter 2 followed by the answer: 1 if the "
        "queried category is the entity's final category, 0 if it is not. Then output that "
        "same single value. For example, if the entity's base-fact premises are 2 and 5, your "
        f"reasoning begins {example} (premise 2 = binary {format(2, 'b')}, delimiter 2, "
        f"premise 5 = binary {format(5, 'b')}), continuing with the fired rules and ending "
        "with a delimiter 2 and the answer."
    )


def _horn_compact_explanation(hops: int) -> str:
    example = _encode_indices([2, 5])
    return (
        "You will receive numbered premises followed by a query \"? E : X\". A premise "
        '"E : A" is a base fact: entity E is in category A. A premise "A B > C" is a rule: '
        "anything that is in BOTH A and B is also in C, so the rule fires for an entity only "
        "once both body categories A and B are established for it, and then establishes C. "
        "Categories are shared between entities, so no single category says which entity a "
        "rule belongs to — only the PAIR of body categories is specific to one entity, and "
        "every rule fires for exactly one of them. The queried entity has exactly two base "
        f"facts; starting from those two categories, fire rules as they become applicable, "
        f"{hops} rules in total, down to the entity's final category. Exactly one rule is "
        "applicable at each step: the one pairing the category you established last with some "
        "category established earlier, which may be arbitrarily far back — always check both "
        "body categories against everything established so far. The queried category may "
        "instead be another entity's final category; the answer is whether it is derivable "
        "FOR THE QUERIED ENTITY. Emit your reasoning as premise numbers in use order, each in "
        "binary (most significant bit first, no leading zeros), separated by the delimiter "
        "value 2: first the queried entity's two base-fact premises in increasing order, then "
        f"each rule you fire in derivation order, {hops} rules in total. Conclude with one "
        "more delimiter 2 and the answer: 1 if the queried category is the entity's final "
        "category, 0 if it is not. Then output that same single value. For example, base-fact "
        f"premises 2 and 5 begin the reasoning {example} (premise 2 = binary "
        f"{format(2, 'b')}, delimiter 2, premise 5 = binary {format(5, 'b')})."
    )


def make_task(strategy: str, hops: int, n_samples: int, seed: int = 42,
              n_chains: int = 2, rho: int = 2, surface: str = "prose") -> SideTaskDefinition:
    if strategy == "answer_only":
        data = _answer_only_data(hops, n_samples, seed)
        explanation = _answer_only_explanation()
        alphabet = [0, 1]
    elif strategy == "proof_trace":
        data = _proof_trace_data(hops, n_samples, seed)
        explanation = _proof_trace_explanation(hops)
        alphabet = [0, 1, 2]  # binary digits 0/1 plus the delimiter 2
    elif strategy == "chain":
        data = _chain_data(hops, n_chains, n_samples, seed)
        explanation = _chain_explanation(hops)
        alphabet = [0, 1, 2]
    elif strategy == "horn":
        assert surface in _HORN_SURFACES, (
            f"unknown horn surface {surface!r}: expected one of {sorted(_HORN_SURFACES)}")
        assert rho >= 2, "horn needs rho >= 2: at rho = 1 the chains partition the categories"
        assert n_chains > rho, (
            f"horn pair privacy (P1) is infeasible at n_chains={n_chains} <= rho={rho}: every "
            "non-terminal would lie on every chain, so every body pair is contained in more "
            "than one chain. Use n_chains > rho (e.g. n_chains=4, rho=2)")
        assert n_chains * (hops + 1) % rho == 0, (
            f"horn needs n_chains * (hops + 1) divisible by rho so that every non-terminal "
            f"lies on exactly rho chains: got {n_chains} * {hops + 1} % {rho} != 0")
        data = _horn_data(hops, n_chains, rho, n_samples, seed, surface)
        explanation = _horn_explanation(hops, surface)
        alphabet = [0, 1, 2]
    else:
        raise ValueError(f"Unknown prontoqa strategy: {strategy}")
    train_data, test_data = partition_split(data, f"prontoqa_{strategy}")
    return SideTaskDefinition(
        name=f"prontoqa_{strategy}",
        train_data=train_data,
        test_data=test_data,
        explanation=explanation,
        alphabet=alphabet,
    )
