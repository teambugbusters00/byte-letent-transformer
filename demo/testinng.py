import spacy
import networkx as nx
import matplotlib.pyplot as plt

# Load spaCy's pre-trained English model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Apple is looking at buying U.K. startup for $1 billion. Tim Cook is the CEO of Apple."

# Process text with spaCy
doc = nlp(text)

# Initialize a directed graph
G = nx.DiGraph()

# Extract entities and add them as nodes
entities = [(ent.text, ent.label_) for ent in doc.ents]
for ent, label in entities:
    G.add_node(ent, label=label)

# Extract simple relationships based on sentence dependency parsing
for sent in doc.sents:
    for token in sent:
        # Look for subject-verb-object triples as relations
        if token.dep_ in ("ROOT"):
            subject = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
            objects = [w for w in token.rights if w.dep_ in ("dobj", "pobj")]
            if subject and objects:
                subj_text = subject[0].text
                obj_text = objects[0].text
                # Add edge with relation label as the verb
                if G.has_node(subj_text) and G.has_node(obj_text):
                    G.add_edge(subj_text, obj_text, relation=token.lemma_)

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10)
edge_labels = nx.get_edge_attributes(G, 'relation')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()
