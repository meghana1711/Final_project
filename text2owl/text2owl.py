import sys
import os
import glob
import re
from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal


# Put the folder that contains your .txt files here
input_folder = r"C:/Users/20236193/Final_project/data/sample"   


def sanitize_id(label: str) -> str:
    """
    Turn an arbitrary label string into a safe local name for a URI:
    - keep only [A-Za-z0-9_], convert others to '_'
    - collapse multiple '_' and strip from ends
    - ensure it starts with a letter or '_'
    """
    s = label.strip()
    # replace any non-alphanumeric/underscore with '_'
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    # collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    # trim underscores
    s = s.strip("_")

    if not s:
        s = "Class"

    # local name must not start with a digit
    if not re.match(r"[A-Za-z_]", s[0]):
        s = "C_" + s

    return s


def parse_taxonomy(lines, indent_size=2):
    """
    Parse an indented text taxonomy into (classes, subclass_relations).
    Indentation level -> subclassOf.
    """
    stack = []  # list of (level, class_label)
    relations = []  # (child_label, parent_label)
    classes = set()

    for raw in lines:
        line = raw.rstrip("\n")
        if not line or line.strip().startswith("#"):
            continue

        # count leading spaces
        leading = len(line) - len(line.lstrip(" "))
        level = leading // indent_size
        label = line.strip()

        classes.add(label)

        # pop until parent level
        while stack and stack[-1][0] >= level:
            stack.pop()

        # if there is a parent, create subclass relation
        if stack:
            parent_label = stack[-1][1]
            relations.append((label, parent_label))

        stack.append((level, label))

    return classes, relations


def text2owl(text, base_iri="https://your-domain.com/ontology/slurm#"):
    classes, relations = parse_taxonomy(text.splitlines())

    g = Graph()
    EX = Namespace(base_iri)

    g.bind("ex", EX)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)

    # map original labels to sanitized IDs
    label_to_id = {lbl: sanitize_id(lbl) for lbl in classes}

    # declare classes
    for lbl, local_id in label_to_id.items():
        uri = EX[local_id]
        g.add((uri, RDF.type, OWL.Class))
        # keep original text as rdfs:label
        g.add((uri, RDFS.label, Literal(lbl)))

    # add subclass axioms
    for child_lbl, parent_lbl in relations:
        child_id = label_to_id[child_lbl]
        parent_id = label_to_id[parent_lbl]
        child_uri = EX[child_id]
        parent_uri = EX[parent_id]
        g.add((child_uri, RDFS.subClassOf, parent_uri))

    return g


if __name__ == "__main__":
    if not os.path.isdir(input_folder):
        print(f"Error: {input_folder} is not a folder or does not exist")
        sys.exit(1)

    # Read all .txt files in the folder
    txt_paths = sorted(glob.glob(os.path.join(input_folder, "*.txt")))
    if not txt_paths:
        print(f"No .txt files found in {input_folder}")
        sys.exit(1)

    all_texts = []
    for path in txt_paths:
        with open(path, "r", encoding="utf-8") as f:
            all_texts.append(f.read())

    combined_text = "\n".join(all_texts)

    graph = text2owl(combined_text)

    output_path = "C:/Users/20236193/Final_project/text2owl/owl1.ttl"
    graph.serialize(destination=output_path, format="turtle")
    print(f"Ontology written to {output_path}")