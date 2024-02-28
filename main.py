import os
from rdflib import Graph
from deep_similarity import DeepSimilarity
import numpy as np
import random
import validators
from rdflib import Graph, URIRef, Namespace, Literal
from rdflib.namespace import OWL
from rdflib import Graph, URIRef
from tqdm import tqdm
import time
import multiprocessing
from embeddings import Embedding
import itertools
import argparse
from compute_files import ComputeFile
import pandas as pd
from model import LLM


start_time = time.time()

output_alignments = {}

question_pattern = """A:"{chain1}". B:"{chain2}". Do you think expressions A and B are similar?. Answer exactly yes or no."""


def append_rows_to_csv(new_rows, measure_file):
    try:
        df = pd.read_csv(measure_file)
    except FileNotFoundError:
        df = pd.DataFrame(
            columns=['Dataset', 'Precision', 'Recall', 'F1-score', 'CoSimThreshold', 'LLM_name', 'CandidatesPairs', 'SelectedCandidates', 'RunningTime'])

    new_data = pd.DataFrame(
        new_rows, columns=['Dataset', 'Precision', 'Recall', 'F1-score', 'CoSimThreshold', 'LLM_name', 'CandidatesPairs', 'SelectedCandidates', 'RunningTime'])
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(measure_file, index=False)


def calculate_alignment_metrics(output_file, truth_file, suffix, co_sim, llm_name, count_pairs, selected_count, running_time):
    measure_file = output_file.replace(
        'tmp_valid_same_as.ttl', 'measure_file.csv')
    output_graph = Graph()
    output_graph.parse(output_file, format="turtle")

    truth_graph = Graph()
    truth_graph.parse(truth_file, format="turtle")

    found_alignments = set(output_graph.subjects())
    true_alignments = set(truth_graph.subjects())
    print('Count of true alignments : ', len(true_alignments))
    intersection = len(found_alignments.intersection(true_alignments))
    precision = round(intersection /
                      len(found_alignments) if len(found_alignments) > 0 else 0.0, 2)
    recall = round(intersection /
                   len(true_alignments) if len(true_alignments) > 0 else 0.0, 2)
    f_measure = round(2 * (precision * recall) / (precision +
                                                  recall) if (precision + recall) > 0 else 0.0, 2)

    append_rows_to_csv([(suffix, precision, recall, f_measure, co_sim, llm_name,
                         count_pairs, selected_count, round(running_time, 2))], measure_file)
    return {
        "precision": precision,
        "recall": recall,
        "f_measure": f_measure
    }

# End of Metrics handling


def cosine_sim(v1=[], v2=[], co_sim=0.0):
    output = 0.0
    dot = np.dot(v1, v2)
    cosine = dot / (np.linalg.norm(v1) * np.linalg.norm(v2))
    output = cosine
    if output >= co_sim:
        return True, output
    return False, output


def build_literal_chain(entity=[]):
    output = []
    for resource, object_value in entity:
        if not validators.url(object_value):
            output.append(object_value)
    if len(output) >= 2:
        return str('. '.join(output))
    return None


def sim(entity1=[], entity2=[], deepSim=None):
    # use LLM to compare entities literals
    """
        1. build chain of literals for each entity
        2. generate the complete sentence for the LLM models
        3. Call the model with the question
    """
    chain1 = build_literal_chain(entity=entity1)
    chain2 = build_literal_chain(entity=entity2)

    if chain1 != None and chain2 != None:
        query = question_pattern.format(chain1=chain1, chain2=chain2)

        print('\n \n')
        print('User Question :#> ', query)
        response = deepSim.run(query=query)
        print("Response # ", response)
        if 'yes' in response.lower():
            print('#>>', response)
            return True
        # print('\n \n')
    return False


def create_and_save_rdf_from_dict(input_dict, output_file):
    graph = Graph()
    # owl = Namespace("http://www.w3.org/2002/07/owl#")
    for source, target in input_dict.items():
        source_uri = URIRef(source)
        target_uri = URIRef(target)
        graph.add((source_uri, OWL.sameAs, target_uri))
    graph.serialize(destination=output_file, format="turtle")


def get_rdf_subjects(rdf_graph):
    output = list(rdf_graph.subjects())
    return output


def get_rdf_triples(rdf_graph):
    subjects = {}
    objects = {}
    for s, p, o in tqdm(rdf_graph):
        s = str(s)
        p = str(p)
        o = str(o)
        if not s in subjects:
            subjects[s] = []
        if not o in objects:
            objects[o] = 0
        objects[o] += 1
        subjects[s].append((p, o))
    return subjects, objects

# End of embedding functions


def parallel_running(sub1, sub2, vector1, vector2, subs1, subs2, co_sim, deepSim):
    v, cos = cosine_sim(v1=vector1, v2=vector2, co_sim=co_sim)
    if v:
        if sim(entity1=subs1[sub1], entity2=subs2[sub2], deepSim=deepSim):
            return sub1, sub2, 1
    return None, None, 0


def process_rdf_files(source, target, output_file, truth_file, suffix, dimension, embedding, co_sim, llm_name):

    graph1 = Graph()
    graph1.parse(source)
    print('Source file loaded ..100%')

    graph2 = Graph()
    graph2.parse(target)
    print('Target file loaded ..100%')

    graph = graph1 + graph2

    embeddings = Embedding(graph=graph, file=source,
                           dimension=dimension, model_name=embedding).run()
    # valid_alignments = Embedding(
    #     file=truth_file, dimension=dimension).build_triples(with_predicate=False)

    graph3 = Graph()
    graph3.parse(truth_file)
    print('Truth file loaded ..100%')

    print('Graph1 Subjects\'s and Objects\' list are building ..0%')
    subjects1, objects1 = get_rdf_triples(graph1)
    print('Graph2 Subjects\'s and Objects\' list are building ..0%')
    subjects2, objects2 = get_rdf_triples(graph2)
    print('Building ended')

    # print('Candidates reducing ')
    print('Instances of source : ', len(list(subjects1.keys())))
    print('Instances of target : ', len(list(subjects2.keys())))

    pairs = list(itertools.product(
        list(subjects1.keys()), list(subjects2.keys())))
    print('In all : ', len(pairs))
    # exit()

    # load the llm
    model = LLM(model_name=llm_name).load()
    deepSim = DeepSimilarity(model_name=llm_name, model=model)
    # loading llm ended

    print('LLM ', llm_name, ' loaded 100% ####>>>>')
    count = 0
    for sub1, sub2 in tqdm(pairs):
        if sub1 in embeddings and sub2 in embeddings:
            result = parallel_running(
                sub1, sub2, embeddings[sub1], embeddings[sub2], subjects1, subjects2, co_sim, deepSim)
            _, _, status = result
            if status == 1:
                output_alignments[sub1] = sub2
                count = count + 1
                print('\n %%%%%%%%%%%%%%%%%%% \n')
    print(f' \n Total alignments {len(list(output_alignments.keys()))}')
    create_and_save_rdf_from_dict(output_alignments, output_file)
    end_time = time.time()
    execution_time = end_time - start_time
    metrics = calculate_alignment_metrics(
        output_file, truth_file, suffix, co_sim, llm_name, len(pairs), count, execution_time)
    print("Precision : ", metrics["precision"])
    print("Recall : ", metrics["recall"])
    print("F-measure : ", metrics["f_measure"])
    print(f" \n Running time : {execution_time} seconds")


if __name__ == "__main__":
    def detect_file(path='', type=''):
        files = ComputeFile(input_path=path).build_list_files()
        for v in files:
            if type in v:
                return v
        return None

    def arg_manager():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_path", type=str, default="./inputs/")
        parser.add_argument("--output_path", type=str, default="./outputs/")
        parser.add_argument("--suffix", type=str, default="doremus")
        parser.add_argument("--dimension", type=int, default=0.0)
        parser.add_argument("--embedding", type=str, default="r2v")
        parser.add_argument("--llm_name", type=str, default="GPT-3.5-turbo")
        parser.add_argument("--co_sim", type=float, default=0.0)
        return parser.parse_args()
    args = arg_manager()
    source = detect_file(path=args.input_path+args.suffix, type='source')
    target = detect_file(path=args.input_path+args.suffix, type='target')
    truth_file = detect_file(path=args.input_path +
                             args.suffix, type='valid_same_as')
    output_path = args.output_path + args.suffix
    output_file = output_path + '/tmp_valid_same_as.ttl'
    print(args.output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print(source, target, output_file, truth_file)
    process_rdf_files(source, target, output_file,
                      truth_file, args.suffix, args.dimension, args.embedding, args.co_sim, args.llm_name)
