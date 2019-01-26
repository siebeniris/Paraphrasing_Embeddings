import re
import csv

def untypify(typed_relation):
    return re.sub(
        r'~[0-9]+~-([^~]+?)-~[0-9]+~',
        r'\1', typed_relation
    )


def read_test_data(fname:str, outputfile:str ='') -> list:
    path_list = []
    with open(fname) as f:
        cr = csv.reader(f)
        next(cr)  # headers
        for row in cr:
            prem, hypo, gold_label = row
            prem = untypify(prem)
            hypo = untypify(hypo)
            path_list.append(prem)
            path_list.append(hypo)

    if outputfile !='':
        with open(outputfile, "w+")as output:
            paths = list(set(path_list))
            for path in paths:
                output.write(path+'\n')

    return path_list


def filter_word2vec_train_untyped(inputfile:str, outputfile:str) -> None:
    paths = read_test_data("test_simplified.csv", '')
    with open(inputfile) as f:
        with open(outputfile,"w")as output:
            for line in f.readlines():
                ent1, path, ent2 = line.split()
                if path in paths:
                    output.write(line)


if __name__ == '__main__':
    filter_word2vec_train_untyped('word2vec_train_untyped_unique.txt', 'word2vec_train_untyped_unique_filtered.txt')
