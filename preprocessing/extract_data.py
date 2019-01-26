import json


### given "entities_path.txt", to calculate the frequencies of entities OR dep_path,
def data_frequency(entity_path_file:str, outputfile:str, ent_output:str, path_output:str) -> None:
    from collections import defaultdict
    import operator

    freq_dict = defaultdict(int)
    with open(entity_path_file) as f:
        for line in f.readlines():
            entities, path = line.split()

            if entities in freq_dict:
                freq_dict[entities] += 1
            else:
                freq_dict[entities] = 1
            if path in freq_dict:
                freq_dict[path] += 1
            else:
                freq_dict[path] = 1

    with open(outputfile, "w+") as f:
        with open(ent_output, "w+")as f_ent:
            with open(path_output, "w+")as fpath:
                for key, freq in dict(sorted(freq_dict.items(), key= operator.itemgetter(1), reverse=True)).items():
                    f.write(key+","+str(freq)+'\n')
                    if key.startswith('m.'):
                        f_ent.write(key+","+str(freq)+'\n')
                    else:
                        fpath.write(key+","+str(freq)+'\n')


def data_threshold(t:int, inputfile:str, outputfile:str)-> None:
    ### only extract data whose frequency are bigger than t.
    with open(inputfile)as f:
        with open(outputfile, "w+") as output:
            for line in f.readlines():
                line= line.replace("\n", "")
                entry, freq = line.rsplit(",",1)
                freq= int(freq)
                if freq > t:
                    output.write(entry+"\n")


def data_filter(filtered:str, entity_path_file:str, outputfile:str, entry:bool)-> None:
    filteredfile = open(filtered)
    filteredlist= [line.replace("\n","") for line in filteredfile.readlines()]
    with open(entity_path_file) as f:
        with open(outputfile, "w+") as output:
            for line in f.readlines():
                ent_pair, path = line.split()
                if entry:
                    if ent_pair in filteredlist:
                        output.write(line)
                else:
                    if path in filteredlist:
                        print(path)
                        output.write(line)

def build_vocabs(inputfile: str, entpair: str, entvocab:str, pathvocab:str) -> None:
    entity_pairs = []
    paths = []
    ents = []
    with open(inputfile) as f:

            for line in f.readlines():
                line = line.replace("\n", "")
                ent_pair , path = line.split()
                ent0, ent1 = ent_pair.split("#")
                entity_pairs.append(ent_pair)
                ents.append(ent0)
                ents.append(ent1)
                paths.append(path)

    entity_pairs = list(set(entity_pairs))
    paths = list(set(paths))
    ents = list(set(ents))
    print("length of entity pairs", len(entity_pairs))
    print("length of paths", len(paths))
    print("length of entities", len(ents))

    entpair2id = {w: idx for idx, w in enumerate(entity_pairs)}
    path2id = {w: idx for idx, w in enumerate(paths)}
    ent2id = {w: idx for idx, w in enumerate(ents)}
    json.dump(entpair2id, fp=open(entpair, "w+"))
    json.dump(ent2id, fp=open(entvocab, "w+"))
    json.dump(path2id, fp=open(pathvocab, "w+"))

