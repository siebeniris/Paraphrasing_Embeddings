import os

from preprocessing.extract_data import data_frequency, data_threshold, data_filter, build_vocabs


def main(num):
    ## create a folder with to-be-used data in it
    data_dir = "data/"
    output_dir = "data_/"


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_frequency(data_dir + "word2vec_train_untyped_unique_filtered.txt",
                   data_dir+ "ent_path_freq.txt", data_dir + "ent_freq.txt", data_dir + "path_freq.txt")
    data_threshold(num, data_dir +"ent_freq.txt", output_dir+"ents.txt")
    data_filter(output_dir+"ents.txt", data_dir+"ent_paths_unique_filtered.txt",
               output_dir+"entities_path_filtered.txt", True)

    build_vocabs(output_dir+"entities_path_filtered.txt",
                      output_dir+"entpair2id.json",
                        output_dir+"ent2id.json",
                      output_dir+"path2id.json")

if __name__ == '__main__':
    num =2
    main(num)