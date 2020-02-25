from data_loading.data_generator import generate, prepare


if __name__ == '__main__':
    prepare('./datasets', './data_loading/mindreader')
    generate(n_experiments=10, filter_unknowns=True, without_top_pop=False, base_dir='./datasets')
