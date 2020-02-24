from data_loading.data_generator import generate


if __name__ == '__main__': 
    generate(n_experiments=10, filter_unknowns=True, without_top_pop=False, base_dir='./datasets')
