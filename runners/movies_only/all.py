from runners.movies_only import item_knn, svd, bpr, user_knn, trans_e, trans_e_kg

def run():
    bpr.run('../results/movies_only', 'bpr')
    svd.run('../results/movies_only', 'svd')
    trans_e.run('../results/movies_only', 'trans_e')
    trans_e_kg.run('../results/movies_only', 'trans_e_kg')
    # mf.run('../results/movies_only', 'mf')
    # joint_mf.run('../results/movies_only', 'joint_mf')
    user_knn.run('../results/movies_only', 'user_knn')
    item_knn.run('../results/movies_only', 'item_knn')
    # pagerank.run('../results/movies_only', 'cpr')
    # joint_pr.run('../results/movies_only', 'jpr')
    # kg_pr.run('../results/movies_only', 'kgpr')



