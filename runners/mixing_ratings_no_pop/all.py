from runners.mixing_ratings_no_pop import item_knn, svd, bpr, user_knn, trans_e

def run():
    bpr.run('../results/mixing_ratings_no_pop', 'bpr')
    svd.run('../results/mixing_ratings_no_pop', 'svd')
    trans_e.run('../results/mixing_ratings_no_pop', 'trans_e')
    # mf.run('../results/mixing_ratings_no_pop', 'mf')
    # joint_mf.run('../results/mixing_ratings_no_pop', 'joint_mf')
    user_knn.run('../results/mixing_ratings_no_pop', 'user_knn')
    item_knn.run('../results/mixing_ratings_no_pop', 'item_knn')
    # pagerank.run('../results/mixing_ratings_no_pop', 'cpr')
    # joint_pr.run('../results/mixing_ratings_no_pop', 'jpr')
    # kg_pr.run('../results/mixing_ratings_no_pop', 'kgpr')



