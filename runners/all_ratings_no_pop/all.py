from runners.all_ratings_no_pop import item_knn, svd, bpr, user_knn, trans_e, trans_e_kg, pagerank

def run():
    bpr.run('../results/all_ratings_no_pop', 'bpr')
    svd.run('../results/all_ratings_no_pop', 'svd')
    trans_e.run('../results/all_ratings_no_pop', 'trans_e')
    trans_e_kg.run('../results/all_ratings_no_pop', 'trans_e_kg')
    # mf.run('../results/all_ratings_no_pop', 'mf')
    # joint_mf.run('../results/all_ratings_no_pop', 'joint_mf')
    user_knn.run('../results/all_ratings_no_pop', 'user_knn')
    item_knn.run('../results/all_ratings_no_pop', 'item_knn')
    pagerank.run('../results/all_ratings_no_pop', 'cpr')
    # joint_pr.run('../results/all_ratings_no_pop', 'jpr')
    # kg_pr.run('../results/all_ratings_no_pop', 'kgpr')



