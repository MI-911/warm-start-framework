from runners.all_ratings import item_knn, svd, bpr, user_knn, trans_e, trans_e_kg, pagerank


def run():
    bpr.run('../results/all_ratings', 'bpr')
    svd.run('../results/all_ratings', 'svd')
    trans_e.run('../results/all_ratings', 'trans_e')
    trans_e_kg.run('../results/all_ratings', 'trans_e_kg')
    # mf.run('../results/all_ratings', 'mf')
    # joint_mf.run('../results/all_ratings', 'joint_mf')
    user_knn.run('../results/all_ratings', 'user_knn')
    item_knn.run('../results/all_ratings', 'item_knn')
    pagerank.run('../results/all_ratings', 'cpr')
    # joint_pr.run('../results/all_ratings', 'jpr')
    # kg_pr.run('../results/all_ratings', 'kgpr')



