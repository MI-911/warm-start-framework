from runners.mixing_ratings import item_knn, svd, bpr, user_knn, trans_e, joint_mf

def run():
    # bpr.run('../results/mixing_ratings', 'bpr')
    # svd.run('../results/mixing_ratings', 'svd')
    # trans_e.run('../results/mixing_ratings', 'trans_e')
    # mf.run('../results/mixing_ratings', 'mf')

    user_knn.run('../results/mixing_ratings', 'user_knn')
    item_knn.run('../results/mixing_ratings', 'item_knn')
    joint_mf.run('../results/mixing_ratings', 'joint_mf')
    # pagerank.run('../results/mixing_ratings', 'cpr')
    # joint_pr.run('../results/mixing_ratings', 'jpr')
    # kg_pr.run('../results/mixing_ratings', 'kgpr')



