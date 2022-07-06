def working_testing():
    from qpoml import collection

    qpo_csv = "./qpoml/tests/references/fake_generated_qpos.csv"
    scalar_collection_csv = "./qpoml/tests/references/fake_generated_scalar_context.csv"

    collection_two = collection()
    context_dict = {"gamma": [1.0, 3.5], "T_in": [0.1, 2.5], "hardness": [0, 1]}
    collection_two.load(
        qpo_csv=qpo_csv,
        context_csv=scalar_collection_csv,
        context_type="scalar",
        context_preprocess=context_dict,
        qpo_preprocess={"frequency": [1, 16], "width": [0.1, 1.6], "amplitude": [1, 6]},
        qpo_approach="eurostep",
    )

    print(collection_two.num_qpos)
    # print(collection_two.qpo_tensor)
    # regr = RandomForestRegressor()

    # collection_two.evaluate(model=regr, model_name='RandomForestRegressor', evaluation_approach='k-fold', folds=5, repetitions=4)


working_testing()
