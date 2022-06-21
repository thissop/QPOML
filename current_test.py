from qpoml.main import collection


def test_load_basic(): 
    from qpoml.better_main import collection

    spectrum_context = './research and development/example_spectrum.csv'
    scalar_context = './research and development/example_scalar.csv'
    qpo = './research and development/example_qpo.csv'
    order_qpo = './research and development/example_qpo_order.csv'

    qpo_preprocess = {'frequency':[0.01,20], 'width':[0.001,4], 'amplitude':[0.001, 5]}

    
    # eurostep qpo approach with spectrum by-row that's rebinned to 
    collection_one = collection()
    collection_one.load(qpo_csv=qpo, context_csv=spectrum_context, context_type='spectrum', 
                    context_preprocess='median', qpo_preprocess=qpo_preprocess, qpo_approach='eurostep', 
                    spectrum_approach='by-row', rebin=2)

    # single qpo approach with spectrum by-column 
    collection_two = collection()
    collection_two.load(qpo_csv=qpo, context_csv=spectrum_context, context_type='spectrum', 
                        context_preprocess='median', qpo_preprocess=qpo_preprocess, qpo_approach='single', 
                        spectrum_approach='by-column')

    # single qpo approach with order ... scalar context 
    collection_three = collection()
    collection_three.load(qpo_csv=order_qpo, context_csv=scalar_context, context_type='scalar', 
                    context_preprocess={'gamma':[1,4], 'tin':[0.1,3]}, qpo_preprocess=qpo_preprocess, qpo_approach='single')

#test_load_basic()

def test_evaluation_single(): 
    
    collection_one = collection()
    collection_one.load()
    
    pass 