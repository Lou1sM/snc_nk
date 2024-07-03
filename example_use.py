import numpy as np
from snc_nk import PredictorAligner

example_results_train = np.load('example_results_train.npz')
example_results_test = np.load('example_results_test.npz')
latents_train = example_results_train['latents']
gts_train = example_results_train['gts']
latents_test = example_results_test['latents']
gts_test = example_results_test['gts']
pa = PredictorAligner(dset_name='dsprites',
                      vae_name='beta',
                      expname='example',
                      max_epochs=10,
                      verbose=True,
                      )
pa.predict_unsupervised_vae(latents_train, gts_train, latents_test, gts_test)

results_fpath = 'example_results.json'
pa.save_and_print_results(results_fpath)
