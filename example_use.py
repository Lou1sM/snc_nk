import numpy as np
from snc_nk import PredictorAligner

example_results_train = np.load('example_results_train.npz')
example_results_test = np.load('example_results_test.npz')
latents_train = example_results_train['latents']
gts_train = example_results_train['gts']
latents_test = example_results_test['latents']
gts_test = example_results_test['gts']
pa = PredictorAligner()
pa.align_and_predict_unsupervised_vae(latents_train, gts_train, latents_test, gts_test)
