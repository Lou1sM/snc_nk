import numpy as np
import json
import os
from os.path import join
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch
import torch.nn.functional as F
from dl_utils.misc import check_dir
from dl_utils.torch_misc import CifarLikeDataset
from dl_utils.tensor_funcs import numpyify
from dl_utils.label_funcs import get_num_labels, get_trans_dict, get_trans_dict_from_cost_mat
from utils import normalize, load_trans_dict, np_ent, cond_ent_for_alignment, combo_acc, round_maybe_list


class PredictorAligner():
    def __init__(self, dset_name, vae_name, expname, max_epochs, quick_run=False, verbose=False):
        self.quick_run = quick_run
        self.expname = expname
        self.dset_name = dset_name
        self.vae_name = vae_name
        self.max_epochs = max_epochs
        self.zs_combo = (2,4)
        self.results = {'train': {}, 'test': {}}
        self.verbose = verbose
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_nt = not expname.startswith('zsc') and vae_name != 'hfs'
        check_dir('mi_mats')
        check_dir('trans_dicts')

    def set_cost_mat(self, latents_, gts):
        if self.quick_run:
            self.cost_mat = np.random.rand(self.nz,self.n_factors)
        latents = normalize(latents_)
        assert latents.ndim == 2
        assert gts.ndim == 2
        self.nz = latents.shape[1]
        cost_mat_ = np.empty((self.nz,self.n_factors)) # tall thin matrix normally
        for i in range(self.nz):
            if self.verbose:
                print(i)
            factor_latents = latents[:,i]
            for j in range(self.n_factors):
                if self.verbose:
                    print('\t' + str(j))
                factor_gts = gts[:,j]
                cost_mat_[i,j] = cond_ent_for_alignment(factor_latents,factor_gts)

        raw_ents = np.array([np_ent(self.gts[:,j]) for j in range(self.n_factors)])
        self.cost_mat = np.transpose(cost_mat_ - raw_ents) # transpose added because bug
        if self.verbose:
            print(self.cost_mat)
        np.save(f'mi_mats/{self.dset_name}_mi_mat_{self.expname}.npy',self.cost_mat)
        if not self.cost_mat.max() <= 0.03: # -MI should be negative but error in computing entropy
            breakpoint()

    def set_maybe_loaded_trans_dict(self, is_load=False, is_save=False):
        possible_fpath = join('trans_dicts',f'{self.dset_name}_{self.vae_name}_trans_dict_{self.expname}.npy')
        if is_load and os.path.isfile(possible_fpath):
            self.trans_dict = load_trans_dict(possible_fpath)
        else:
            self.set_cost_mat(self.latents,self.gts)
            self.trans_dict = get_trans_dict_from_cost_mat(self.cost_mat)
        if is_save:
            np.save(possible_fpath,np.array([[k,v] for k,v in self.trans_dict.items()]))

    def set_data(self, latents, gts, latents_test, gts_test):
        assert latents.ndim == 2
        assert gts.ndim == 2
        assert latents_test.ndim == 2
        assert gts_test.ndim == 2
        assert latents.shape[1] == latents_test.shape[1]
        assert gts.shape[1] == gts_test.shape[1]
        assert latents.shape[0] == gts.shape[0]
        assert latents_test.shape[0] == gts_test.shape[0]
        self.latents, self.gts = latents, gts
        self.latents_test, self.gts_test = latents_test, gts_test
        self.n_factors = self.gts.shape[1]
        self.nz = self.latents.shape[1]

    def save_and_print_results(self, save_fpath):
        if self.quick_run:
            save_fpath += '.test'
        if os.path.isfile(save_fpath):
            print(f'WARNING: accs file already exists at {save_fpath}')
            save_fpath += '.safe'
            print(f'saving to {save_fpath} instead')
        self.results = {k:{k2:round_maybe_list(v2) for k2,v2 in v.items()}
                            for k,v in self.results.items()}
        for k,v in self.results.items():
            print('\n'+k.upper()+':')
            for k2,v2 in v.items():
                print(f'{k2}: {v2}')

        with open(save_fpath,'w') as f:
            json.dump(self.results,f)

    def train_classif(self, x, y, x_test, y_test, is_mlp=True):
        ngts = max(y.max(), y_test.max()) + 1
        if x.ndim == 1: x=np.expand_dims(x,1)
        if is_mlp:
            fc = nn.Sequential(nn.Linear(x.shape[1],256,device=self.device),nn.ReLU(),nn.Linear(256,ngts,device=self.device))
        else:
            fc = nn.Linear(x.shape[1],ngts,device=self.device)
        opt = Adam(fc.parameters(),lr=1e-2,weight_decay=0e-2)
        scheduler = lr_scheduler.ExponentialLR(opt,gamma=0.65)

        assert not ((x_test == 'none').all() ^ (y_test=='none').all())
        if (x_test == 'none').all():
            dset = CifarLikeDataset(x,y)
            len_trainset = int(len(x)*.8)
            lengths = [len_trainset,len(x)-len_trainset]
            trainset, testset = random_split(dset,lengths)
            test_gt = y[testset.indices]
        else:
            if x_test.ndim == 1: x_test=np.expand_dims(x_test,1)
            trainset = CifarLikeDataset(x,y)
            testset = CifarLikeDataset(x_test,y_test)
            test_gt = y_test
        train_loader = DataLoader(trainset,shuffle=True,batch_size=4096)
        test_loader = DataLoader(testset,shuffle=False,batch_size=4096)
        train_losses = []
        tol = 0
        best_acc = 0
        best_corrects = np.zeros(len(y_test)).astype(bool)
        for i in range(self.max_epochs):
            if i > 0 and (i&(i-1) == 0): # power of 2
                scheduler.step()
            train_preds_list = []
            train_gts_list = []
            for batch_idx,(xb,yb) in enumerate(train_loader):
                preds = fc(xb.to(self.device))
                loss = F.cross_entropy(preds,yb.to(self.device))
                loss.backward(); opt.step()
                for p in fc.parameters():
                    p.grad = None
                train_losses.append(loss.item())
                train_preds_list.append(preds.argmax(axis=1))
                train_gts_list.append(yb)

            train_preds_array = numpyify(torch.cat(train_preds_list))
            train_gt = numpyify(torch.cat(train_gts_list))
            train_corrects = (train_preds_array==train_gt)
            train_acc = train_corrects.mean()
            preds_list = []
            with torch.no_grad():
                for xb,yb in test_loader:
                    preds = fc(xb.to(self.device))
                    preds_list.append(preds.argmax(axis=1))
                preds_array = numpyify(torch.cat(preds_list))
                corrects = (preds_array==test_gt)
            acc = corrects.mean()
            if self.verbose:
                print(f'Epoch: {i}\ttrain acc: {train_acc:.3f}\ttest acc:{acc:.3f}')
            if train_acc > .99:
                if i==0:
                    best_corrects=corrects
                break
            if train_acc>best_acc:
                tol = 0
                best_acc = train_acc
                best_corrects = corrects
            else:
                tol += 1
                if tol == 4 and self.verbose:
                    print('breaking at', i)
                    break
            if self.quick_run: break
        return train_corrects, best_corrects

    def simple_accs(self, x, y, xt, yt):
        num_classes = get_num_labels(y)
        bin_dividers = np.sort(x)[np.arange(0,len(x),len(x)/num_classes).astype(int)]
        bin_dividers[0] = min(bin_dividers[0],min(xt))
        bin_vals = sum([x<bd for bd in bin_dividers]) # sum is over K np arrays where K is num classes, produces labels for the dset
        bin_vals_test = sum([xt<bd for bd in bin_dividers])
        trans_dict = get_trans_dict(bin_vals,y,subsample_size=30000)
        train_corrects = np.array([trans_dict[z] for z in bin_vals]) == y
        test_corrects = np.array([trans_dict[z] for z in bin_vals_test]) == yt
        return train_corrects, test_corrects
        train_correctsa = (bin_vals==y)
        train_correctsb = ((num_classes-1-bin_vals)==y)
        use_reverse_encoding = train_correctsb.mean() > train_correctsa.mean()
        if use_reverse_encoding:
            test_corrects = ((num_classes-1-bin_vals_test)==yt)
            return train_correctsb, test_corrects
        else:
            test_corrects = (bin_vals_test==yt)
            return train_correctsa, test_corrects

    def accs_from_alignment(self, is_single_neurons, is_mlp):
        if is_single_neurons and not hasattr(self,'trans_dict'):
            self.set_maybe_loaded_trans_dict()
        all_train_corrects = []
        all_test_corrects = []
        for factor in range(self.gts.shape[1]):
            factor_gts = self.gts[:,factor]
            factor_gts_test = self.gts_test[:,factor]
            if is_single_neurons:
                corresponding_latent = self.trans_dict[factor]
                if self.verbose:
                    print(f'predicting gt {factor} with neuron {corresponding_latent}')
                factor_latents = self.latents[:,corresponding_latent]
                factor_latents_test = self.latents_test[:,corresponding_latent]
                train_corrects,test_corrects = self.simple_accs(factor_latents,factor_gts,factor_latents_test,factor_gts_test)
            else:
                train_corrects,test_corrects = self.train_classif(self.latents,factor_gts,self.latents_test,factor_gts_test,is_mlp=is_mlp)
            all_train_corrects.append(train_corrects)
            all_test_corrects.append(test_corrects)
        return all_train_corrects, all_test_corrects

    def zs_combo_accs_from_corrects(self, all_cs):
        zs_acc = combo_acc(all_cs[self.zs_combo[0]],all_cs[self.zs_combo[1]])
        return [c.mean() for c in all_cs] + [zs_acc]

    def standarize(x):
        x = x-x.mean(axis=0)
        x /= (x.std(axis=0)+1e-8)
        return x

    def progressive_knockout(self, start_at):
        self.results['train']['full_knockouts'] = {k:{} for k in range(self.n_factors)}
        self.results['test']['full_knockouts'] = {k:{} for k in range(self.n_factors)}
        if not hasattr(self,'trans_dict'):
            self.set_maybe_loaded_trans_dict()
        for m in range(self.n_factors):
            y = self.gts[:,m]
            y_test = self.gts_test[:,m]
            X = np.copy(self.latents)
            X_test = np.copy(self.latents_test)
            cost_mat = np.transpose(self.cost_mat)
            tmp_trans_dict = self.trans_dict
            for num_excluded in range(self.nz):
                if num_excluded >= start_at:
                    print(f'predicting factor {m} with {num_excluded} neurons knocked out')
                    tr_corrects, ts_corrects = self.train_classif(X,y,X_test,y_test,is_mlp=True)
                    self.results['train']['full_knockouts'][m][num_excluded] = tr_corrects.mean()
                    self.results['test']['full_knockouts'][m][num_excluded] = tr_corrects.mean()
                tmp_trans_dict = get_trans_dict_from_cost_mat(cost_mat)
                latent_to_exclude = tmp_trans_dict[m]
                X = np.delete(X,latent_to_exclude,1)
                X_test = np.delete(X_test,latent_to_exclude,1)
                cost_mat = np.delete(cost_mat,latent_to_exclude,1)

    def exclusive_mlp(self, test_factor, exclude_factor, num_to_exclude):
        first_latent_to_exclude = self.trans_dict[exclude_factor]
        X = np.delete(self.latents,first_latent_to_exclude,1)
        X_test = np.delete(self.latents_test,first_latent_to_exclude,1)
        if num_to_exclude == 2:
            cost_mat2 = np.delete(self.cost_mat,first_latent_to_exclude,1)
            trans_dict2 = get_trans_dict_from_cost_mat(cost_mat2)
            second_latent_to_exclude = trans_dict2[exclude_factor]
            X = np.delete(X,second_latent_to_exclude,1)
            X_test = np.delete(X_test,second_latent_to_exclude,1)

        y = self.gts[:,test_factor]
        y_test = self.gts_test[:,test_factor]
        return self.train_classif(X,y,X_test,y_test,is_mlp=True)

    def set_cross_NK_results(self, num_to_exclude): # excl neuron for other zs feature
        if not hasattr(self,'trans_dict'):
            self.set_maybe_loaded_trans_dict()
        train_cs1, test_cs1 = self.exclusive_mlp(self.zs_combo[0],self.zs_combo[1],num_to_exclude)
        train_cs2, test_cs2 = self.exclusive_mlp(self.zs_combo[1],self.zs_combo[0],num_to_exclude)
        zs_train_acc = combo_acc(train_cs1, train_cs2)
        zs_test_acc = combo_acc(test_cs1, test_cs2)
        self.results['train'][f'e{num_to_exclude}f1'] = train_cs1.mean()
        self.results['train'][f'e{num_to_exclude}f2'] = train_cs2.mean()
        self.results['train'][f'e{num_to_exclude}zs'] = zs_train_acc
        self.results['test'][f'e{num_to_exclude}f1'] = test_cs1.mean()
        self.results['test'][f'e{num_to_exclude}f2'] = test_cs2.mean()
        self.results['test'][f'e{num_to_exclude}zs'] = zs_test_acc

    def set_NK_results(self, num_to_exclude):
        results_name = f're{num_to_exclude}'
        self.results['train'][results_name] = []
        self.results['test'][results_name] = []
        if self.is_nt:
            for fn in range(self.n_factors):
                re_train_cs,re_test_cs = self.exclusive_mlp(fn,fn,num_to_exclude)
                self.results['train'][results_name].append(re_train_cs.mean())
                self.results['test'][results_name].append(re_test_cs.mean())

    def predict_unsupervised_vae(self, latents, gts, latents_test, gts_test):
        if self.expname.startswith('exclf'):
            max_testset_size = min(1000000,len(latents_test))
            idxs = np.random.choice(np.arange(len(gts)+len(gts_test)),size=max_testset_size)
            x = np.concatenate([latents,latents_test])[idxs]
            y = np.concatenate([gts,gts_test])[idxs]
            self.set_data(x,y,x,y)
        else:
            self.set_data(latents,gts,latents_test,gts_test)
        if self.verbose:
            print('\nComputing SNC')
        sn_train_cs, sn_test_cs = self.accs_from_alignment(is_single_neurons=True,is_mlp='none')
        self.results['train']['single-neuron'] = self.zs_combo_accs_from_corrects(sn_train_cs)
        self.results['test']['single-neuron'] = self.zs_combo_accs_from_corrects(sn_test_cs)
        if self.verbose:
            print('\nComputing NK')
        self.set_NK_results(num_to_exclude=1)
        if self.expname.startswith('normal'):
            self.set_cross_NK_results(num_to_exclude=1)
        if self.verbose:
            print('\nComputing full MLPs')
        self.set_full_accs_from_classif_func('none')

    def default_full_classif_func_(self, is_mlp): # pvae used not use this,  may not again in future
        train_cs, test_cs = self.accs_from_alignment(is_single_neurons=False,is_mlp=is_mlp)
        train_accs = self.zs_combo_accs_from_corrects(train_cs)
        test_accs = self.zs_combo_accs_from_corrects(test_cs)
        return train_accs, test_accs

    def set_full_accs_from_classif_func(self, classif_func):
        if classif_func == 'none':
            classif_func = self.default_full_classif_func_

        mlp_train_accs, mlp_test_accs = classif_func(is_mlp=True)
        linear_train_accs, linear_test_accs = classif_func(is_mlp=False)
        self.results['train']['mlp'] = mlp_train_accs
        self.results['test']['mlp'] = mlp_test_accs
        self.results['train']['linear'] = linear_train_accs
        self.results['test']['linear'] = linear_test_accs
