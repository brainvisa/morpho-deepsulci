from __future__ import print_function, absolute_import
from capsul.api import Process, Pipeline
import traits.api as traits
import time
import os
import numpy as np
from sklearn.model_selection import KFold
import logging
import socket
import time

class CVprocess(Process):
#    parallel_job_info = {'config_name': 'native',
#                         'nodes_number': 1,
#                         'cpu_per_node': 3}
    def __init__(self):
        super(CVprocess, self).__init__()
        ## input
        self.add_trait('slist_train', traits.ListStr(output=False))
        self.add_trait('slist_test', traits.ListStr(output=False))
        self.add_trait('cv', traits.Str(output=False))
        self.add_trait('n_cv_inner', traits.Int(output=False))
        self.add_trait('data_path', traits.Directory(output=False))
        self.add_trait('sulci_side_list', traits.ListStr(output=False))
        self.add_trait('method', traits.Str(output=False))
        
        ## adhoc
        self.add_trait('n_nn', traits.Int(output=False, optional=True))

        ## patch
        self.add_trait('patch_sizes', traits.ListInt(output=False, optional=True))
        self.add_trait('multipoint', traits.Bool(output=False, optional=True))

        ## deep learning
        self.add_trait('names_fold', traits.Str(output=False, optional=True))
        self.add_trait('lr', traits.Float(output=False, optional=True))
        self.add_trait('momentum', traits.Float(output=False, optional=True))
        self.add_trait('num_filter', traits.Int(output=False, optional=True))
        self.add_trait('optimizer', traits.Str(output=False, optional=True))
        self.add_trait('image_size', traits.ListInt(
            output=False, optional=True))
        self.add_trait('batch_size', traits.Int(output=False, optional=True))
        self.add_trait('data_path_pretrain', traits.Directory(
            output=False, optional=True))
        self.add_trait('early_stopping', traits.Bool(
            output=False, optional=True, default_value=True))
        self.add_trait('intensity', traits.Bool(
            output=False, optional=True, default_value=False))
        self.add_trait('grey_white', traits.Bool(
            output=False, optional=True, default_value=False))
        self.add_trait('bucket', traits.Bool(
            output=False, optional=True, default_value=True))
        self.add_trait('skeleton', traits.Bool(
            output=False, optional=True, default_value=False))
        self.add_trait('data_augmentation', traits.Bool(
            output=False, optional=True, default_value=True))
        self.add_trait('predict_bck', traits.Bool(
            output=False, optional=True, default_value=True))

        ## output
        self.add_trait('result_path', traits.Directory(output=True))

    @property
    def parallel_job_info(self):
        if self.n_cv_inner > 1:
            return {'config_name': 'native',
                    'nodes_number': 1,
                    'cpu_per_node': self.n_cv_inner}
        return {}

    def _run_process(self):

#        log_file_handler = logging.FileHandler(
#            os.path.expandvars("$HOME/logtest_cvprocess_%s" %
#                               (socket.gethostname())))
#        logger = logging.getLogger("CVProcess")
#        logger.setLevel(logging.DEBUG)
#        logger.addHandler(log_file_handler)

        slist = np.asarray(self.slist_train)
        side = 'left' if slist[0][0] == 'L' else 'right'
        dlist = [[s, self.data_path] for s in slist]

        # compute kfolds
#        kf = KFold(n_splits=self.n_cv_inner, random_state=0)
#        strain_list = []
#        stest_list = []
#        for train, test in kf.split(slist):
#            strain_list.append(slist[train])
#            stest_list.append(slist[test])
        strain_list, stest_list = [], []        
        slist_path = '/neurospin/lnao/Panabase/lborne/results/sulci_recognition/morphologist_base2018_new/subjects_list'
        for i in range(self.n_cv_inner):
            strain_list.append(np.array(np.load(os.path.join(
                slist_path, side, self.cv, 'cv_%i' % i, 'subjects_list_train.npy')), dtype=str))
            stest_list.append(np.array(np.load(os.path.join(
                slist_path, side, self.cv, 'cv_%i' % i, 'subjects_list_val.npy')), dtype=str))

        # save outpath        
        side_path = os.path.join(self.result_path, side)
        cvo_path = os.path.join(side_path, self.cv)
        rcvi_path = os.path.join(cvo_path, 'results')
        if not os.path.exists(rcvi_path):
            os.makedirs(rcvi_path)

        if self.method == 'adhoc':
            from lborne.sulci_recognition.method.adhoc import Method_adhoc
            method = Method_adhoc(self.data_path, self.n_nn,
                                  self.sulci_side_list)
            children = []
            for i in range(1, self.n_cv_inner):
                child = os.fork()
                if child:
                    children.append(child)
                else:
                    strain = strain_list[i]
                    stest = stest_list[i]
                    cvi_path = os.path.join(cvo_path, '%s%i'%(self.cv, i))
                    if not os.path.exists(cvi_path):
                        os.makedirs(cvi_path)
                    method.cv_inner(strain, stest, cvi_path, rcvi_path)
                    os._exit(0)
                    return 0

            strain = strain_list[0]
            stest = stest_list[0]
            cvi_path = os.path.join(cvo_path, '%s%i'%(self.cv, 0))
            if not os.path.exists(cvi_path):
                os.makedirs(cvi_path)
            method.cv_inner(strain, stest, cvi_path, rcvi_path)
            for child in children:
                print('wait for child', child, '...')
                sp = os.waitpid(child, 0)
                print('finished', sp)

            method.find_hyperparameters(slist, cvo_path, rcvi_path)
            method.learning(slist)

        elif self.method == 'patch':
            start_time = time.time()
            from lborne.sulci_recognition.method.patch_based_approach import Method_patch
            method = Method_patch(self.data_path, self.sulci_side_list,
                                  multipoint=self.multipoint)
            children = []
            for i in range(1, self.n_cv_inner):
                child = os.fork()
                if child:
                    children.append(child)
                else:
#                    logger.debug('cross validation %i' % i)
                    strain = strain_list[i]
                    stest = stest_list[i]
                    cvi_path = os.path.join(cvo_path, '%s%i'%(self.cv, i))
                    if not os.path.exists(cvi_path):
                        os.makedirs(cvi_path)
                    method.cv_inner(strain, stest, cvi_path, rcvi_path)
#                    logger.debug('end cross validation %i after %i s' % (i, time.time()-start_time))
                    print('end child')
                    os._exit(0)
                    return 0

            
#            logger.debug('cross validation 0')
            strain = strain_list[0]
            stest = stest_list[0]
            cvi_path = os.path.join(cvo_path, '%s%i'%(self.cv, 0))
            if not os.path.exists(cvi_path):
                os.makedirs(cvi_path)
            method.cv_inner(strain, stest, cvi_path, rcvi_path)
#            logger.debug('end cross validation %i after %i s' % (0, time.time()-start_time))
            print('father waiting...')
            for child in children:
                os.waitpid(child, 0)
            print('father stop waiting!')

            method.find_hyperparameters(slist, cvo_path, rcvi_path)
            method.learning(slist)

        elif self.method == 'patch_nn':
            from lborne.sulci_recognition.method.deep_learning import Method_deep
            method = Method_deep(self.data_path, self.names_fold,
                                 self.sulci_side_list,
                                 skeleton=self.bucket,
                                 intensity=self.intensity,
                                 early_stopping=self.early_stopping,
                                 lr=self.lr, batch_size=self.batch_size)
            # cv inner
            for step in range(3):
                for i in range(self.n_cv_inner):
                    strain = strain_list[i]
                    stest = stest_list[i]
                    dlist_train = [d for d in dlist if d[0] in strain]
                    dlist_test = [d for d in dlist if d[0] in stest]

                    cvi_path = os.path.join(cvo_path, '%s%i'%(self.cv, i))
                    if not os.path.exists(cvi_path):
                        os.makedirs(cvi_path)

                    method.cv_inner(dlist_train, dlist_test,
                                    cvi_path, rcvi_path, cvo_path, step)
                method.find_hyperparameters(
                    dlist, cvo_path, rcvi_path, step)

        elif self.method == '3dunet':
            from lborne.sulci_recognition.method.unet3d import Method_3DUNet
            method = Method_3DUNet(self.data_path, self.names_fold,
                                   self.sulci_side_list,
                                   batch_size=self.batch_size,
                                   lr=self.lr, momentum=self.momentum,
                                   opt=self.optimizer,
                                   num_filter=self.num_filter,
                                   intensity=self.intensity,
                                   skeleton=self.skeleton,
                                   bucket=self.bucket,
                                   grey_white=self.grey_white,
                                   data_augmentation=self.data_augmentation,
                                   predict_bck=self.predict_bck)
            # cv inner
            for step in range(3):
                for i in range(self.n_cv_inner):
                    strain = strain_list[i]
                    stest = stest_list[i]
                    dlist_train = [d for d in dlist if d[0] in strain]
                    dlist_test = [d for d in dlist if d[0] in stest]
    
                    cvi_path = os.path.join(cvo_path, '%s%i'%(self.cv, i))
                    if not os.path.exists(cvi_path):
                        os.makedirs(cvi_path)

                    method.cv_inner(dlist_train, dlist_test,
                                    cvi_path, rcvi_path, cvo_path, step)
                method.find_hyperparameters(
                    dlist, cvo_path, rcvi_path, step)

        elif self.method == 'vnet':
            from lborne.sulci_recognition.method.vnet import Method_vnet
            method = Method_vnet(self.data_path, self.names_fold)
            method.learning()
        elif self.method == 'fcn':
            from lborne.sulci_recognition.method.fcn import Method_FCN
            method = Method_FCN(self.data_path, self.names_fold,
                                   image_size=self.image_size,
                                   batch_size=self.batch_size)
        elif self.method == 'patch_fcn':
            from lborne.sulci_recognition.method.patch_fcn import Method_patchFCN
            method = Method_patchFCN(self.data_path, self.names_fold,
                                     image_size=self.image_size,
                                     patch_size=self.patch_size,
                                     batch_size=self.batch_size)
        elif self.method == 'multiscale':
            from lborne.sulci_recognition.method.deep_patch_multiscale import Method_deep
            method = Method_deep(self.data_path, self.names_fold)
        elif self.method == '3dunet_test':
            from lborne.sulci_recognition.method.unet3d import Method_3DUNet
            method = Method_3DUNet(self.data_path, self.names_fold,
                                   self.sulci_side_list,
                                   batch_size=self.batch_size,
                                   lr=self.lr, momentum=self.momentum,
                                   opt=self.optimizer,
                                   num_filter=self.num_filter,
                                   intensity=self.intensity,
                                   skeleton=self.skeleton,
                                   bucket=self.bucket,
                                   grey_white=self.grey_white,
                                   data_augmentation=self.data_augmentation,
                                   predict_bck=self.predict_bck)
            from sklearn.model_selection import train_test_split
            dlist_train, dlist_test = train_test_split(
                dlist, test_size=0.1, random_state=0, shuffle=True)
            method.learning(dlist_train, dlist_test)
        else:
            print('ERROR: Select a correct method!')
            return -1

        start_time = time.time()
        for subject in self.slist_test:
            side = 'left' if subject[0] == 'L' else 'right'
            side_path = os.path.join(self.result_path, side)
            subject_path = os.path.join(side_path, self.cv, subject)
            if not os.path.exists(subject_path):
                os.makedirs(subject_path)
            result_file = os.path.join(side_path, 'results', 'result_%s.csv' % subject)
            method.labelisation(subject, subject_path, result_file)
        print('Labelisation time: %i s.' % (time.time() - start_time))
        
        return 0


class CVpipeline(Pipeline):

    def pipeline_definition(self):
        self.add_iterative_process(
            'labelisation', CVprocess(),
            iterative_plugs=['slist_train', 'slist_test', 'cv'])

        # iterative plugs
        self.export_parameter('labelisation', 'slist_train', 'slist_train_list')
        self.export_parameter('labelisation', 'slist_test', 'slist_test_list')
        self.export_parameter('labelisation', 'cv', 'cv_list')

        ## adhoc - optional
        self.export_parameter('labelisation', 'n_nn', 'n_nn')
        self.export_parameter('labelisation', 'multipoint', 'multipoint')

        ## deep learning - optional
        self.export_parameter('labelisation', 'patch_sizes', 'patch_sizes')
        self.export_parameter('labelisation', 'lr', 'lr')
        self.export_parameter('labelisation', 'momentum', 'momentum')
        self.export_parameter('labelisation', 'num_filter', 'num_filter')
        self.export_parameter('labelisation', 'optimizer', 'optimizer')
        self.export_parameter('labelisation', 'names_fold', 'names_fold')
        self.export_parameter('labelisation', 'image_size', 'image_size')
        self.export_parameter('labelisation', 'batch_size', 'batch_size')
        self.export_parameter('labelisation', 'data_path_pretrain',
                              'data_path_pretrain')
        self.export_parameter('labelisation', 'early_stopping',
                              'early_stopping')
        self.export_parameter('labelisation', 'bucket', 'bucket')
        self.export_parameter('labelisation', 'skeleton', 'skeleton')
        self.export_parameter('labelisation', 'grey_white', 'grey_white')
        self.export_parameter('labelisation', 'intensity', 'intensity')
        self.export_parameter('labelisation', 'data_augmentation',
                              'data_augmentation')
        self.export_parameter('labelisation', 'predict_bck', 'predict_bck')


